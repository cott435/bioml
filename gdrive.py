import os
import pickle
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import threading


def collect_directories(files: list[Path]) -> list[Path]:
    dirs = set()
    for path in files:
        for i in range(1, len(path.parts)):
            dirs.add(Path(*path.parts[:i]))
    return sorted(dirs, key=lambda p: len(p.parts))

class GoogleDriveSync:
    """
    Async-capable class for syncing Python files to Google Drive.
    Uploads run concurrently using ThreadPoolExecutor for speed.
    Folder creation/listing remains sequential (usually fast).

    Features:
    - Uses git ls-files as source of truth (tracked files only)
    - Preserves directory structure under the given Drive folder
    - Creates directories as needed
    - Overwrites remote files when content differs
    - Hash-based skip (MD5) for fast no-op sync
    - Progress bar (tqdm) with fallback
    """

    SCOPES = ['https://www.googleapis.com/auth/drive']
    DRIVE_FOLDER_MIME = 'application/vnd.google-apps.folder'

    def __init__(
        self,
        credentials_file: str = 'credentials.json',
        token_file: str = 'token.pickle',
        extensions: tuple = ('.py',),
        max_concurrent_uploads: int = 10,
        hash_skip: bool = True,
        show_progress: bool = True,
        md5_chunk_size: int = 1024 * 1024,  # 1MB chunks
    ):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.extensions = extensions
        self.max_concurrent = max_concurrent_uploads
        self.hash_skip = hash_skip
        self.show_progress = show_progress
        self.md5_chunk_size = md5_chunk_size
        self._thread_local = threading.local()

        self._folder_cache: Dict[Tuple[str, str], str] = {}

    def _get_drive_service(self):
        creds = None
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES
                )
                creds = flow.run_local_server(port=0)

            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)

        return build('drive', 'v3', credentials=creds)

    def _get_thread_service(self):
        if not hasattr(self._thread_local, "service"):
            self._thread_local.service = self._get_drive_service()
        return self._thread_local.service

    def list_tracked_files(self) -> List[Path]:
        """
        Returns all tracked files matching extensions using git ls-files.
        """
        patterns = [f'*{ext}' for ext in self.extensions]
        cmd = ['git', 'ls-files', *patterns]
        output = subprocess.check_output(cmd, text=True)
        return [Path(line.strip()) for line in output.splitlines() if line.strip()]

    def list_files_in_dir(self, base_dir, rm_base=False) -> List[Path]:
        base = Path(base_dir)
        paths = []
        for path in base.rglob('*'):
            if path.is_file() and path.suffix in self.extensions:
                path = path.relative_to(base) if rm_base else path
                paths.append(path)
        return paths

    def _md5_file(self, path: Path) -> str:
        """
        Compute MD5 of a file in chunks.
        """
        h = hashlib.md5()
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(self.md5_chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _build_directory_map(self, directories, root_folder_id):
        dir_map = {Path(): root_folder_id}

        for d in directories:
            parent = dir_map[d.parent]
            folder_id = self._find_or_create_folder(d.name, parent)
            dir_map[d] = folder_id

        return dir_map

    def _prepare_directories(self, files, root_folder_id):
        dirs = collect_directories(files)
        return self._build_directory_map(dirs, root_folder_id)

    def _find_or_create_folder(self, name: str, parent_id: str) -> str:
        """
        Finds or creates a folder under parent_id. Cached for efficiency.
        """
        cache_key = (parent_id, name)
        if cache_key in self._folder_cache:
            return self._folder_cache[cache_key]

        query = (
            f"name='{name}' and "
            f"mimeType='{self.DRIVE_FOLDER_MIME}' and "
            f"'{parent_id}' in parents and trashed=false"
        )

        results = self._get_thread_service().files().list(
            q=query,
            fields="files(id, name)",
            pageSize=1
        ).execute()

        files = results.get('files', [])
        if files:
            folder_id = files[0]['id']
        else:
            folder_metadata = {
                'name': name,
                'mimeType': self.DRIVE_FOLDER_MIME,
                'parents': [parent_id]
            }
            folder = self._get_thread_service().files().create(
                body=folder_metadata,
                fields='id'
            ).execute()
            folder_id = folder['id']

        self._folder_cache[cache_key] = folder_id
        return folder_id

    def _find_existing_file_metadata(self, name: str, parent_id: str) -> Optional[dict]:
        """
        Returns metadata dict for the first matching file in the folder, else None.
        Includes md5Checksum when available (non-Google Docs formats).
        """
        query = (
            f"name='{name}' and "
            f"'{parent_id}' in parents and trashed=false"
        )
        results = self._get_thread_service().files().list(
            q=query,
            fields="files(id, name, mimeType, md5Checksum, size)",
            pageSize=1
        ).execute()

        files = results.get('files', [])
        return files[0] if files else None

    def _upload_file(self, local_path: Path, dir_map) -> str:
        """
        Upload a single file (overwrite if exists and differs).
        Returns a status string: 'uploaded', 'skipped', or raises on error.
        """
        parent_id = dir_map[local_path.parent]
        meta = self._find_existing_file_metadata(local_path.name, parent_id)

        # Hash-based skip (only when remote md5Checksum exists)
        if self.hash_skip and meta and meta.get('md5Checksum'):
            local_md5 = self._md5_file(local_path)
            if local_md5 == meta['md5Checksum']:
                return "skipped"

        media = MediaFileUpload(str(local_path), resumable=False)

        if meta:  # Overwrite
            self._get_thread_service().files().update(
                fileId=meta['id'],
                media_body=media
            ).execute()
            return str(local_path)
        else: # Create new
            metadata = {'name': local_path.name, 'parents': [parent_id]}
            self._get_thread_service().files().create(
                body=metadata,
                media_body=media,
                fields='id'
            ).execute()
            return "uploaded"

    def sync(self, root_folder_id: str, local_base=None) -> dict:
        """
        Sync all tracked files to the given Google Drive folder.
        Returns a summary dict.
        """
        files = self.list_files_in_dir(local_base)
        dir_map = self._prepare_directories(files, root_folder_id)
        total = len(files)
        if total == 0:
            print("No matching tracked files found.")
            return {"total": 0, "uploaded": 0, "skipped": 0, "failed": 0}

        uploaded, skipped, overwritten, failed = 0,0,0,0
        overwritten_files = []
        pbar = tqdm(total=total, desc="Syncing", unit="file")

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_path = {
                executor.submit(self._upload_file, path, dir_map): path
                for path in files
            }

            done_count = 0
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    status = future.result()
                    if status == "uploaded":
                        uploaded += 1
                    elif status == "skipped":
                        skipped += 1
                    else:
                        overwritten += 1
                        overwritten_files.append(status)
                except Exception as e:
                    failed += 1
                    print(f"[FAILED] {path}: {e}")

                done_count += 1
                pbar.update(1)

        if pbar:
            pbar.close()

        summary = {"total": total, "uploaded": uploaded, "overwritten": overwritten, "skipped": skipped, "failed": failed}
        print(f"Done: {summary}")
        if overwritten_files:
            print(f"Overwritten: {overwritten_files}")
        return summary

if __name__ == '__main__':
    syncer = GoogleDriveSync()
    code_folder = '1XXwUka6h3JpfZpOa6aEwbbs-Nm73qzKY'
    syncer.sync(code_folder, local_base='proteins')
