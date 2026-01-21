import os
import pickle
from typing import Optional, List
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


class GoogleDriveSync:
    """
    Async-capable class for syncing Python files to Google Drive.
    Uploads run concurrently using ThreadPoolExecutor for speed.
    Folder creation/listing remains sequential (usually fast).
    """

    SCOPES = ['https://www.googleapis.com/auth/drive']

    def __init__(
            self,
            credentials_file: str = 'credentials.json',
            token_file: str = 'token.pickle',
            extensions: tuple = ('.py',),
            max_concurrent_uploads: int = 10,  # Tune: 5–20 depending on your connection
    ):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.extensions = extensions
        self.max_concurrent = max_concurrent_uploads
        self.service = self._get_drive_service()

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

    def _get_or_create_folder(self, folder_name: str, parent_id: Optional[str] = None) -> str:
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        else:
            query += " and 'root' in parents"

        response = self.service.files().list(
            q=query, fields='files(id, name)', spaces='drive', corpora='user'
        ).execute()

        files = response.get('files', [])
        if files:
            return files[0]['id']

        folder_metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
        if parent_id:
            folder_metadata['parents'] = [parent_id]

        folder = self.service.files().create(body=folder_metadata, fields='id').execute()
        print(f"Created folder: {folder_name} (ID: {folder.get('id')})")
        return folder.get('id')

    def _upload_or_update_file(self, local_path: str, file_name: str, parent_id: str) -> str:
        query = f"name='{file_name}' and '{parent_id}' in parents and trashed=false"
        response = self.service.files().list(
            q=query, fields='files(id, name)', spaces='drive'
        ).execute()

        existing = response.get('files', [])
        media = MediaFileUpload(local_path, resumable=True)

        if existing:
            file_id = existing[0]['id']
            updated = self.service.files().update(
                fileId=file_id, media_body=media, fields='id'
            ).execute()
            print(f"Overwritten: {file_name} (ID: {file_id})")
            return file_id
        else:
            metadata = {'name': file_name, 'parents': [parent_id]}
            file = self.service.files().create(
                body=metadata, media_body=media, fields='id'
            ).execute()
            print(f"Uploaded: {file_name} (ID: {file.get('id')})")
            return file.get('id')

    async def sync_python_files(
            self,
            input_dir: str,
            target_drive_folder_id: Optional[str] = None
    ) -> None:
        """
        Asynchronously sync .py files to Google Drive.
        Folders created sequentially; uploads run concurrently.
        """
        input_dir = os.path.abspath(input_dir)
        if not os.path.isdir(input_dir):
            raise ValueError(f"Input directory not found: {input_dir}")

        root_folder_name = os.path.basename(input_dir.rstrip(os.sep))
        current_parent_id = target_drive_folder_id or self._get_or_create_folder(root_folder_name)
        print(f"Syncing into Drive folder: {root_folder_name if not target_drive_folder_id else '[custom]'} "
              f"(ID: {current_parent_id})")

        upload_tasks = []  # list of (local_path, file_name, parent_id)

        for root, dirs, files in os.walk(input_dir):
            if '.venv' in root or 'venv' in root or '__pycache__' in root or 'raw_data_files' in root:
                continue
            dirs[:] = [d for d in dirs if d not in {'.venv', 'venv', '__pycache__', '.git', '.idea'}]

            rel_path = os.path.relpath(root, input_dir)
            drive_parent_id = current_parent_id

            if rel_path != '.':
                for part in rel_path.split(os.sep):
                    if part:
                        drive_parent_id = self._get_or_create_folder(part, drive_parent_id)

            for file_name in files:
                if file_name.lower().endswith(self.extensions):
                    local_path = os.path.join(root, file_name)
                    upload_tasks.append((local_path, file_name, drive_parent_id))

        if not upload_tasks:
            print("No .py files found to sync.")
            return

        print(f"Found {len(upload_tasks)} .py file(s) to sync. Starting concurrent uploads "
              f"(max {self.max_concurrent} at a time)...")

        loop = asyncio.get_running_loop()

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_file = {
                loop.run_in_executor(
                    executor,
                    lambda p=local_path, n=file_name, pid=parent_id:
                    self._upload_or_update_file(p, n, pid)
                ): (local_path, file_name)
                for local_path, file_name, parent_id in upload_tasks
            }

            for future in as_completed(future_to_file):
                try:
                    file_id = future.result()  # blocks until done
                    local_path, _ = future_to_file[future]
                    # You could add tqdm progress here if desired
                except Exception as e:
                    local_path, file_name = future_to_file[future]
                    print(f"Error uploading {file_name}: {e}")

        print("\nSync complete! All .py files uploaded/overwritten.")


# ────────────────────────────────────────────────
# Example usage (run with asyncio)
# ────────────────────────────────────────────────

if __name__ == '__main__':
    import asyncio
    from pathlib import Path

    async def main():
        drive_sync = GoogleDriveSync(max_concurrent_uploads=12)  # adjust as needed

        LOCAL_PROJECT_DIR = Path.cwd()
        TARGET_FOLDER_ID = '1QCxeP3GzBP3X0j3SGOqr6f8jg410VsCc'# '1tYlqhfuB5pkCgSndijNAUBEuQ2GKEY5m'

        await drive_sync.sync_python_files(
            input_dir=LOCAL_PROJECT_DIR,
            target_drive_folder_id=TARGET_FOLDER_ID
        )


    asyncio.run(main())