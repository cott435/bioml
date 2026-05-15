import os
import shutil
import subprocess
import pandas as pd
from .parse import data_dir
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Iterable, List
from pathlib import Path
from dataclasses import asdict
import csv
from tensordict import TensorDict


def resolve_data_dir(save_dir) -> Path:
    base = Path(save_dir)
    if base.is_absolute():
        return base
    if base.exists():
        return base
    alt = Path.cwd() / "data" / base
    if alt.exists():
        return alt
    return base


def is_colab() -> bool:
    return os.environ.get("RUNTIME_ENV", "").strip().lower() == "colab"


COLAB_LOCAL_ROOT = Path("/content/local_data")
_COLAB_DRIVE_ROOTS = (
    "/content/drive/MyDrive/",
    "/content/drive/Shareddrives/",
    "/content/drive/",
)


def colab_local_path(drive_path) -> Path:
    """Mirror a Drive (FUSE) path to a fast local working path under /content/local_data."""
    drive_path = Path(drive_path)
    s = str(drive_path)
    for root in _COLAB_DRIVE_ROOTS:
        if s.startswith(root):
            return COLAB_LOCAL_ROOT / s[len(root):]
    return COLAB_LOCAL_ROOT / drive_path.name


def copy_path(src, dst, verbose: bool = True) -> bool:
    """Copy a file or directory tree from src to dst, replacing dst if it exists.

    Returns True if a copy was performed, False if src does not exist.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Copying {src} -> {dst}")
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)
    return True


def make_sequence_fasta(
    sequences,
    save_dir,
    force=False
):
    save_dir = resolve_data_dir(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = save_dir / "sequences.fasta"
    if force or not fasta_path.exists():
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        from Bio import SeqIO
        records = []
        for id_, seq in sequences.items():
            records.append(SeqRecord(Seq(seq), id=id_, description=""))
        SeqIO.write(records, fasta_path, "fasta")
        print(f'Wrote sequences to {fasta_path}')
    else:
        print("Fasta file already exists; set force=True to overwrite")
    return fasta_path


def df_save(data, name='dataframe', file_dir=data_dir, force=False):
    file_dir = resolve_data_dir(file_dir)
    file_dir.mkdir(parents=True, exist_ok=True)
    file_path = file_dir / f'{name}.parquet'
    if force or not file_path.exists():
        data.to_parquet(file_path, engine='pyarrow')
        print(f'Saved dataframe to {file_path}')


def df_load(data_name, file_dir=data_dir):
    file_dir = resolve_data_dir(file_dir)
    file_path = file_dir / f'{data_name}.parquet'
    return pd.read_parquet(file_path, engine='pyarrow')


def _cd_hit_word_size(cluster_coef: float) -> str:
    if cluster_coef >= 0.7:
        return "5"
    if cluster_coef >= 0.6:
        return "4"
    if cluster_coef >= 0.5:
        return "3"
    return "2"


def cluster_fasta(fasta_path, cluster_coef=0.5, force=False):
    fasta_path = Path(fasta_path)
    base = fasta_path.parent
    output_prefix = base / f"clustered_{int(cluster_coef*100)}_sequences"
    clstr_file = output_prefix.with_suffix(".clstr")
    if force or not clstr_file.exists():
        if not force:
            print("Clustering file not found, generating new")
        subprocess.run([
            "cd-hit", "-i", str(fasta_path), "-o", str(output_prefix),
            "-c", str(cluster_coef), "-n", _cd_hit_word_size(cluster_coef), "-M", "16000", "-T", "8"
        ], check=True)
    else:
        print("Clustering file found, parsing in data")
    return clstr_file


def parse_cd_hit_clstr(clstr_file, seq_ids_order, allow_ungrouped=False):
    cluster_map = {}
    cluster_id = 0
    seq_ids_order = list(seq_ids_order)
    seq_ids_set = set(seq_ids_order)
    with open(clstr_file) as handle:
        for line in handle:
            if line.startswith(">Cluster"):
                cluster_id = int(line.split()[-1])
            else:
                seq_id = line.split(">")[1].split("...")[0]
                if seq_id in seq_ids_set:  # Map back to original order
                    cluster_map[seq_id] = cluster_id
    if not allow_ungrouped:
        ungrouped = [id_ for id_ in seq_ids_order if id_ not in cluster_map]
        if len(ungrouped) > 0:
            start_group_id = max(cluster_map.values()) + 1 if cluster_map else 0
            cluster_map.update({id_: start_group_id+i for i, id_ in enumerate(ungrouped)})
    return cluster_map


def add_clusters_to_df(df, clstr_path):
    data = df.copy()
    ids = df['ID'].replace(" ", "")
    cluster_map = parse_cd_hit_clstr(clstr_path, set(ids))
    if not cluster_map:
        raise Exception("Error while loading cluster file")
    data['group_id'] = ids.map(cluster_map)
    return data

def esm_extract_sequences(
    model_name: str,
    fasta_path: str,
    output_dir: str,
    toks_per_batch: int = 8000,
    repr_layers: Iterable[int] = (-1,),
    include: Iterable[str] = ("mean","per_tok")):
    cmd = [
        "esm-extract",
        model_name,
        fasta_path,
        output_dir,
        "--toks_per_batch", str(toks_per_batch),
        "--include", *include,
    ]
    if repr_layers:
        cmd.extend(["--repr_layers", *map(str, repr_layers)])
    subprocess.run(cmd, check=True)

def pad_collate_fn(batch):
    """
    batch: List of (x, y)
        x: Tensor[T, ...]
        y: Tensor[T] or Tensor[T, ...]
    """
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)

    x_padded = pad_sequence(xs, batch_first=True)
    y_padded = pad_sequence(ys, batch_first=True)

    mask = torch.arange(
        x_padded.size(1),
        device=lengths.device
    )[None, :] < lengths[:, None]

    return x_padded, y_padded, mask


def bucket_collate_fn(batch, rel_thresh=0.15):
    """
    sort greedily within given percent
    """
    sorted_batch = sorted(batch, key=lambda x: len(x[1]))
    lengths = torch.tensor([y.shape[0] for _, y in sorted_batch], dtype=torch.long)

    sub_batches = []
    i = 0
    while i < len(lengths):
        current_len = lengths[i]
        group_items = []

        # Add sequences that are within threshold of current max
        while i < len(lengths) and lengths[i] <= current_len * (1 + rel_thresh):
            x, y = sorted_batch[i]
            group_items.append((x, y))
            i += 1

        if not group_items:
            break

        xs, ys = zip(*group_items)
        x_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
        y_padded = pad_sequence(ys, batch_first=True, padding_value=0.0)  # adjust value if needed

        lengths_tensor = torch.tensor([y.shape[0] for y in ys], dtype=torch.long)
        max_len = y_padded.shape[1]
        mask = (
                torch.arange(max_len, device=y_padded.device)[None, :]
                < lengths_tensor[:, None]
        )

        sub_batches.append((x_padded, y_padded, mask))

    return sub_batches


import time
from contextlib import contextmanager

@contextmanager
def timer(label="elapsed"):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{label}: {end - start:.6f} s")


def save_params_as_csv(file_dir: Path, params):
    file_dir.mkdir(parents=True, exist_ok=True)
    filepath = file_dir / 'params.csv'
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Parameter', 'Value'])
        for key, value in params.items():
            writer.writerow([key, value])
