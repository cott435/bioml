import subprocess
import pandas as pd
from .parse import data_dir
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Iterable, List
from pathlib import Path
from dataclasses import asdict
import csv

def make_sequence_fasta(
    sequences,
    save_dir,
    force=False
):
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
    file_path = file_dir / f'{name}.parquet'
    if force or not file_path.exists():
        data.to_parquet(file_path, engine='pyarrow')
        print(f'Saved dataframe to {file_path}')


def df_load(data_name, file_dir=data_dir):
    file_path = file_dir / f'{data_name}.parquet'
    return pd.read_parquet(file_path, engine='pyarrow')


def cluster_fasta(fasta_path, cluster_coef=0.5, force=False):
    base = fasta_path.parent
    output_prefix = base / f"clustered_{int(cluster_coef*100)}_sequences"
    clstr_file = output_prefix.with_suffix(".clstr")
    if force or not clstr_file.exists():
        if not force:
            print("Clustering file not found, generating new")
        subprocess.run([
            "cd-hit", "-i", fasta_path, "-o", output_prefix,
            "-c", str(cluster_coef), "-n", "2", "-M", "16000", "-T", "8"
        ], check=True)
    else:
        print("Clustering file found, parsing in data")
    return clstr_file


def parse_cd_hit_clstr(clstr_file, seq_ids_order, allow_ungrouped=False):
    cluster_map = {}
    cluster_id = 0
    for line in open(clstr_file):
        if line.startswith(">Cluster"):
            cluster_id = int(line.split()[-1])
        else:
            seq_id = line.split(">")[1].split("...")[0]
            if seq_id in seq_ids_order:  # Map back to original order
                cluster_map[seq_id] = cluster_id
    if not allow_ungrouped:
        ungrouped = [id_ for id_ in seq_ids_order if id_ not in cluster_map]
        if len(ungrouped) > 0:
            start_group_id = max(cluster_map.values()) + 1
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
