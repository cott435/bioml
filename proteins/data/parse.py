import subprocess
from os import path
import pandas as pd
from esm.constants import proteinseq_toks

data_dir = "./data_files"

def sanitize_sequence(seq):
    return ''.join(c if c in proteinseq_toks['toks'] else 'X' for c in seq)

def get_tdc_epitope(name, file_dir=data_dir):
    from tdc.single_pred import Epitope
    mapping = {'Antigen_ID': 'ID', 'Antigen': 'X'}
    data = Epitope(name=name,  path=file_dir).get_data().rename(columns=mapping)
    data['ID'] = data['ID'].str.replace(" ", "")
    return data

def get_tdc_ppi(name, split=False, neg_frac=1, file_dir=data_dir):
    from tdc.multi_pred import PPI
    data = PPI(name=name, path=file_dir).neg_sample(frac=neg_frac)
    return data.get_split() if split else data.get_data()

def make_sequence_fasta(
    sequences,
    ids,
    file_dir=data_dir,
    force=False
):
    fasta_path = path.join(file_dir, "sequences.fasta")
    if force or not path.exists(fasta_path):
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        from Bio import SeqIO
        records = []
        for seq, sid in zip(sequences, ids):
            seq = sanitize_sequence(seq)
            records.append(SeqRecord(Seq(seq), id=sid, description=""))
        SeqIO.write(records, fasta_path, "fasta")
        print(f'Wrote sequences to {fasta_path}')
    else:
        print("Fasta file already exists; set force=True to overwrite")
    return fasta_path


def df_save(data, name='dataframe', file_dir=data_dir, force=False):
    file_path = path.join(file_dir, f'{name}.parquet')
    if force or not path.exists(file_path):
        data.to_parquet(file_path, engine='pyarrow')
        print(f'Saved dataframe to {file_path}')


def df_load(data_name, file_dir=data_dir):
    file_path = path.join(file_dir, f'{data_name}.parquet')
    return pd.read_parquet(file_path, engine='pyarrow')

def cluster_fasta(fasta_path, cluster_coef=0.5, force=False):
    base = path.dirname(fasta_path)
    output_prefix = path.join(base, "clustered_sequences")
    clstr_file = output_prefix + ".clstr"
    if force or not path.exists(clstr_file):
        if not force:
            print("Clustering file not found, generating new")
        subprocess.run([
            "cd-hit", "-i", fasta_path, "-o", output_prefix,
            "-c", str(cluster_coef), "-n", "2", "-M", "16000", "-T", "8"
        ], check=True)
    else:
        print("Clustering file found, parsing in data")
    return clstr_file

def parse_cd_hit_clstr(clstr_file, seq_ids_order):
    cluster_map = {}
    cluster_id = 0
    for line in open(clstr_file):
        if line.startswith(">Cluster"):
            cluster_id = int(line.split()[-1])
        else:
            seq_id = line.split(">")[1].split("...")[0]
            if seq_id in seq_ids_order:  # Map back to original order
                cluster_map[seq_id] = cluster_id
    return cluster_map

def add_clusters_to_df(df, clstr_path):
    data = df.copy()
    ids = df['ID'].replace(" ", "")
    cluster_map = parse_cd_hit_clstr(clstr_path, set(ids))
    if not cluster_map:
        raise Exception("Error while loading cluster file")
    data['cluster_id'] = ids.map(cluster_map)
    return data
