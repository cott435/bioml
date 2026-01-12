import subprocess
from os import path
import pandas as pd

root_dir = "./data_files"

def get_tdc_epitope(name, split=False, file_dir=root_dir):
    from tdc.single_pred import Epitope
    data = Epitope(name=name,  path=file_dir)
    return data.get_split() if split else data.get_data()

def get_tdc_ppi(name, split=False, neg_frac=1, file_dir=root_dir):
    from tdc.multi_pred import PPI
    data = PPI(name=name, path=file_dir).neg_sample(frac=neg_frac)
    return data.get_split() if split else data.get_data()

def df_save(data, name, file_dir=root_dir):
    file_path = path.join(file_dir, name+'.parquet')
    data.to_parquet(file_path, engine='pyarrow')

def df_load(name, file_dir=root_dir):
    file_path = path.join(file_dir, name+'.parquet')
    return pd.read_parquet(file_path, engine='pyarrow')

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

def cluster_sequences(data, data_id, cluster_coef=0.4, force=False, sequence_col='Antigen', id_col='Antigen_ID'):
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    data=data.copy()
    data[id_col] = data[id_col].str.replace(' ', '')
    fasta_path = f"./raw_data_files/{data_id}.fasta"
    if force or not path.exists(fasta_path):
        records = []
        for idx, row in data.iterrows():
            seq = row[sequence_col]
            antigen_id = row.get(id_col, f"antigen_{idx}")
            records.append(SeqRecord(Seq(seq), id=antigen_id, description=""))
        SeqIO.write(records, fasta_path, "fasta")
    output_fasta = f"./raw_data_files/{data_id}_clustered.fasta"
    clstr_file = output_fasta + ".clstr"
    if force or not path.exists(clstr_file):
        if not force:
            print("Clustering file not found, generating new")
        subprocess.run([
            "cd-hit", "-i", fasta_path, "-o", output_fasta,
            "-c", str(cluster_coef), "-n", "2", "-M", "16000", "-T", "8"
        ], check=True)
    else:
        print("Clustering file found, parsing in data")

    antigen_ids = [row.get(id_col, f"antigen_{i}") for i, row in data.iterrows()]
    cluster_map = parse_cd_hit_clstr(clstr_file, set(antigen_ids))
    if not cluster_map:
        if force:
            raise Exception("Error while clustering")
        else:
            print("No results found in old cluster file, generating new")
            return cluster_sequences(data, data_id, cluster_coef=cluster_coef, force=True,
                              sequence_col=sequence_col, id_col=id_col)

    data['cluster_id'] = data[id_col].map(cluster_map)
    return data

