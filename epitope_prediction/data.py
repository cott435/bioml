from torch.utils.data import Dataset
from tdc.single_pred import Epitope
from tdc.multi_pred import PPI
import numpy as np
import torch
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import subprocess
from os import path

data_dir = "./raw_data_files"

def get_tdc_epitope(name, split=False, file_dir=data_dir):
    data = Epitope(name=name,  path=file_dir)
    return data.get_split() if split else data.get_data()

def get_tdc_ppi(name, split=False, neg_frac=1, file_dir=data_dir):
    data = PPI(name=name, path=file_dir).neg_sample(frac=neg_frac)
    return data.get_split() if split else data.get_data()

def plot_viz(sequences, bind_idx, max_seq_len=None):
    seq_len = sequences.apply(lambda x: len(x))
    active_sites = bind_idx.apply(lambda x: len(x))
    if max_seq_len:
        active_sites = active_sites[seq_len < max_seq_len]
        seq_len = seq_len[seq_len < max_seq_len]
    ratio = active_sites / seq_len

    fig, axs = plt.subplots(3, 1)
    axs[0].hist(seq_len, bins=100)
    axs[0].set_title("Hist of Sequence Lengths")
    axs[1].hist(active_sites, bins=100)
    axs[1].set_title("Hist of Active Sites")
    axs[2].hist(ratio, bins=100)
    axs[2].set_title("Hist of Active Site Ratio")
    fig.tight_layout()

    fig, axs = plt.subplots(2, 1)
    axs[0].scatter(seq_len, ratio)
    axs[0].set_ylabel("Active Site Ratio")
    axs[1].scatter(seq_len, active_sites)
    axs[1].set_xlabel("Sequence Length")
    axs[1].set_ylabel("Active Sites")


class EpitopeDataset(Dataset):
    def __init__(self, data, x_col='X', y_col='Y'):
        self.sequences = data[x_col]
        self.binding_idx = data[y_col]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences.iloc[idx], self.binding_idx.iloc[idx]

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

class SequenceEpitopeTokenizer:

    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.batch_converter = self.alphabet.get_batch_converter()
        self.available_tok = {'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X',
             'B', 'U', 'Z', 'O', '.', '-'}

    def sanitize_sequence(self, seq):
        return ''.join(c if c in self.available_tok else 'X' for c in seq)

    def collate_batch(self, batch):
        sequences = [("", self.sanitize_sequence(b[0])) for b in batch]
        labels_lists = [b[1] for b in batch]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
        labels = torch.zeros_like(batch_tokens)
        start = int(self.alphabet.prepend_bos)
        rows = torch.repeat_interleave(
            torch.arange(labels.shape[0]),
            torch.tensor([len(x) for x in labels_lists])
        )
        cols = torch.tensor([i for sub in labels_lists for i in sub]) + start
        labels[rows, cols] = 1
        return batch_tokens, labels

