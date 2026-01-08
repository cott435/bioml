from torch.utils.data import Dataset
from tdc.single_pred import Epitope
import numpy as np
import torch
import matplotlib.pyplot as plt

common_aa = {'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X',
             'B', 'U', 'Z', 'O', '.', '-'}

def get_tdc_epitope(name, split=True, file_dir="./raw_data_files"):
    data = Epitope(name=name,  path=file_dir)
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

def data_to_list(df, esm=True, max_seq_len=None):
    skipped=0
    all_sequences = df['Antigen'].tolist()
    labels, sequences = [], []
    for y, seq in zip(df['Y'], all_sequences):
        seq_len = len(seq)
        if max_seq_len is not None and seq_len > max_seq_len:
            skipped+=1
            continue
        label = [0] * seq_len
        for idx in y:
            if 0 <= idx < seq_len:
                label[idx] = 1
        labels.append(np.array(label))
        sequences.append(seq)
    if esm:
        def sanitize(seq, allowed):
            return ''.join(c if c in allowed else 'X' for c in seq)
        sequences = [("", sanitize(s, common_aa)) for s in sequences]
    if skipped>0:
        print("skipped {} sequences".format(skipped))
    return sequences, labels

def alphabet_batch_convert(batch_converter, sequences, labels_list):
    esm2_batch_labels, esm2_batch_strs, esm2_batch_tokens = batch_converter(sequences)
    labels = torch.zeros_like(esm2_batch_tokens)
    start = int(batch_converter.alphabet.prepend_bos)
    for i, l in enumerate(labels_list):
        labels[i, start:len(l) + start] = torch.tensor(l)
    return esm2_batch_tokens, labels, esm2_batch_strs

class EpitopeDataset(Dataset):
    def __init__(self, sequences, binding_idx):
        assert len(sequences) == len(binding_idx)
        self.sequences = sequences
        self.binding_idx = binding_idx

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.binding_idx[idx]
