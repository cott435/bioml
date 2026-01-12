import torch
from esm import Alphabet
import os
import subprocess
from typing import List, Iterable, Optional
from data_parse import root_dir

available_tok = {'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M',
                 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-'}

def sanitize_sequence(seq):
    return ''.join(c if c in available_tok else 'X' for c in seq)

def esm_extract_sequences(
    sequences: List[str],
    data_name: str,
    model_name: str,
    root_dir=root_dir,
    batch_size: int = 1,
    repr_layers: Iterable[int] = (-1,),
    include: Iterable[str] = ("mean",),
    overwrite: bool = False,
    fasta_prefix: str = "seq",
) -> str:
    """
    Precompute ESM embeddings for a list of protein sequences.

    Parameters
    ----------
    sequences : list[str]
        Raw protein sequences
    data_name : str
        Dataset identifier (used for directory naming)
    model_name : str
        ESM model name (e.g. esm2_t33_650M_UR50D)
    batch_size : int
        Batch size for esm extract
    repr_layers : iterable[int]
        Transformer layers to extract (use -1 for last layer)
    include : iterable[str]
        What representations to save (mean, per_tok, cls, bos, eos)
    overwrite : bool
        If True, re-run extraction even if outputs exist
    fasta_prefix : str
        Prefix for FASTA sequence labels

    Returns
    -------
    output_dir : str
        Directory containing extracted .pt embeddings
    """

    output_dir = os.path.join(root_dir, data_name, model_name)
    os.makedirs(output_dir, exist_ok=True)

    fasta_path = os.path.join(output_dir, f"{data_name}.fasta")

    # Skip if embeddings already exist
    if not overwrite and any(f.endswith(".pt") for f in os.listdir(output_dir)):
        return output_dir

    # Sanitize sequences
    clean_sequences = [sanitize_sequence(seq) for seq in sequences]

    # Write FASTA
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(clean_sequences):
            f.write(f">{fasta_prefix}_{i}\n")
            f.write(seq + "\n")

    # Build esm extract command
    cmd = [
        "esm", "extract",
        model_name,
        fasta_path,
        output_dir,
        "--batch_size", str(batch_size),
        "--include", *include,
    ]

    # Handle repr layers
    if repr_layers:
        cmd.extend(["--repr_layers", *map(str, repr_layers)])

    # Run extraction
    subprocess.run(cmd, check=True)

    return output_dir


class ESMBatcher:

    def __init__(self, alphabet: Alphabet):
        self.alphabet = alphabet
        self.batch_converter = self.alphabet.get_batch_converter()

    def collate_batch(self, batch: List[str]):
        sequences = [("", sanitize_sequence(b)) for b in batch]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
        return batch_tokens

class EpitopeBatcher(ESMBatcher):

    def collate_batch(self, batch):
        sequences, labels_lists = [b[0] for b in batch], [b[1] for b in batch]
        batch_tokens = super().collate_batch(sequences)
        reduce = int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos)
        labels = torch.zeros_like(batch_tokens)[:, reduce:]
        rows = torch.repeat_interleave(
            torch.arange(labels.shape[0]),
            torch.tensor([len(x) for x in labels_lists])
        )
        cols = torch.tensor([i for sub in labels_lists for i in sub])
        labels[rows, cols] = 1
        return batch_tokens, labels


