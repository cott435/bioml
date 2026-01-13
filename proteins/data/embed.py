import torch
from esm import Alphabet
import os
import subprocess
from typing import List, Iterable, Optional
from .parse import data_dir

available_tok = {'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M',
                 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-'}

def sanitize_sequence(seq):
    return ''.join(c if c in available_tok else 'X' for c in seq)

def esm_extract_sequences(
    sequences: List[str],
    data_name: str,
    model_name: str,
    root_dir=data_dir,
    batch_size: int = 1,
    repr_layers: Iterable[int] = (-1,),
    include: Iterable[str] = ("mean",),
    overwrite: bool = False,
    fasta_prefix: str = "seq",
) -> str:


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


