import torch
from esm import Alphabet
from typing import List

available_tok = {'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M',
                 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-'}

def sanitize_sequence(seq):
    return ''.join(c if c in available_tok else 'X' for c in seq)


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


