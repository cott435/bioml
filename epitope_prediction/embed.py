import os
import numpy as np
import torch
from typing import List, Iterator, Tuple
import random
from esm import Alphabet


class EmbeddingStore:
    def __init__(self, storage_path: str):
        """
        Initialize the EmbeddingStore with a path to store the embeddings on disk.

        :param storage_path: Directory path where embeddings are saved as .npy files.
        """
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

    def save_embeddings(self, embeddings_dict: dict[int, torch.Tensor]):
        """
        Save a dictionary of embeddings to disk as individual .npy files.
        """
        for idx, tensor in embeddings_dict.items():
            if not isinstance(idx, int):
                raise ValueError("Indexes must be integers.")
            np_array = tensor.cpu().numpy()
            file_path = os.path.join(self.storage_path, f"{idx}.npy")
            np.save(file_path, np_array)

    def get_batch(self, indexes: List[int], device: str = 'cpu') -> torch.Tensor:
        """
        Load and stack embeddings for a list of indexes.
        """
        arrays = []
        for idx in indexes:
            file_path = os.path.join(self.storage_path, f"{idx}.npy")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Embedding for index {idx} not found.")
            arrays.append(np.load(file_path))
        stacked = np.stack(arrays)
        return torch.from_numpy(stacked).to(device)

    def get_dataloader(
            self,
            indexes: List[int],
            batch_size: int,
            shuffle: bool = True,
            device: str = 'cpu',
            drop_last: bool = False
    ) -> Iterator[torch.Tensor]:
        """
        Returns an iterator that yields batches of embeddings.

        :param indexes: List of integer indexes to sample from.
        :param batch_size: Size of each batch.
        :param shuffle: Whether to shuffle the indexes at the start of iteration.
        :param device: Device to load tensors onto ('cpu' or 'cuda').
        :param drop_last: If True, drop the last incomplete batch.
        :return: Iterator yielding batched torch.Tensors.
        """
        idx_list = list(indexes)
        if shuffle:
            random.shuffle(idx_list)

        if drop_last:
            total_batches = len(idx_list) // batch_size
            idx_list = idx_list[:total_batches * batch_size]

        for i in range(0, len(idx_list), batch_size):
            batch_indexes = idx_list[i:i + batch_size]
            yield self.get_batch(batch_indexes, device=device)

class ESMBatcher:

    def __init__(self, alphabet: Alphabet):
        self.alphabet = alphabet
        self.batch_converter = self.alphabet.get_batch_converter()
        self.available_tok = {'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X',
             'B', 'U', 'Z', 'O', '.', '-'}

    def sanitize_sequence(self, seq):
        return ''.join(c if c in self.available_tok else 'X' for c in seq)

    def collate_batch(self, batch):
        sequences = [("", self.sanitize_sequence(b)) for b in batch]
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

