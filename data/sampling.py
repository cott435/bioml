import torch
from torch.utils.data import Sampler
from typing import List, Iterable, Optional


class TokenLimitedBatchSampler(Sampler[List[int]]):
    """
    Base class for token-limited batch samplers.
    Handles the greedy batch packing logic once max_tokens is reached.

    Subclasses must implement:
    - get_ordered_indices() → returns Iterable of indices in desired order
    """

    def __init__(
            self,
            data_source,
            lengths: List[int],
            max_tokens: int,
            drop_last: bool = False,
    ):
        if len(lengths) != len(data_source):
            raise ValueError("Lengths must match the dataset size.")

        self.data_source = data_source
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.drop_last = drop_last

    def get_ordered_indices(self) -> Iterable[int]:
        """Subclasses override this to control the order of indices."""
        raise NotImplementedError

    def __iter__(self) -> Iterable[List[int]]:
        indices = self.get_ordered_indices()

        current_batch = []
        current_sum = 0

        for idx in indices:
            length = self.lengths[idx]
            if length > self.max_tokens:
                continue  # Skip overly long sequences

            if current_sum + length > self.max_tokens and current_batch:
                yield current_batch
                current_batch = []
                current_sum = 0

            current_batch.append(idx)
            current_sum += length

        if current_batch and not self.drop_last:
            yield current_batch

    def __len__(self) -> int:
        # Approximate — good enough for most use cases (DataLoader logging, etc.)
        valid_lengths = [min(l, self.max_tokens) for l in self.lengths if l <= self.max_tokens]
        total_tokens = sum(valid_lengths)
        n_batches = total_tokens // self.max_tokens
        remainder = total_tokens % self.max_tokens
        if remainder > 0 and not self.drop_last:
            n_batches += 1
        return n_batches


class RandomTokenBatchSampler(TokenLimitedBatchSampler):
    """
    Training sampler: shuffles indices randomly, then packs batches up to max_tokens.
    """

    def get_ordered_indices(self) -> Iterable[int]:
        return torch.randperm(len(self.data_source)).tolist()


class SortedTokenBatchSampler(TokenLimitedBatchSampler):
    """
    Validation sampler: sorts indices by length, then packs batches up to max_tokens.
    Great for minimizing padding during evaluation.
    """

    def __init__(
            self,
            data_source,
            lengths: List[int],
            max_tokens: int,
            drop_last: bool = False,
            descending: bool = True,  # descending = pack longest-first (common choice)
    ):
        super().__init__(data_source, lengths, max_tokens, drop_last)
        self.descending = descending

    def get_ordered_indices(self) -> Iterable[int]:
        return sorted(
            range(len(self.data_source)),
            key=lambda i: self.lengths[i],
            reverse=self.descending
        )