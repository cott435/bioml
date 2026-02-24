import torch
from torch.utils.data import Sampler
from typing import List, Iterable, Optional
import numpy as np
from sklearn.model_selection import KFold


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


class ClusterPairSplitter:
    """
    Splits paired data based on cluster sets to prevent data leakage.
    Supports both C2 and C3 splitting strategies.

    Parameters:
    -----------
    n_splits : int
        Number of folds for K-Fold.
    split_mode : str
        'C3' (default): Validation pairs must have BOTH clusters unseen.
                        (Hardest - New biology generalization).
                        Discards mixed (Train-Val) pairs.

        'C2':           Validation pairs must have EXACTLY ONE unseen cluster.
                        (Medium - Finding new partners for known proteins).
                        Discards fully unseen (Val-Val) pairs to isolate C2 performance.
    shuffle : bool
        Whether to shuffle clusters before splitting.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(self, n_splits=5, split_mode='C3', shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.split_mode = split_mode.upper()
        self.shuffle = shuffle
        self.random_state = random_state

        if self.split_mode not in ['C2', 'C3']:
            raise ValueError("split_mode must be either 'C2' or 'C3'")

    def split(self, X, y=None, groups=None):
        """
        Yields train_idx and val_idx.

        Args:
            X: Placeholder.
            y: Placeholder.
            groups: Dataframe
        """
        c1, c2 = np.split(groups.values, 2)

        unique_clusters = np.unique(np.concatenate([c1, c2]))
        kfold = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

        for train_clusters_idx, val_clusters_idx in kfold.split(unique_clusters):
            train_cluster_set = set(unique_clusters[train_clusters_idx])
            val_cluster_set = set(unique_clusters[val_clusters_idx])

            c1_in_train = np.isin(c1, list(train_cluster_set))
            c1_in_val = np.isin(c1, list(val_cluster_set))

            c2_in_train = np.isin(c2, list(train_cluster_set))
            c2_in_val = np.isin(c2, list(val_cluster_set))

            train_mask = c1_in_train & c2_in_train

            if self.split_mode == 'C3':  # No group overlap
                val_mask = c1_in_val & c2_in_val

            elif self.split_mode == 'C2':  # One group overlap
                val_mask = (c1_in_train & c2_in_val) | (c1_in_val & c2_in_train)

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]

            yield train_idx, val_idx


