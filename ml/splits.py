from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


@dataclass(slots=True)
class SplitIndices:
    name: str
    train_idx: np.ndarray
    val_idx: np.ndarray


class BaseGroupSplitStrategy(ABC):
    def __init__(self, group_column: str = "cluster", seed: int = 42):
        self.group_column = group_column
        self.seed = seed

    def _groups(self, frame: pd.DataFrame) -> np.ndarray:
        if self.group_column not in frame.columns:
            raise ValueError(f"Group column '{self.group_column}' not found in frame.")
        return frame[self.group_column].to_numpy()

    @abstractmethod
    def iter_splits(self, frame: pd.DataFrame) -> Iterator[SplitIndices]:
        raise NotImplementedError


class SingleGroupSplitStrategy(BaseGroupSplitStrategy):
    def __init__(self, test_size: float = 0.2, group_column: str = "cluster", seed: int = 42):
        super().__init__(group_column=group_column, seed=seed)
        self.test_size = test_size

    def iter_splits(self, frame: pd.DataFrame) -> Iterator[SplitIndices]:
        groups = self._groups(frame)
        splitter = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.seed)
        train_idx, val_idx = next(splitter.split(frame, groups=groups))
        yield SplitIndices(name="split_0", train_idx=train_idx, val_idx=val_idx)


class GroupKFoldSplitStrategy(BaseGroupSplitStrategy):
    def __init__(
        self,
        n_splits: int = 5,
        group_column: str = "cluster",
    ):
        super().__init__(group_column=group_column)
        self.n_splits = n_splits

    def iter_splits(self, frame: pd.DataFrame) -> Iterator[SplitIndices]:
        groups = self._groups(frame)
        splitter = GroupKFold(n_splits=self.n_splits)
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(frame, groups=groups)):
            yield SplitIndices(name=f"fold_{fold_idx}", train_idx=train_idx, val_idx=val_idx)
