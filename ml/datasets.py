from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class TokenizedMLDataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_column: str = "y"
    group_column: str = "cluster"

    @property
    def num_tokens(self) -> int:
        return len(self.frame)

    @property
    def num_features(self) -> int:
        return len(self.feature_columns)


class BaseTokenFrameBuilder(ABC):
    def __init__(self, dataset, seed: int = 42):
        self.dataset = dataset
        self.seed = seed

    @abstractmethod
    def build(self, max_tokens: int | None = None) -> TokenizedMLDataset:
        raise NotImplementedError


class ESMCTokenFrameBuilder(BaseTokenFrameBuilder):
    """
    Flattens sequence-level embeddings into token-level rows.
    """

    def build(self, max_tokens: int | None = None) -> TokenizedMLDataset:
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be > 0 when provided.")
        if "ID" not in self.dataset.data.columns or "cluster" not in self.dataset.data.columns:
            raise ValueError("dataset.data must include 'ID' and 'cluster' columns.")

        ids = self.dataset.data["ID"].to_numpy()
        clusters = self.dataset.data["cluster"].to_numpy()

        rng = np.random.default_rng(self.seed)
        seq_indices = np.arange(len(self.dataset))
        if max_tokens is not None:
            rng.shuffle(seq_indices)

        feature_chunks: list[np.ndarray] = []
        y_chunks: list[np.ndarray] = []
        id_chunks: list[np.ndarray] = []
        cluster_chunks: list[np.ndarray] = []
        seq_chunks: list[np.ndarray] = []
        token_chunks: list[np.ndarray] = []

        remaining = max_tokens
        for seq_idx in seq_indices:
            if remaining == 0:
                break

            emb, y = self.dataset[seq_idx]
            emb_np = emb.detach().cpu().numpy().astype(np.float32, copy=False)
            y_np = (y.detach().cpu().numpy().astype(np.float32, copy=False) > 0.5).astype(np.int8, copy=False)

            if emb_np.ndim != 2:
                raise ValueError(f"Expected embedding shape (seq_len, dim), got {emb_np.shape}")
            if y_np.ndim != 1 or y_np.shape[0] != emb_np.shape[0]:
                raise ValueError(f"Label shape {y_np.shape} does not match embedding length {emb_np.shape[0]}.")

            token_idx = np.arange(emb_np.shape[0], dtype=np.int32)
            if remaining is not None and emb_np.shape[0] > remaining:
                keep = np.sort(rng.choice(emb_np.shape[0], size=remaining, replace=False))
                emb_np = emb_np[keep]
                y_np = y_np[keep]
                token_idx = token_idx[keep]

            n_tokens = emb_np.shape[0]
            if n_tokens == 0:
                continue

            feature_chunks.append(emb_np)
            y_chunks.append(y_np)
            id_chunks.append(np.repeat(ids[seq_idx], n_tokens))
            cluster_chunks.append(np.repeat(clusters[seq_idx], n_tokens))
            seq_chunks.append(np.repeat(seq_idx, n_tokens))
            token_chunks.append(token_idx)

            if remaining is not None:
                remaining -= n_tokens

        if not feature_chunks:
            raise ValueError("No tokens were collected. Check dataset contents or max_tokens.")

        features = np.concatenate(feature_chunks, axis=0)
        feature_columns = tuple(f"f{i:04d}" for i in range(features.shape[1]))

        frame = pd.DataFrame(
            {
                "sequence_idx": np.concatenate(seq_chunks),
                "token_idx": np.concatenate(token_chunks),
                "ID": np.concatenate(id_chunks),
                "cluster": np.concatenate(cluster_chunks),
                "y": np.concatenate(y_chunks),
            }
        )
        feature_frame = pd.DataFrame(features, columns=feature_columns)
        frame = pd.concat([frame, feature_frame], axis=1)

        return TokenizedMLDataset(frame=frame, feature_columns=feature_columns)
