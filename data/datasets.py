from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from .embedding_store import (
    H5EmbeddingStore,
    LMDBEmbeddingStore,
    embedding_file_path,
    normalize_storage,
)
from .parse import data_dir
from .pre_embed import MultiSequenceDS, SingleSequenceDS
from .utils import resolve_data_dir


RepresentationMode = Literal["embeddings", "hidden_states", "concat"]


def _load_store(path: Path, storage: str, readonly: bool):
    if storage == "lmdb":
        return LMDBEmbeddingStore(path, readonly=readonly, lock=False, readahead=False)
    if storage == "h5":
        return H5EmbeddingStore(path, mode="r" if readonly else "a")
    raise ValueError(f"Unsupported storage '{storage}'.")


class _EmbeddingReaderMixin:
    def _resolve_storage_path(self, model_name: str, model_dir: Path, storage: str, allow_storage_fallback: bool):
        storage = normalize_storage(storage)
        requested = embedding_file_path(model_dir, model_name, storage)
        if requested.exists():
            return requested, storage

        if allow_storage_fallback:
            fallback_storage = "h5" if storage == "lmdb" else "lmdb"
            fallback = embedding_file_path(model_dir, model_name, fallback_storage)
            if fallback.exists():
                return fallback, fallback_storage
        raise FileNotFoundError(
            f"Embedding file not found. Checked: '{requested}'"
            + (f" and fallback storage for '{storage}'." if allow_storage_fallback else ".")
        )

    @staticmethod
    def _normalize_hidden_layers(hidden_layers):
        if hidden_layers is None:
            return None
        if isinstance(hidden_layers, int):
            return [hidden_layers]
        return [int(layer) for layer in hidden_layers]

    def _select_hidden_layers(self, hidden: np.ndarray, stored_layers: np.ndarray | None):
        selected = self.hidden_layers
        if selected is None:
            return hidden

        if stored_layers is None:
            n_layers = hidden.shape[0]
            idx = []
            for layer in selected:
                normalized = layer if layer >= 0 else n_layers + layer
                if normalized < 0 or normalized >= n_layers:
                    raise IndexError(f"Hidden layer index {layer} out of range for {n_layers} layers.")
                idx.append(normalized)
            return hidden[idx]

        stored_layers = stored_layers.astype(np.int64)
        idx = []
        for layer in selected:
            matches = np.where(stored_layers == layer)[0]
            if matches.size == 0:
                raise KeyError(
                    f"Requested hidden layer {layer} not stored. Available layers: {stored_layers.tolist()}."
                )
            idx.append(int(matches[0]))
        return hidden[idx]

    def _build_representation(self, record: dict[str, np.ndarray]) -> np.ndarray:
        emb = record["embeddings"]
        if self.representation == "embeddings":
            return emb

        hidden = record.get("hidden_states")
        if hidden is None:
            raise KeyError(
                f"Hidden states requested with representation='{self.representation}', "
                "but no hidden_states were found in the embedding store."
            )

        hidden = self._select_hidden_layers(hidden, record.get("hidden_layers"))
        hidden_flat = np.transpose(hidden, (1, 0, 2)).reshape(hidden.shape[1], -1)

        if self.representation == "hidden_states":
            return hidden_flat
        return np.concatenate([emb, hidden_flat], axis=-1)

    @staticmethod
    def _build_token_labels(target, seq_len: int) -> torch.Tensor:
        if isinstance(target, np.ndarray) and target.ndim == 1 and target.shape[0] == seq_len:
            return torch.from_numpy(target.astype(np.float32, copy=False))

        if isinstance(target, (list, tuple, set, np.ndarray)):
            active_sites = np.asarray(list(target), dtype=np.int64)
            y = torch.zeros(seq_len, dtype=torch.float32)
            if active_sites.size > 0:
                valid = active_sites[(active_sites >= 0) & (active_sites < seq_len)]
                if valid.size > 0:
                    y.index_fill_(0, torch.from_numpy(valid), 1.0)
            return y

        # Sequence-level scalar fallback.
        return torch.full((seq_len,), float(target), dtype=torch.float32)


class ESMCSingleDS(SingleSequenceDS, torch.utils.data.Dataset, _EmbeddingReaderMixin):
    def __init__(
        self,
        data_name,
        model_name,
        df=None,
        cluster_coef=0.5,
        column_map=None,
        save_dir=data_dir,
        force=False,
        missing: Literal["raise", "remove"] = "remove",
        max_len=5000,
        storage: str = "lmdb",
        allow_storage_fallback: bool = True,
        representation: RepresentationMode = "embeddings",
        hidden_layers: int | list[int] | tuple[int, ...] | None = None,
    ):
        super().__init__(
            data_name,
            df=df,
            cluster_coef=cluster_coef,
            column_map=column_map,
            save_dir=save_dir,
            force=force,
        )
        if missing not in {"raise", "remove"}:
            raise ValueError("missing must be either 'raise' or 'remove'")
        if representation not in {"embeddings", "hidden_states", "concat"}:
            raise ValueError("representation must be one of: 'embeddings', 'hidden_states', 'concat'")

        save_dir = resolve_data_dir(save_dir)
        self.model_name = model_name
        self.model_dir = save_dir / data_name / model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.representation = representation
        self.hidden_layers = self._normalize_hidden_layers(hidden_layers)

        self.file_path, self.storage = self._resolve_storage_path(
            model_name=model_name,
            model_dir=self.model_dir,
            storage=storage,
            allow_storage_fallback=allow_storage_fallback,
        )
        self.store = _load_store(self.file_path, self.storage, readonly=True)
        stored_ids = set(self.store.list_ids())

        filtered_unique = {
            seq_id: seq
            for seq_id, seq in self.unique_sequences.items()
            if len(seq) < max_len and seq_id in stored_ids
        }
        dropped_for_len = len(self.unique_sequences) - len(
            {seq_id: seq for seq_id, seq in self.unique_sequences.items() if len(seq) < max_len}
        )
        if dropped_for_len:
            print(f"Dropped {dropped_for_len} sequences over len {max_len}")

        missing_ids = [
            seq_id
            for seq_id, seq in self.unique_sequences.items()
            if len(seq) < max_len and seq_id not in stored_ids
        ]
        if missing_ids and missing == "raise":
            raise ValueError(f"Missing embeddings for {len(missing_ids)} IDs in {self.file_path}")
        if missing_ids and missing == "remove":
            print(f"Removed {len(missing_ids)} rows with missing embeddings")

        self.unique_sequences = filtered_unique
        self.data = self.data[self.data["ID"].isin(self.unique_sequences.keys())].reset_index(drop=True)
        self.ids = self.data["ID"].astype(str).to_numpy()
        self.labels = self.data["Y"].to_list()
        if len(self.ids) == 0:
            raise ValueError("No rows left after filtering by embedding availability.")

        self.embed_dim = int(self._get_embedding(self.ids[0]).shape[-1])

    def __len__(self):
        return len(self.data)

    def _get_embedding(self, seq_id: str) -> np.ndarray:
        record = self.store.read(seq_id)
        return self._build_representation(record).astype(np.float32, copy=False)

    def __getitem__(self, idx):
        seq_id = self.ids[idx]
        emb_np = self._get_embedding(seq_id)
        emb = torch.from_numpy(emb_np)
        y = self._build_token_labels(self.labels[idx], emb.shape[0])
        return emb, y

    def close(self):
        if getattr(self, "store", None) is not None:
            self.store.close()
            self.store = None

    def __del__(self):
        self.close()

    def to_lmdb(self, overwrite=False):
        if self.storage == "lmdb" and not overwrite:
            return self.file_path
        target = embedding_file_path(self.model_dir, self.model_name, "lmdb")
        with LMDBEmbeddingStore(target, readonly=False) as writer:
            for seq_id in tqdm(self.store.list_ids(), desc="Converting to LMDB"):
                writer.write(seq_id, self.store.read(seq_id), overwrite=overwrite)
        return target

    def save_full_embedding(self, float16=True):
        emb_list = []
        offsets = []
        lengths = []
        labels = []

        offset = 0
        for i, seq_id in enumerate(tqdm(self.ids, desc="Packing embeddings")):
            emb = self._get_embedding(seq_id)
            y = self._build_token_labels(self.labels[i], emb.shape[0]).numpy().astype(np.float32, copy=False)
            emb_list.append(emb)
            labels.append(y)
            offsets.append(offset)
            lengths.append(len(emb))
            offset += len(emb)

        embeddings = np.concatenate(emb_list, axis=0)
        if float16:
            embeddings = embeddings.astype(np.float16, copy=False)

        base = self.model_dir
        np.save(base / "embeddings.npy", embeddings)
        np.save(base / "offsets.npy", np.asarray(offsets, dtype=np.int64))
        np.save(base / "lengths.npy", np.asarray(lengths, dtype=np.int64))
        np.save(base / "labels.npy", np.concatenate(labels).astype(np.float32, copy=False))

    def heatmap(self, idx: int | list[int]):
        if isinstance(idx, list):
            emb = np.concatenate([self[i][0].numpy() for i in idx], axis=0).T
        else:
            emb = self[idx][0].numpy().T
        plt.figure()
        plt.imshow(emb, aspect="auto", cmap="viridis")
        plt.colorbar(label="Value")
        plt.xlabel("Tokens")
        plt.ylabel("Dim")
        plt.title(f"Protein {idx}")
        plt.show()


class ESMLMDBDataset(ESMCSingleDS):
    def __init__(self, *args, **kwargs):
        kwargs["storage"] = "lmdb"
        super().__init__(*args, **kwargs)


class PackedSequenceDataset(SingleSequenceDS, torch.utils.data.Dataset):
    def __init__(
        self,
        data_name,
        model_name,
        cluster_coef=0.5,
        column_map=None,
        save_dir=data_dir,
        force=False,
        max_len=5000,
    ):
        super().__init__(
            data_name,
            cluster_coef=cluster_coef,
            column_map=column_map,
            save_dir=save_dir,
            force=force,
        )
        files_dir = resolve_data_dir(save_dir) / data_name / model_name
        self.embeddings = np.load(files_dir / "embeddings.npy", mmap_mode="r")
        self.labels = np.load(files_dir / "labels.npy", mmap_mode="r")
        self.offsets = np.load(files_dir / "offsets.npy")
        self.lengths = np.load(files_dir / "lengths.npy")
        unique_sequences = {seq_id: seq for seq_id, seq in self.unique_sequences.items() if len(seq) < max_len}
        self.data = self.data[self.data["ID"].isin(unique_sequences.keys())].reset_index(drop=True)
        self.embed_dim = self.embeddings.shape[1]

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        start = self.offsets[idx]
        end = start + self.lengths[idx]
        emb = torch.from_numpy(self.embeddings[start:end].copy()).to(torch.float32)
        y = torch.from_numpy(self.labels[start:end].copy()).to(torch.float32)
        return emb, y


class ESMCPairDS(MultiSequenceDS, torch.utils.data.Dataset, _EmbeddingReaderMixin):
    """
    Pair dataset for interaction-style tasks where each row has two sequences.
    Returns (emb1, emb2, y_scalar).
    """

    def __init__(
        self,
        data_name,
        model_name,
        df=None,
        cluster_coef=0.5,
        column_map=None,
        save_dir=data_dir,
        force=False,
        max_len=5000,
        storage: str = "lmdb",
        allow_storage_fallback: bool = True,
        representation: RepresentationMode = "embeddings",
        hidden_layers: int | list[int] | tuple[int, ...] | None = None,
    ):
        super().__init__(
            data_name,
            df=df,
            cluster_coef=cluster_coef,
            column_map=column_map,
            save_dir=save_dir,
            force=force,
        )
        if representation not in {"embeddings", "hidden_states", "concat"}:
            raise ValueError("representation must be one of: 'embeddings', 'hidden_states', 'concat'")

        save_dir = resolve_data_dir(save_dir)
        self.model_name = model_name
        self.model_dir = save_dir / data_name / model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.representation = representation
        self.hidden_layers = self._normalize_hidden_layers(hidden_layers)

        self.file_path, self.storage = self._resolve_storage_path(
            model_name=model_name,
            model_dir=self.model_dir,
            storage=storage,
            allow_storage_fallback=allow_storage_fallback,
        )
        self.store = _load_store(self.file_path, self.storage, readonly=True)
        stored_ids = set(self.store.list_ids())

        keep_mask = (
            self.data["ID1"].isin(stored_ids)
            & self.data["ID2"].isin(stored_ids)
            & (self.data["Sequence1"].str.len() < max_len)
            & (self.data["Sequence2"].str.len() < max_len)
        )
        self.data = self.data[keep_mask].reset_index(drop=True)
        if len(self.data) == 0:
            raise ValueError("No paired rows left after filtering by embeddings and max_len.")

        self.ids1 = self.data["ID1"].astype(str).to_numpy()
        self.ids2 = self.data["ID2"].astype(str).to_numpy()
        self.targets = self.data["Y"].to_numpy()
        self.embed_dim = int(self._build_representation(self.store.read(self.ids1[0])).shape[-1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        emb1 = torch.from_numpy(self._build_representation(self.store.read(self.ids1[idx])).astype(np.float32, copy=False))
        emb2 = torch.from_numpy(self._build_representation(self.store.read(self.ids2[idx])).astype(np.float32, copy=False))
        y = torch.tensor(float(self.targets[idx]), dtype=torch.float32)
        return emb1, emb2, y

    def close(self):
        if getattr(self, "store", None) is not None:
            self.store.close()
            self.store = None

    def __del__(self):
        self.close()

