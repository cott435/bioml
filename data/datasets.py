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
from esm.sdk.api import ESMProtein
from .embed import compute_sasa_from_pdb

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

    def _setup_esm3_store(
        self,
        save_dir: Path,
        data_name: str,
        esm3_model_name: str | None,
        include_structure: bool,
        storage: str,
        allow_storage_fallback: bool,
    ):
        """Open a secondary store for ESM3 structure/sasa outputs if requested.
        Sets self.esm3_store, self.include_structure, self.esm3_extra_dim.
        Returns the set of seq_ids that have ALL requested tracks available, or None if no ESM3.
        """
        self.include_structure = bool(include_structure)
        self.esm3_store = None
        self.esm3_extra_dim = 0
        if not self.include_structure:
            return None
        if esm3_model_name is None:
            raise ValueError(
                "esm3_model_name must be provided when include_structure is True."
            )
        esm3_dir = save_dir / data_name / esm3_model_name
        esm3_path, esm3_storage = self._resolve_storage_path(
            model_name=esm3_model_name,
            model_dir=esm3_dir,
            storage=storage,
            allow_storage_fallback=allow_storage_fallback,
        )
        self.esm3_store = _load_store(esm3_path, esm3_storage, readonly=True)
        required_tracks = ['structure' ,'sasa_total'] if self.include_structure else []

        ready: set[str] = set()
        for seq_id in self.esm3_store.list_ids():
            try:
                record = self.esm3_store.read(seq_id)
            except Exception:
                continue
            if all(
                t in record and np.asarray(record[t]).size > 0 for t in required_tracks
            ):
                ready.add(seq_id)
        return ready

    def _esm3_feature_vector(self, seq_id: str, seq_len: int) -> np.ndarray:
        """Return concatenated structural features shape (seq_len, extra_dim).

        Features per residue:
          - plddt (1)
          - sasa_relativeSideChain (1)
          - sasa_relativePolar (1)
          - sasa_relativeApolar (1)
          - mean_pae_to_3d_neighbors (1)
        Total: 5 features per residue.
        """
        extra_dim = 5
        try:
            record = self.esm3_store.read(seq_id)
        except (KeyError, FileNotFoundError, AttributeError):
            return np.zeros((seq_len, extra_dim), dtype=np.float32)

        protein = ESMProtein(sequence=self.data.set_index('ID').loc[seq_id, 'Sequence'], coordinates=torch.tensor(record['structure']))
        pdb_string = protein.to_pdb_string()
        sasa_info = compute_sasa_from_pdb(pdb_string)


        # --- pLDDT (L,) ---
        plddt = np.asarray(record["plddt"], dtype=np.float32)
        # Normalize to [0, 1] if it's on the 0-100 scale
        if plddt.max() > 1.5:
            plddt = plddt / 100.0

        # TODO: plot this np.stack([record[r] for r in ['sasa_relativeTotal', 'sasa_relativePolar', 'sasa_relativeApolar']])

        # --- Relative SASA features (L,) each ---
        rel_sc = np.asarray(record["sasa_relativeSideChain"], dtype=np.float32)
        rel_polar = np.asarray(record["sasa_relativePolar"], dtype=np.float32)
        rel_apolar = np.asarray(record["sasa_relativeApolar"], dtype=np.float32)

        # FreeSASA emits NaN for residues where relative areas couldn't be computed
        # (nonstandard residues, missing atoms). Mask those to 0 and let the model
        # learn from the surrounding context.
        has_rel = np.asarray(record["sasa_hasRelativeAreas"], dtype=bool)
        rel_sc = np.where(has_rel, np.nan_to_num(rel_sc, nan=0.0), 0.0)
        rel_polar = np.where(has_rel, np.nan_to_num(rel_polar, nan=0.0), 0.0)
        rel_apolar = np.where(has_rel, np.nan_to_num(rel_apolar, nan=0.0), 0.0)

        # --- Mean PAE to 3D neighbors (L,) ---
        pae = np.asarray(record["pae"], dtype=np.float32)
        ca_coords = record["structure"][:, 1, :]
        pae_neighbor = self._mean_pae_to_3d_neighbors(pae, ca_coords, k=10)
        # PAE is in Å (typically 0-30+). Normalize to roughly [0, 1] for stability.
        pae_neighbor = np.clip(pae_neighbor / 30.0, 0.0, 1.0)

        # --- Stack and align to seq_len ---
        feats = np.stack([plddt, rel_sc, rel_polar, rel_apolar, pae_neighbor], axis=1)
        feats = feats.astype(np.float32)

        # Length safety: pad or truncate to seq_len in case of off-by-one mismatches
        L = feats.shape[0]
        if L == seq_len:
            return feats
        if L > seq_len:
            return feats[:seq_len]
        padded = np.zeros((seq_len, extra_dim), dtype=np.float32)
        padded[:L] = feats
        return padded

    @staticmethod
    def _mean_pae_to_3d_neighbors(
            pae: np.ndarray, ca_coords: np.ndarray, k: int = 10
    ) -> np.ndarray:
        """Mean PAE from residue i to its k nearest 3D neighbors. Returns (L,)."""
        L = ca_coords.shape[0]
        if L == 0:
            return np.zeros(0, dtype=np.float32)
        if L == 1:
            return np.zeros(1, dtype=np.float32)

        diffs = ca_coords[:, None, :] - ca_coords[None, :, :]
        dist = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dist, np.inf)  # exclude self

        k_eff = min(k, L - 1)
        neighbor_idx = np.argpartition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]
        rows = np.arange(L)[:, None]
        pae_to_neighbors = pae[rows, neighbor_idx]
        return pae_to_neighbors.mean(axis=1).astype(np.float32)

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
        include_structure: bool = False,
        esm3_model_name: str | None = None,
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

        esm3_ready_ids = self._setup_esm3_store(
            save_dir=save_dir,
            data_name=data_name,
            esm3_model_name=esm3_model_name,
            include_structure=include_structure,
            storage=storage,
            allow_storage_fallback=allow_storage_fallback,
        )

        def _has_embeddings_and_esm3(seq_id):
            if seq_id not in stored_ids:
                return False
            if esm3_ready_ids is not None and seq_id not in esm3_ready_ids:
                return False
            return True

        filtered_unique = {
            seq_id: seq
            for seq_id, seq in self.unique_sequences.items()
            if len(seq) < max_len and _has_embeddings_and_esm3(seq_id)
        }
        dropped_for_len = len(self.unique_sequences) - len(
            {seq_id: seq for seq_id, seq in self.unique_sequences.items() if len(seq) < max_len}
        )
        if dropped_for_len:
            print(f"Dropped {dropped_for_len} sequences over len {max_len}")

        missing_ids = [
            seq_id
            for seq_id, seq in self.unique_sequences.items()
            if len(seq) < max_len and not _has_embeddings_and_esm3(seq_id)
        ]
        if missing_ids and missing == "raise":
            raise ValueError(
                f"Missing embeddings (or ESM3 tracks) for {len(missing_ids)} IDs."
            )
        if missing_ids and missing == "remove":
            print(f"Removed {len(missing_ids)} rows with missing embeddings/ESM3 tracks")

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
        emb_record = self.store.read(seq_id)
        emb = self._build_representation(emb_record).astype(np.float32, copy=False)
        if self.include_structure:
            structure_record = self._esm3_feature_vector(seq_id, seq_len=emb.shape[0])
            if structure_record.shape[0] != emb.shape[0]:
                """min_len = min(structure_record.shape[0], emb.shape[0])
                emb = emb[:min_len]
                structure_record = structure_record[:min_len]"""
                raise ValueError("ESM3 records have a different length than ESMC embeddings.")
            emb = np.concatenate([emb, structure_record], axis=-1)
        return emb

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
        if getattr(self, "esm3_store", None) is not None:
            self.esm3_store.close()
            self.esm3_store = None

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
        include_structure: bool = False,
        esm3_model_name: str | None = None,
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

        esm3_ready_ids = self._setup_esm3_store(
            save_dir=save_dir,
            data_name=data_name,
            esm3_model_name=esm3_model_name,
            include_structure=include_structure,
            storage=storage,
            allow_storage_fallback=allow_storage_fallback,
        )

        available = stored_ids if esm3_ready_ids is None else (stored_ids & esm3_ready_ids)

        keep_mask = (
            self.data["ID1"].isin(available)
            & self.data["ID2"].isin(available)
            & (self.data["Sequence1"].str.len() < max_len)
            & (self.data["Sequence2"].str.len() < max_len)
        )
        self.data = self.data[keep_mask].reset_index(drop=True)
        if len(self.data) == 0:
            raise ValueError("No paired rows left after filtering by embeddings and max_len.")

        self.ids1 = self.data["ID1"].astype(str).to_numpy()
        self.ids2 = self.data["ID2"].astype(str).to_numpy()
        self.targets = self.data["Y"].to_numpy()
        self.embed_dim = int(self._get_pair_features(self.ids1[0]).shape[-1])

    def _get_pair_features(self, seq_id: str) -> np.ndarray:
        emb = self._build_representation(self.store.read(seq_id)).astype(np.float32, copy=False)
        if self.include_structure or self.include_sasa:
            extras = self._esm3_feature_vector(seq_id, seq_len=emb.shape[0])
            if extras.shape[0] != emb.shape[0]:
                min_len = min(extras.shape[0], emb.shape[0])
                emb = emb[:min_len]
                extras = extras[:min_len]
            emb = np.concatenate([emb, extras], axis=-1)
        return emb

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        emb1 = torch.from_numpy(self._get_pair_features(self.ids1[idx]))
        emb2 = torch.from_numpy(self._get_pair_features(self.ids2[idx]))
        y = torch.tensor(float(self.targets[idx]), dtype=torch.float32)
        return emb1, emb2, y

    def close(self):
        if getattr(self, "store", None) is not None:
            self.store.close()
            self.store = None
        if getattr(self, "esm3_store", None) is not None:
            self.esm3_store.close()
            self.esm3_store = None

    def __del__(self):
        self.close()

