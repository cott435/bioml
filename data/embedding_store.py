from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import h5py
import lmdb
import numpy as np


SUPPORTED_STORAGES = {"lmdb", "h5"}
LMDB_META_PREFIX = "__meta__:"
LMDB_KEYS_KEY = f"{LMDB_META_PREFIX}keys"


def normalize_storage(storage: str) -> str:
    normalized = storage.strip().lower()
    if normalized not in SUPPORTED_STORAGES:
        raise ValueError(f"Unsupported storage '{storage}'. Expected one of {sorted(SUPPORTED_STORAGES)}.")
    return normalized


def embedding_file_path(model_dir: Path, model_name: str, storage: str) -> Path:
    storage = normalize_storage(storage)
    return model_dir / f"{model_name}_embeddings.{storage}"


def _pack_npz(record: Dict[str, np.ndarray], compressed: bool = False) -> bytes:
    buffer = io.BytesIO()
    writer = np.savez_compressed if compressed else np.savez
    writer(buffer, **record)
    return buffer.getvalue()


def _unpack_npz(blob: bytes) -> Dict[str, np.ndarray]:
    with np.load(io.BytesIO(blob), allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


class BaseEmbeddingStore(ABC):
    def __init__(self, path: Path):
        self.path = Path(path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @abstractmethod
    def list_ids(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def contains(self, seq_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def read(self, seq_id: str) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def write(self, seq_id: str, record: Dict[str, np.ndarray], overwrite: bool = False):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError


class H5EmbeddingStore(BaseEmbeddingStore):
    def __init__(self, path: Path, mode: str = "r", compression: str = "gzip", compression_opts: int = 4):
        super().__init__(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.compression = compression
        self.compression_opts = compression_opts
        self._h5 = h5py.File(self.path, mode)

    def list_ids(self) -> list[str]:
        return sorted(self._h5.keys())

    def contains(self, seq_id: str) -> bool:
        return seq_id in self._h5

    def read(self, seq_id: str) -> Dict[str, np.ndarray]:
        node = self._h5.get(seq_id)
        if node is None:
            raise KeyError(seq_id)
        if isinstance(node, h5py.Dataset):
            return {"embeddings": node[:]}
        record: Dict[str, np.ndarray] = {}
        for key in node.keys():
            record[key] = node[key][:]
        if "embeddings" not in record:
            raise KeyError(f"Missing 'embeddings' dataset for id '{seq_id}'.")
        return record

    def write(self, seq_id: str, record: Dict[str, np.ndarray], overwrite: bool = False):
        if seq_id in self._h5:
            if not overwrite:
                return
            del self._h5[seq_id]
        group = self._h5.create_group(seq_id)
        for key, array in record.items():
            if key == "hidden_layers":
                group.create_dataset(key, data=array)
            else:
                group.create_dataset(
                    key,
                    data=array,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

    def close(self):
        if getattr(self, "_h5", None) is not None:
            self._h5.flush()
            self._h5.close()
            self._h5 = None


class LMDBEmbeddingStore(BaseEmbeddingStore):
    def __init__(
        self,
        path: Path,
        readonly: bool = False,
        map_size: int = 1 << 40,
        readahead: bool = False,
        lock: bool = True,
        compressed: bool = False,
    ):
        super().__init__(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.readonly = readonly
        self.compressed = compressed
        self._env = lmdb.open(
            str(self.path),
            map_size=map_size,
            subdir=True,
            readonly=readonly,
            readahead=readahead,
            lock=lock,
            create=not readonly,
            max_dbs=1,
        )
        self._cached_keys: set[str] | None = None

    def _load_keys(self) -> set[str]:
        if self._cached_keys is not None:
            return self._cached_keys
        with self._env.begin() as txn:
            blob = txn.get(LMDB_KEYS_KEY.encode("utf-8"))
            if blob:
                keys = {k for k in blob.decode("utf-8").split("\n") if k}
            else:
                keys = set()
                cursor = txn.cursor()
                for key_bytes, _ in cursor:
                    key = key_bytes.decode("utf-8")
                    if not key.startswith(LMDB_META_PREFIX):
                        keys.add(key)
        self._cached_keys = keys
        return keys

    def _persist_keys(self, txn, keys: set[str]):
        if self.readonly:
            return
        joined = "\n".join(sorted(keys)).encode("utf-8")
        txn.put(LMDB_KEYS_KEY.encode("utf-8"), joined, overwrite=True)

    def list_ids(self) -> list[str]:
        return sorted(self._load_keys())

    def contains(self, seq_id: str) -> bool:
        return seq_id in self._load_keys()

    def read(self, seq_id: str) -> Dict[str, np.ndarray]:
        with self._env.begin(write=False) as txn:
            payload = txn.get(seq_id.encode("utf-8"))
        if payload is None:
            raise KeyError(seq_id)

        if payload[:2] == b"PK":
            return _unpack_npz(payload)

        # Backward compatibility for legacy fixed header format:
        # 16 bytes = int64(seq_len, dim) + raw float32 payload.
        if len(payload) < 16:
            raise ValueError(f"Invalid LMDB record for id '{seq_id}'.")
        shape = np.frombuffer(payload[:16], dtype=np.int64, count=2)
        embeddings = np.frombuffer(payload[16:], dtype=np.float32).reshape(tuple(shape)).copy()
        return {"embeddings": embeddings}

    def write(self, seq_id: str, record: Dict[str, np.ndarray], overwrite: bool = False):
        if self.readonly:
            raise RuntimeError("LMDB store opened readonly; cannot write.")
        payload = _pack_npz(record, compressed=self.compressed)
        keys = self._load_keys()
        with self._env.begin(write=True) as txn:
            txn.put(seq_id.encode("utf-8"), payload, overwrite=overwrite)
            if overwrite or seq_id not in keys:
                keys.add(seq_id)
                self._persist_keys(txn, keys)

    def close(self):
        if getattr(self, "_env", None) is not None:
            if not self.readonly:
                self._env.sync()
            self._env.close()
            self._env = None
