from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from esm.models.esmc import ESMC
from esm.sdk import client
from esm.sdk.api import ESMProtein, ESMProteinTensor, GenerationConfig, LogitsConfig, LogitsOutput
from esm.utils.misc import stack_variable_length_tensors
from esm.utils.sampling import _BatchedESMProteinTensor
from tqdm.auto import tqdm
from os import unlink

from .embedding_store import (
    H5EmbeddingStore,
    LMDBEmbeddingStore,
    embedding_file_path,
    normalize_storage,
)
from .utils import resolve_data_dir

"""
Note:
    Local model hidden states outputs L layers worth of hidden_repr.
    Forge model hidden states outputs L+1 layers worth of hidden_repr
    (forge prepends embeddings before transformer blocks).
"""

def get_token(token=None):
    if token is None:
        from dotenv import load_dotenv
        from os import getenv

        load_dotenv()
        return getenv("FORGE_TOKEN")
    return token


def get_hf_token(token=None):
    if token is None:
        from dotenv import load_dotenv
        from os import getenv

        load_dotenv()
        return getenv("HUGGINGFACE_TOKEN")
    return token


def _to_numpy_dtype(dtype) -> np.dtype:
    if isinstance(dtype, np.dtype):
        return dtype
    if isinstance(dtype, str):
        norm = dtype.strip().lower()
        if norm in {"float16", "fp16"}:
            return np.float16
        if norm in {"float32", "fp32"}:
            return np.float32
    if dtype in {np.float16, np.float32}:
        return np.dtype(dtype)
    raise ValueError(f"Unsupported dtype '{dtype}'. Use float16 or float32.")


def compute_sasa_from_pdb(pdb_string: str) -> dict:
    import freesasa
    """Per-residue SASA (Å²) via FreeSASA. Deterministic, geometry-based."""
    # FreeSASA needs a file or stream; use a temp file for simplicity.
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
        f.write(pdb_string)
        path = f.name
    try:
        structure = freesasa.Structure(path)
        result = freesasa.calc(structure)
        residue_areas = result.residueAreas()
        per_res = defaultdict(list)
        chain = next(iter(residue_areas))
        for resnum in sorted(residue_areas[chain], key=lambda x: int(x)):
            for k, v in residue_areas[chain][resnum].__dict__.items():
                per_res[f'sasa_{k}'].append(v)
        data = {k: np.array(v) for k, v in per_res.items()}
        data['sasa_residueNumber'] = data['sasa_residueNumber'].astype(np.int16)
        return data
    finally:
        unlink(path)


class ESMCEmbedder:
    """
    Embeds protein sequences with ESMC.
    Long sequences are split with overlap and merged by averaging.
    """

    def __init__(
        self,
        model_name="esmc_300m",
        save_dir=None,
        storage: str = "lmdb",
        include_hidden_states: bool = False,
        hidden_layers: int | list[int] | tuple[int, ...] | None = None,
        dtype: str | np.dtype = "float16",
        max_seq_len=2000,
        seq_overlap=250,
        device: torch.device | str = "cpu",
    ):
        self.storage = normalize_storage(storage)
        self.include_hidden_states = bool(include_hidden_states)
        self.hidden_layers = self._normalize_hidden_layers(hidden_layers)
        self.numpy_dtype = _to_numpy_dtype(dtype)
        self.max_seq_len = int(max_seq_len)
        self.seq_overlap = int(seq_overlap)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.hidden_layer_offset = 0

        self.model_name = model_name
        self.output_dir = None
        self.file_path = None
        if save_dir is not None:
            resolved = resolve_data_dir(save_dir)
            self.output_dir = resolved / model_name
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.file_path = embedding_file_path(self.output_dir, model_name, self.storage)

        self.emb_config = LogitsConfig(
            sequence=True,
            return_embeddings=True,
            return_hidden_states=self.include_hidden_states,
        )
        self._set_model(model_name)

    @staticmethod
    def _normalize_hidden_layers(hidden_layers):
        if hidden_layers is None:
            return None
        if isinstance(hidden_layers, int):
            return [hidden_layers]
        return [int(layer) for layer in hidden_layers]

    def _set_model(self, model_name: str):
        self.model = ESMC.from_pretrained(model_name, device=self.device)

    def _open_store(self, readonly=False):
        if self.file_path is None:
            raise ValueError("save_dir was not provided; cannot open embedding store.")
        if self.storage == "lmdb":
            return LMDBEmbeddingStore(
                self.file_path,
                readonly=readonly,
                lock=not readonly,
                readahead=False,
            )
        return H5EmbeddingStore(self.file_path, mode="r" if readonly else "a")

    def _check_current_ids(self) -> set[str]:
        if self.file_path is None or not self.file_path.exists():
            return set()
        with self._open_store(readonly=True) as store:
            return set(store.list_ids())

    def _get_new_ids(self, sequences: dict, force=False) -> dict:
        if force:
            return sequences
        existing_ids = self._check_current_ids()
        return {seq_id: seq for seq_id, seq in sequences.items() if seq_id not in existing_ids}

    def _tensorize(self, sequence: str) -> List[ESMProteinTensor]:
        protein = ESMProtein(sequence=sequence)
        protein_tensor = self.model.encode(protein)
        return self._split_tensor_sequences(protein_tensor)

    def _split_tensor_sequences(self, protein_tensor: ESMProteinTensor, min_size=40) -> List[ESMProteinTensor]:
        if len(protein_tensor.sequence) < (self.max_seq_len + min_size):
            return [protein_tensor]
        tokens = protein_tensor.sequence
        chunks = []
        start, end = 0, 0
        while start < len(tokens) and end < len(tokens):
            end = min(len(tokens), start + self.max_seq_len)
            chunk = tokens[start:end]
            if len(chunk) < (min_size + self.seq_overlap):
                chunks[-1] = torch.concat([chunks[-1], chunk[self.seq_overlap :]])
                break
            chunks.append(chunk)
            start += self.max_seq_len - self.seq_overlap
        return [ESMProteinTensor(sequence=seq.to(self.device)) for seq in chunks]

    def _merge_split_embeddings(self, outputs: List[LogitsOutput]) -> LogitsOutput:
        if len(outputs) == 1:
            return outputs[0]

        attrs = ["embeddings"]
        if self.include_hidden_states:
            attrs.append("hidden_states")

        unique_lengths = [outputs[0].embeddings.shape[1]]
        unique_lengths.extend([out.embeddings.shape[1] - self.seq_overlap for out in outputs[1:]])
        indexes = np.array([0] + unique_lengths).cumsum()

        merged_output = {}
        for attr in attrs:
            tensors = [getattr(out, attr) for out in outputs]
            shape = list(tensors[0].shape)
            shape[-2] = int(indexes[-1])
            merged = torch.zeros(*shape, dtype=tensors[0].dtype, device=self.device)
            counts = torch.zeros_like(merged)
            for i in range(len(unique_lengths)):
                start = max(0, int(indexes[i]) - self.seq_overlap)
                end = int(indexes[i + 1])
                merged[..., start:end, :] += tensors[i]
                counts[..., start:end, :] += 1
            merged_output[attr] = merged / counts.clamp(min=1)
        return LogitsOutput(**merged_output)

    def embed_sequence(self, sequence: str) -> LogitsOutput:
        protein_tensors = self._tensorize(sequence)
        outputs = [self.model.logits(protein_tensor, self.emb_config) for protein_tensor in protein_tensors]
        return self._merge_split_embeddings(outputs)

    def _select_hidden_tensor(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
        if self.hidden_layer_offset:
            hidden_states = hidden_states[self.hidden_layer_offset :]
        n_layers = hidden_states.shape[0]
        if self.hidden_layers is None:
            layer_ids = np.arange(n_layers, dtype=np.int16)
            return hidden_states, layer_ids

        selected_idx = []
        selected_layer_ids = []
        for layer in self.hidden_layers:
            normalized = layer if layer >= 0 else n_layers + layer
            if normalized < 0 or normalized >= n_layers:
                raise IndexError(f"Hidden layer index {layer} out of range for {n_layers} layers.")
            selected_idx.append(normalized)
            selected_layer_ids.append(normalized)
        selected = hidden_states[selected_idx]
        return selected, np.asarray(selected_layer_ids, dtype=np.int16)

    def _build_record_from_output(self, output: LogitsOutput, batch_idx: int, seq_len: int):
        seq_slice = slice(1, seq_len + 1)
        emb = output.embeddings[batch_idx, seq_slice].to(torch.float32).cpu().numpy().astype(
            self.numpy_dtype, copy=False
        )
        record = {"embeddings": emb}
        if self.include_hidden_states:
            hidden_states = output.hidden_states[:, batch_idx, seq_slice, :]
            hidden_states, layer_ids = self._select_hidden_tensor(hidden_states)
            record["hidden_states"] = hidden_states.to(torch.float32).cpu().numpy().astype(
                self.numpy_dtype, copy=False
            )
            record["hidden_layers"] = layer_ids
        return record


class ESMCBatchEmbedder(ESMCEmbedder):
    """
    Embeds many sequences in sorted batches.
    Lazy by default: existing IDs in storage are skipped unless force=True.
    """

    def _batch_tensorize(
        self, sequences: dict, max_tok_per_batch: int
    ) -> List[Tuple[List[str], List[_BatchedESMProteinTensor]]]:
        tensor_sequences = {seq_id: self._tensorize(seq) for seq_id, seq in sequences.items()}

        batches, current_keys, current_values = [], [], []
        current_length = None

        for key, tensor_list in tensor_sequences.items():
            length = len(tensor_list)
            current_length = length if current_length is None else current_length
            tok_in_batch = (length - 1) * self.max_seq_len * (len(current_keys) + 1) + (
                (len(current_keys) + 1) * len(tensor_list[-1])
            )
            if (length != current_length or tok_in_batch > max_tok_per_batch) and current_keys:
                stacked = [
                    _BatchedESMProteinTensor(
                        sequence=stack_variable_length_tensors(
                            ts, constant_value=self.model.tokenizer.pad_token_id
                        )
                    )
                    for ts in zip(*current_values)
                ]
                batches.append((current_keys, stacked))
                current_keys, current_values = [], []
                current_length = length

            current_keys.append(key)
            current_values.append([tensor.sequence for tensor in tensor_list])

        if current_keys:
            stacked = [
                _BatchedESMProteinTensor(
                    sequence=stack_variable_length_tensors(ts, constant_value=self.model.tokenizer.pad_token_id)
                )
                for ts in zip(*current_values)
            ]
            batches.append((current_keys, stacked))

        return batches

    def batch_save(self, sequences: dict, max_tok_per_batch=5000, force=False):
        if self.file_path is None:
            raise ValueError("save_dir was not provided; cannot persist embeddings.")

        sequences = self._get_new_ids(sequences, force=force)
        if not sequences:
            return 0

        sorted_sequences = sorted(sequences.items(), key=lambda item: len(item[1]))
        batches = self._batch_tensorize(OrderedDict(sorted_sequences), max_tok_per_batch)

        n_written = 0
        with self._open_store(readonly=False) as store:
            for ids, batched_tensors in tqdm(batches, desc="Embedding batches"):
                split_outputs = [self.model.logits(protein_tensor, self.emb_config) for protein_tensor in batched_tensors]
                merged_output = self._merge_split_embeddings(split_outputs)
                for i, seq_id in enumerate(ids):
                    record = self._build_record_from_output(merged_output, batch_idx=i, seq_len=len(sequences[seq_id]))
                    store.write(seq_id, record, overwrite=force)
                    n_written += 1
        return n_written


class ESMCForgeEmbedder(ESMCEmbedder):
    def __init__(
        self,
        model_name="esmc_300m",
        token=None,
        save_dir=None,
        storage: str = "lmdb",
        include_hidden_states: bool = False,
        hidden_layers: int | list[int] | tuple[int, ...] | None = None,
        dtype: str | np.dtype = "float16",
        max_seq_len=2000,
        seq_overlap=250,
    ):
        self.token = get_token(token)
        super().__init__(
            model_name=model_name,
            save_dir=save_dir,
            storage=storage,
            include_hidden_states=include_hidden_states,
            hidden_layers=hidden_layers,
            dtype=dtype,
            max_seq_len=max_seq_len,
            seq_overlap=seq_overlap,
            device="cpu",
        )
        self.hidden_layer_offset = 1

    def _set_model(self, model_name):
        api_name = model_name + "-2024-12" if "-2024-12" not in model_name else model_name
        api_name = api_name.replace("_", "-")
        self.model = client(api_name, url="https://forge.evolutionaryscale.ai", token=self.token)

    def batch_save(self, sequences: dict, force=False):
        if self.file_path is None:
            raise ValueError("save_dir was not provided; cannot persist embeddings.")

        sequences = self._get_new_ids(sequences, force=force)
        if not sequences:
            return 0

        n_written = 0
        with self._open_store(readonly=False) as store:
            for seq_id, sequence in tqdm(sequences.items(), desc="Embedding (forge)"):
                output = self.embed_sequence(sequence)
                record = self._build_record_from_output(output, batch_idx=0, seq_len=len(sequence))
                store.write(seq_id, record, overwrite=force)
                n_written += 1
        return n_written


ESM3_TRACKS = ("structure", "sasa_esm", "sasa_total")

ESM3_FORGE_NAME_MAP = {
    "esm3_sm_open_v1": "esm3-open-small-2024-03",
    "esm3_medium_2024_03": "esm3-medium-2024-03",
    "esm3_large_2024_03": "esm3-large-2024-03",
}


class ESM3Generator:
    """
    Generates ESM3 structure and/or SASA tracks per sequence.

    Outputs per record (written into the same store keyed by seq_id):
      - 'structure': float array, typically shape (L, 37, 3) of all-atom coordinates
      - 'sasa':      float array, shape (L,) of per-residue SASA (NaN where missing)

    Lazy by default: if a record for seq_id already contains the track, it is skipped.
    Missing tracks for an existing record are generated and merged in.
    """

    def __init__(
        self,
        model_name: str = "esm3_sm_open_v1",
        save_dir=None,
        storage: str = "lmdb",
        dtype: str | np.dtype = "float16",
        max_seq_len: int = 2048,
        device: torch.device | str = "cpu",
        num_steps: int | None = None,
        temperature=1.0
    ):
        self.storage = normalize_storage(storage)
        self.numpy_dtype = _to_numpy_dtype(dtype)
        self.max_seq_len = int(max_seq_len)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.num_steps = num_steps
        self.model_name = model_name
        self.temperature = temperature

        self.output_dir = None
        self.file_path = None
        if save_dir is not None:
            resolved = resolve_data_dir(save_dir)
            self.output_dir = resolved / model_name
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.file_path = embedding_file_path(self.output_dir, model_name, self.storage)

        self._set_model(model_name)

    def _set_model(self, model_name: str):
        raise NotImplementedError

    def _open_store(self, readonly=False):
        if self.file_path is None:
            raise ValueError("save_dir was not provided; cannot open ESM3 store.")
        if self.storage == "lmdb":
            return LMDBEmbeddingStore(
                self.file_path,
                readonly=readonly,
                lock=not readonly,
                readahead=False,
            )
        return H5EmbeddingStore(self.file_path, mode="r" if readonly else "a")

    def _load_existing_track_map(self) -> dict[str, set[str]]:
        """For each target track, return the set of seq_ids already stored."""
        per_track: dict[str, set[str]] = {t: set() for t in ESM3_TRACKS}
        if self.file_path is None or not self.file_path.exists():
            return per_track
        with self._open_store(readonly=True) as store:
            for seq_id in store.list_ids():
                try:
                    record = store.read(seq_id)
                except Exception:
                    continue
                for track in ESM3_TRACKS:
                    if track in record and np.asarray(record[track]).size > 0:
                        per_track[track].add(seq_id)
        return per_track

    def _num_steps_for(self, seq_len: int) -> int:
        if self.num_steps is not None:
            return max(1, int(self.num_steps))
        return max(1, seq_len // 8)

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if value is None:
            return np.asarray([])
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (list, tuple)):
            cleaned = [np.nan if v is None else float(v) for v in value]
            return np.asarray(cleaned, dtype=np.float32)
        return np.asarray(value)

    def generate(
        self,
        sequence: str,
        init_structure: np.ndarray | torch.Tensor | None = None,
    ) -> dict:
        protein = ESMProtein(sequence=sequence)
        steps = 1 if self.device.type == 'cpu' else self._num_steps_for(len(sequence))
        outputs: dict = {}
        if init_structure is not None:
            coords_t = init_structure
            if isinstance(coords_t, np.ndarray):
                coords_t = torch.from_numpy(np.ascontiguousarray(coords_t))
            protein.coordinates = coords_t.to(torch.float32)
        else:
            config = GenerationConfig(track="structure", num_steps=steps, temperature=self.temperature)
            protein = self.model.generate(protein, config)
            outputs["structure"] = self._to_numpy(protein.coordinates)
            outputs["plddt"] = self._to_numpy(protein.plddt)
            if protein.ptm is not None:
                outputs["ptm"] = self._to_numpy(protein.ptm)
            outputs["pae"] = self._to_numpy(protein.pae)[0]
        # Generate esm sasa
        config = GenerationConfig(track="sasa", num_steps=steps, temperature=self.temperature)
        protein = self.model.generate(protein, config)
        outputs["sasa_esm"] = self._to_numpy(protein.sasa)
        # generate freesasa
        pdb_string = protein.to_pdb_string()
        sasa = compute_sasa_from_pdb(pdb_string)
        outputs.update(sasa)
        return outputs

    def batch_save(self, sequences: dict, force: bool = False, desc: str = "ESM3 generating", max_block=None):
        if self.file_path is None:
            raise ValueError("save_dir was not provided; cannot persist ESM3 outputs.")

        existing = {t: set() for t in ESM3_TRACKS} if force else self._load_existing_track_map()

        n_written = 0
        n_skipped_len = 0
        with self._open_store(readonly=False) as store:
            for seq_id, sequence in tqdm(sequences.items(), desc=desc):
                if len(sequence) > self.max_seq_len:
                    n_skipped_len += 1
                    continue
                if force:
                    missing = ESM3_TRACKS
                else:
                    missing = tuple(t for t in ESM3_TRACKS if seq_id not in existing[t])
                if not missing:
                    continue

                init_structure = None
                if (
                    "sasa" in missing
                    and "structure" not in missing
                    and store.contains(seq_id)
                ):
                    try:
                        prev = store.read(seq_id)
                        if "structure" in prev and np.asarray(prev["structure"]).size > 0:
                            init_structure = prev["structure"]
                    except Exception:
                        init_structure = None

                new_record = self.generate(
                    sequence, init_structure=init_structure
                )

                if not force and store.contains(seq_id):
                    try:
                        prev = store.read(seq_id)
                        prev.update(new_record)
                        new_record = prev
                    except Exception:
                        pass
                store.write(seq_id, new_record, overwrite=True)
                n_written += 1
                if max_block and n_written >= max_block:
                    print("Max limit for session reached. Ending session.")
                    break

        if n_skipped_len:
            print(f"Skipped {n_skipped_len} sequences exceeding max_seq_len={self.max_seq_len}.")
        return n_written


class ESM3LocalGenerator(ESM3Generator):
    def __init__(self, hf_token: str | None = None, **kwargs):
        self.hf_token = get_hf_token(hf_token)
        super().__init__(**kwargs)

    def _set_model(self, model_name: str):
        if self.hf_token:
            import os

            os.environ.setdefault("HF_TOKEN", self.hf_token)
            try:
                from huggingface_hub import login

                login(token=self.hf_token, add_to_git_credential=False)
            except Exception as exc:
                print(f"huggingface_hub login failed ({exc}); relying on HF_TOKEN env var.")
        from esm.models.esm3 import ESM3

        self.model = ESM3.from_pretrained(model_name).to(self.device)


class ESM3ForgeGenerator(ESM3Generator):
    def __init__(self, forge_token: str | None = None, **kwargs):
        self.forge_token = get_token(forge_token)
        kwargs.setdefault("device", "cpu")
        super().__init__(**kwargs)

    def _resolve_forge_name(self, model_name: str) -> str:
        if model_name in ESM3_FORGE_NAME_MAP:
            return ESM3_FORGE_NAME_MAP[model_name]
        if model_name.startswith("esm3-"):
            return model_name
        return model_name.replace("_", "-")

    def _set_model(self, model_name: str):
        api_name = self._resolve_forge_name(model_name)
        self.model = client(api_name, url="https://forge.evolutionaryscale.ai", token=self.forge_token)
