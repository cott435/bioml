from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .datasets import ESMCPairDS, ESMCSingleDS
from .embed import (
    ESM3ForgeGenerator,
    ESM3LocalGenerator,
    ESMCBatchEmbedder,
    ESMCForgeEmbedder,
    VALID_ESM3_TRACKS,
)
from .parse import data_dir
from .pre_embed import MultiSequenceDS, SingleSequenceDS
from .utils import resolve_data_dir


SequenceKind = Literal["single", "multi"]
EmbedderKind = Literal["local", "forge"]


@dataclass(slots=True)
class PipelineResult:
    sequence_dataset: SingleSequenceDS | MultiSequenceDS
    embedding_file: Path
    storage: str
    written: int
    esm3_file: Path | None = None
    esm3_written: int = 0
    esm3_tracks: tuple[str, ...] = field(default_factory=tuple)


class SequenceProcessingPipeline:
    """
    End-to-end, lazy preprocessing pipeline:
      1) ensure FASTA
      2) ensure clustering
      3) ensure ESMC embeddings (LMDB by default, H5 optional)
      4) optionally ensure ESM3 structure/SASA generation
    """

    def __init__(
        self,
        data_name: str,
        model_name: str = "esmc_300m",
        sequence_kind: SequenceKind = "single",
        save_dir=data_dir,
        cluster_coef: float = 0.5,
    ):
        if sequence_kind not in {"single", "multi"}:
            raise ValueError("sequence_kind must be 'single' or 'multi'.")
        self.data_name = data_name
        self.model_name = model_name
        self.sequence_kind = sequence_kind
        self.cluster_coef = cluster_coef
        self.save_dir = resolve_data_dir(save_dir)

    def _build_sequence_dataset(self, df=None, column_map=None, force=False):
        dataset_cls = SingleSequenceDS if self.sequence_kind == "single" else MultiSequenceDS
        return dataset_cls(
            self.data_name,
            df=df,
            cluster_coef=self.cluster_coef,
            column_map=column_map,
            save_dir=self.save_dir,
            force=force,
        )

    def run(
        self,
        df=None,
        column_map=None,
        embedder_kind: EmbedderKind = "local",
        storage: str = "lmdb",
        include_hidden_states: bool = False,
        hidden_layers: int | list[int] | tuple[int, ...] | None = None,
        dtype: str = "float16",
        device: str = "cpu",
        max_tok_per_batch: int = 50000,
        force_preprocess: bool = False,
        force_embed: bool = False,
        token: str | None = None,
        esm3_tracks: list[str] | tuple[str, ...] | str | None = None,
        esm3_model_name: str = "esm3_sm_open_v1",
        esm3_max_seq_len: int = 2048,
        esm3_num_steps: int | None = None,
        esm3_dtype: str | None = None,
        force_esm3: bool = False,
        hf_token: str | None = None,
    ) -> PipelineResult:
        ds = self._build_sequence_dataset(df=df, column_map=column_map, force=force_preprocess)
        ds.ensure_preprocessed(force=force_preprocess)

        if embedder_kind == "local":
            embedder = ESMCBatchEmbedder(
                model_name=self.model_name,
                save_dir=self.save_dir / self.data_name,
                storage=storage,
                include_hidden_states=include_hidden_states,
                hidden_layers=hidden_layers,
                dtype=dtype,
                device=device,
            )
            written = embedder.batch_save(ds.unique_sequences, max_tok_per_batch=max_tok_per_batch, force=force_embed)
        elif embedder_kind == "forge":
            embedder = ESMCForgeEmbedder(
                model_name=self.model_name,
                token=token,
                save_dir=self.save_dir / self.data_name,
                storage=storage,
                include_hidden_states=include_hidden_states,
                hidden_layers=hidden_layers,
                dtype=dtype,
            )
            written = embedder.batch_save(ds.unique_sequences, force=force_embed)
        else:
            raise ValueError("embedder_kind must be 'local' or 'forge'.")

        esm3_file: Path | None = None
        esm3_written = 0
        esm3_tracks_used: tuple[str, ...] = ()

        if esm3_tracks:
            tracks_tuple = (esm3_tracks,) if isinstance(esm3_tracks, str) else tuple(esm3_tracks)
            invalid = [t for t in tracks_tuple if t not in VALID_ESM3_TRACKS]
            if invalid:
                raise ValueError(
                    f"Invalid esm3_tracks {invalid}. Expected subset of {VALID_ESM3_TRACKS}."
                )

            esm3_kwargs = dict(
                model_name=esm3_model_name,
                save_dir=self.save_dir / self.data_name,
                storage=storage,
                dtype=esm3_dtype or dtype,
                max_seq_len=esm3_max_seq_len,
                tracks=tracks_tuple,
                num_steps=esm3_num_steps,
            )

            if embedder_kind == "local":
                generator = ESM3LocalGenerator(
                    hf_token=hf_token,
                    device=device,
                    **esm3_kwargs,
                )
            else:
                generator = ESM3ForgeGenerator(
                    forge_token=token,
                    **esm3_kwargs,
                )

            esm3_written = generator.batch_save(ds.unique_sequences, force=force_esm3)
            esm3_file = generator.file_path
            esm3_tracks_used = tracks_tuple

        return PipelineResult(
            sequence_dataset=ds,
            embedding_file=embedder.file_path,
            storage=storage,
            written=written,
            esm3_file=esm3_file,
            esm3_written=esm3_written,
            esm3_tracks=esm3_tracks_used,
        )

    def build_training_dataset(
        self,
        storage: str = "lmdb",
        representation: str = "embeddings",
        hidden_layers: int | list[int] | tuple[int, ...] | None = None,
        max_len: int = 5000,
        include_structure: bool = True,
        include_sasa: bool = True,
        esm3_model_name: str = "esm3_sm_open_v1",
    ):
        esm3_name = esm3_model_name if (include_structure or include_sasa) else None
        if self.sequence_kind == "single":
            return ESMCSingleDS(
                self.data_name,
                self.model_name,
                save_dir=self.save_dir,
                storage=storage,
                representation=representation,
                hidden_layers=hidden_layers,
                max_len=max_len,
                include_structure=include_structure,
                include_sasa=include_sasa,
                esm3_model_name=esm3_name,
            )
        return ESMCPairDS(
            self.data_name,
            self.model_name,
            save_dir=self.save_dir,
            storage=storage,
            representation=representation,
            hidden_layers=hidden_layers,
            max_len=max_len,
            include_structure=include_structure,
            include_sasa=include_sasa,
            esm3_model_name=esm3_name,
        )
