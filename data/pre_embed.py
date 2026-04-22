from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_parquet

from data.parse import data_dir
from data.utils import (
    cluster_fasta,
    make_sequence_fasta,
    parse_cd_hit_clstr,
    resolve_data_dir,
)
from plotting import plot_seq_info


class BaseSequenceDS:
    id_columns: tuple[str, ...] = ()
    sequence_columns: tuple[str, ...] = ()
    cluster_columns: tuple[str, ...] = ()

    def __init__(
        self,
        data_name,
        df=None,
        cluster_coef=0.5,
        column_map=None,
        save_dir=data_dir,
        force=False,
    ):
        self.save_dir = resolve_data_dir(save_dir)
        self.base_dir = self.save_dir / data_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.data_name = data_name
        self.cluster_coef = cluster_coef
        self.force = force
        self.column_map = {} if column_map is None else dict(column_map)

        self.data_path = self.base_dir / f"finalized_{int(cluster_coef * 100)}_df.parquet"
        self._fasta_path = self.base_dir / "sequences.fasta"
        self._clstr_path = self.base_dir / f"clustered_{int(cluster_coef * 100)}_sequences.clstr"

        if df is not None and (force or not self.data_path.exists()):
            self.data = df.rename(columns=self.column_map).copy()
            self._validate_columns()
            self.data = self.data.reset_index(drop=True)
            self.ensure_preprocessed(force=force)
            self._save_dataframe()
        elif self.data_path.exists():
            self.data = read_parquet(self.data_path, engine="pyarrow")
            self._validate_columns()
            if force or any(col not in self.data.columns for col in self.cluster_columns):
                self.ensure_preprocessed(force=force)
                self._save_dataframe()
        else:
            raise FileNotFoundError(
                f"File not found: {self.data_path}. Provide `df` to build the dataset first."
            )

        self.data = self.data.reset_index(drop=True)
        self.unique_sequences = self._get_unique_sequences()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def _save_dataframe(self):
        self.data.to_parquet(self.data_path, engine="pyarrow")

    def _validate_columns(self):
        required = {"Y", *self.id_columns, *self.sequence_columns}
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(
                f"Missing required columns for {self.__class__.__name__}: {sorted(missing)}"
            )

    def ensure_fasta(self, force=False) -> Path:
        self.unique_sequences = self._get_unique_sequences()
        self._fasta_path = make_sequence_fasta(
            self.unique_sequences,
            save_dir=self.base_dir,
            force=force,
        )
        return self._fasta_path

    def ensure_clusters(self, force=False) -> Path:
        if force or not self._fasta_path.exists():
            self.ensure_fasta(force=force)
        self._clstr_path = cluster_fasta(
            self._fasta_path,
            force=force,
            cluster_coef=self.cluster_coef,
        )
        return self._clstr_path

    def ensure_preprocessed(self, force=False):
        clstr_path = self.ensure_clusters(force=force)
        if force or any(col not in self.data.columns for col in self.cluster_columns):
            self._add_clusters_to_df(clstr_path)

    def get_lengths(self):
        raise NotImplementedError

    def _add_clusters_to_df(self, clstr_path: Path):
        raise NotImplementedError

    def _get_unique_sequences(self) -> dict[str, str]:
        raise NotImplementedError

    def plot_seq_info(self):
        raise NotImplementedError

    def get_data_groups(self):
        if len(self.cluster_columns) == 1:
            return self.data[self.cluster_columns[0]]

        if len(self.cluster_columns) == 2:
            c1 = self.data[self.cluster_columns[0]].astype(str).to_numpy()
            c2 = self.data[self.cluster_columns[1]].astype(str).to_numpy()
            low = np.minimum(c1, c2)
            high = np.maximum(c1, c2)
            return pd.Series(low + "|" + high, index=self.data.index, name="cluster_pair")

        return self.data[list(self.cluster_columns)].astype(str).agg("|".join, axis=1)

    def get_cluster_pairs(self):
        if len(self.cluster_columns) == 2:
            return self.data[list(self.cluster_columns)]
        return None

    @property
    def fasta_path(self):
        return self._fasta_path if self._fasta_path.exists() else self.ensure_fasta(force=self.force)

    @property
    def clstr_path(self):
        return self._clstr_path if self._clstr_path.exists() else self.ensure_clusters(force=self.force)


class SingleSequenceDS(BaseSequenceDS):
    id_columns = ("ID",)
    sequence_columns = ("Sequence",)
    cluster_columns = ("cluster",)

    def get_lengths(self):
        return self.data["Sequence"].str.len().to_numpy()

    def _add_clusters_to_df(self, clstr_path: Path):
        ids = self.data["ID"].astype(str)
        cluster_map = parse_cd_hit_clstr(clstr_path, ids.tolist())
        self.data["cluster"] = ids.map(cluster_map).astype(np.int64)

    def _get_unique_sequences(self) -> dict[str, str]:
        unique_df = self.data.drop_duplicates(subset=["ID"]).reset_index(drop=True)
        ids = unique_df["ID"].astype(str).tolist()
        seqs = unique_df["Sequence"].astype(str).tolist()
        return dict(zip(ids, seqs))

    def plot_seq_info(self):
        plot_seq_info(self.data["Sequence"], self.data["Y"])


class MultiSequenceDS(BaseSequenceDS):
    id_columns = ("ID1", "ID2")
    sequence_columns = ("Sequence1", "Sequence2")
    cluster_columns = ("cluster1", "cluster2")

    def _get_unique_sequences(self) -> dict[str, str]:
        unique_proteins = pd.concat(
            [
                self.data[["ID1", "Sequence1"]].rename(
                    columns={"ID1": "ID", "Sequence1": "Sequence"}
                ),
                self.data[["ID2", "Sequence2"]].rename(
                    columns={"ID2": "ID", "Sequence2": "Sequence"}
                ),
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["ID"])
        ids = unique_proteins["ID"].astype(str).tolist()
        seqs = unique_proteins["Sequence"].astype(str).tolist()
        return dict(zip(ids, seqs))

    def _add_clusters_to_df(self, clstr_path: Path):
        cluster_map = parse_cd_hit_clstr(clstr_path, self.unique_sequences.keys())
        self.data["cluster1"] = self.data["ID1"].astype(str).map(cluster_map).astype(np.int64)
        self.data["cluster2"] = self.data["ID2"].astype(str).map(cluster_map).astype(np.int64)

    def get_lengths(self):
        len1 = self.data["Sequence1"].str.len().to_numpy()
        len2 = self.data["Sequence2"].str.len().to_numpy()
        return len1 + len2

    def plot_seq_info(self):
        lens = [len(seq) for seq in self.unique_sequences.values()]
        plt.hist(np.array(lens), bins=100)
