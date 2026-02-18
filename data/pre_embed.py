import numpy as np
import pandas as pd
from pandas import read_parquet
from data.parse import data_dir
from data.utils import parse_cd_hit_clstr, make_sequence_fasta, cluster_fasta
from plotting import plot_seq_info
import matplotlib.pyplot as plt


class SingleSequenceDS:

    def __init__(self, data_name, df=None, cluster_coef=0.5, column_map=None, save_dir=data_dir, force=False):
        self.base_dir = save_dir / data_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        data_path = self.base_dir / f'finalized_{int(cluster_coef*100)}_df.parquet'
        self.data_name = data_name
        self._clstr_path = self.base_dir / f"clustered_{int(cluster_coef*100)}_sequences.clstr"
        self._fasta_path = self.base_dir / "sequences.fasta"
        self._fasta_path = self._fasta_path if self._fasta_path.exists() else None
        self._clstr_path = self._clstr_path if self._clstr_path.exists() else None
        self.force = force
        self.cluster_coef = cluster_coef

        if data_path.exists() or force:
            self.data = read_parquet(data_path, engine="pyarrow")
            self.unique_sequences = self._get_unique_sequences()
        elif df is None:
            raise FileNotFoundError(f"File not found: {data_path} and df=None, please provide data")
        else:
            column_map={} if column_map is None else column_map
            self.data = df.rename(columns=column_map)
            self.unique_sequences = self._get_unique_sequences()
            self._add_clusters_to_df()
            self.data.to_parquet(data_path, engine="pyarrow")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def get_lengths(self):
        return self.data['Sequence'].apply(len).values

    def _add_clusters_to_df(self):
        ids = self.data['ID']
        cluster_map = parse_cd_hit_clstr(self.clstr_path, set(ids))
        self.data['cluster'] = ids.map(cluster_map)

    def _get_unique_sequences(self)-> dict:
        unique_df = self.data.drop_duplicates(subset=["ID"]).reset_index(drop=True)
        return dict(zip(unique_df['ID'], unique_df['Sequence']))

    @property
    def fasta_path(self):
        return self._fasta_path if self._fasta_path is not None \
            else make_sequence_fasta(self.unique_sequences, save_dir=self.base_dir, force=self.force)

    @property
    def clstr_path(self):
        return self._clstr_path if self._clstr_path else cluster_fasta(self.fasta_path, force=self.force,
                                                                       cluster_coef=self.cluster_coef)

    def plot_seq_info(self):
        plot_seq_info(self.data['Sequence'], self.data['Y'])

    def get_data_groups(self):
        return self.data['cluster']


class MultiSequenceDS(SingleSequenceDS):

    def _get_unique_sequences(self)-> dict:
        unique_proteins = pd.concat(
            [
                self.data[["ID1", "Sequence1"]]
                .rename(columns={"ID1": "ID", "Sequence1": "Sequence"}),
                self.data[["ID2", "Sequence2"]]
                .rename(columns={"ID2": "ID", "Sequence2": "Sequence"}),
            ],
            ignore_index=True
        ).drop_duplicates(subset=["ID"]).reset_index(drop=True)
        return dict(zip(unique_proteins['ID'], unique_proteins['Sequence']))

    def get_data_groups(self):
        return self.data[['cluster1', 'cluster2']]

    def _add_clusters_to_df(self):
        cluster_map = parse_cd_hit_clstr(self.clstr_path, set(self.unique_sequences.keys()))
        self.data['cluster1'] = self.data['ID1'].map(cluster_map)
        self.data['cluster2'] = self.data['ID2'].map(cluster_map)

    def plot_seq_info(self):
        lens = [len(seq) for seq in self.unique_sequences.values()]
        plt.hist(np.array(lens), bins=100)
