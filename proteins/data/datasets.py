import torch
from torch.utils.data import Dataset
from os import path, makedirs
from pandas import read_parquet
from .parse import data_dir
from .utils import make_sequence_fasta, cluster_fasta, add_clusters_to_df, esm_extract_sequences
from proteins.plotting import plot_seq_info

class SingleSequenceDS(Dataset):

    def __init__(self, data_name, df=None, cluster_coef=0.5, column_map=None, save_dir=data_dir, force=False):
        self.base_dir = path.join(save_dir, data_name)
        makedirs(self.base_dir, exist_ok=True)
        data_path = path.join(self.base_dir, f'finalized_{cluster_coef}_df.parquet')
        self.data_name = data_name
        self._clstr_path = path.join(self.base_dir, f"clustered_{cluster_coef}_sequences.clstr")
        self._fasta_path = path.join(self.base_dir, "sequences.fasta")
        self._fasta_path = self._fasta_path if path.exists(self._fasta_path) else None
        self._clstr_path = self._clstr_path if path.exists(self._clstr_path) else None
        self.force = force
        self.cluster_coef = cluster_coef

        if path.exists(data_path) or force:
            self.data = read_parquet(data_path, engine="pyarrow")
        elif df is None:
            raise FileNotFoundError(f"File not found: {self.base_dir} and df=None, please provide data")
        else:
            column_map={} if column_map is None else column_map
            self.data = df.rename(columns=column_map)
            self.data = add_clusters_to_df(self.data, self.clstr_path)
            self.data.to_parquet(data_path, engine="pyarrow")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    @property
    def fasta_path(self):
        return self._fasta_path if self._fasta_path is not None \
            else make_sequence_fasta(self.data['X'], self.data['ID'], save_dir=self.base_dir, force=self.force)

    @property
    def clstr_path(self):
        return self._clstr_path if self._clstr_path else cluster_fasta(self.fasta_path, force=self.force,
                                                                       cluster_coef=self.cluster_coef)

    def plot_seq_info(self):
        plot_seq_info(self.data['X'], self.data['Y'])

class ESM2EmbeddingDS(SingleSequenceDS):
    """legacy code for esm2. Max embed len is 1024"""
    # TODO: add sliding window for longer sequences

    def __init__(self, data_name, model_name, df=None, cluster_coef=0.5, column_map=None, save_dir=data_dir, force=False):
        super().__init__(data_name, df=df, cluster_coef=cluster_coef, column_map=column_map, save_dir=save_dir, force=force)
        self.embedding_dir = path.join(self.base_dir, model_name)
        if not path.exists(self.embedding_dir):
            makedirs(self.embedding_dir, exist_ok=True)
            esm_extract_sequences(model_name, self.fasta_path, self.embedding_dir)
        self.embed_dim = self[0][0].shape[-1]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        active_sites = row['Y']
        emb_file = row['ID'] + ".pt"
        filepath = path.join(self.embedding_dir, emb_file)
        data = torch.load(filepath, map_location="cpu")
        reps = data["representations"]
        repr_layer = max(reps.keys())
        emb = reps[repr_layer]
        if len(row['X']) != len(emb):
            raise IndexError("X and emb have different length")
            breakpoint()
        y = torch.zeros(len(emb))
        y[active_sites] = 1
        return emb, y



class EpitopeDataset(Dataset):
    def __init__(self, data, x_col='X', y_col='Y'):
        self.sequences = data[x_col]
        self.binding_idx = data[y_col]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences.iloc[idx], self.binding_idx.iloc[idx]


