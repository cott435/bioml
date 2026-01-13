import torch
from torch.utils.data import Dataset
from typing import Union
from os import path, makedirs, listdir
from pandas import Series, read_parquet
from .parse import data_dir, make_sequence_fasta, cluster_fasta, add_clusters_to_df

class SingleSequenceDS(Dataset):

    def __init__(self, data_name, df=None, column_map=None, root_dir=data_dir, lazy=True, force=False):
        self.base_dir = path.join(root_dir, data_name)
        makedirs(self.base_dir, exist_ok=True)
        data_path = path.join(self.base_dir, 'finalized_df.parquet')
        self.data_name = data_name
        self._clstr_path = path.join(self.base_dir, "clustered_sequences")
        self._fasta_path = path.join(self.base_dir, "sequences.fasta")
        self._fasta_path = self._fasta_path if path.exists(self._fasta_path) else None
        self._clstr_path = self._clstr_path if path.exists(self._clstr_path) else None
        self.force = force

        if path.exists(data_path) or force:
            self.data = read_parquet(data_path, engine="pyarrow")
        elif df is None:
            raise FileNotFoundError(f"File not found: {self.base_dir} and df=None, please provide data")
        else:
            column_map={} if column_map is None else column_map
            self.data = add_clusters_to_df(df.rename(columns=column_map), self.clstr_path)

        if not lazy:
            f,p = self.fasta_path, self.clstr_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    @property
    def fasta_path(self):
        return self._fasta_path if self._fasta_path is None \
            else make_sequence_fasta(self._df['X'], self._df['ID'], save_dir=self.base_dir, force=self.force)

    @property
    def clstr_path(self):
        return self._clstr_path if self._clstr_path else cluster_fasta(self.fasta_path, force=self.force)






class ESMEmbeddingDataset(Dataset):
    """
    Dataset for precomputed ESM embeddings.
    """

    def __init__(
        self,
        data_name: str,
        model_name: str,
        root_dir=data_dir,
        repr_layer: Union[int, None] = None,
        include_sequence: bool = False,
    ):
        """
        Parameters
        ----------
        data_name : str
            Dataset identifier
        model_name : str
            ESM model name
        repr_layer : int or None
            Which transformer layer to load (None = infer if only one exists)
        include_sequence : bool
            If True, also return raw amino acid sequence
        """

        self.emb_dir = path.join(root_dir, data_name, model_name)
        self.files = sorted(
            f for f in listdir(self.emb_dir) if f.endswith(".pt")
        )

        if not self.files:
            raise RuntimeError("No ESM embeddings found")

        self.repr_layer = repr_layer
        self.include_sequence = include_sequence

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = path.join(self.emb_dir, self.files[idx])
        data = torch.load(path, map_location="cpu")

        reps = data["representations"]

        # Auto-select layer if needed
        if self.repr_layer is None:
            if len(reps) != 1:
                raise ValueError("Multiple layers present; specify repr_layer")
            layer = next(iter(reps))
        else:
            layer = self.repr_layer

        emb = reps[layer]

        if self.include_sequence:
            return emb, data["sequence"]

        return emb


class EpitopeDataset(Dataset):
    def __init__(self, data, x_col='X', y_col='Y'):
        self.sequences = data[x_col]
        self.binding_idx = data[y_col]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences.iloc[idx], self.binding_idx.iloc[idx]


