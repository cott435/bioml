import torch
from torch.utils.data import Dataset
from typing import Union
import os
from parse import root_dir

class ESMEmbeddingDataset(Dataset):
    """
    Dataset for precomputed ESM embeddings.
    """

    def __init__(
        self,
        data_name: str,
        model_name: str,
        root_dir=root_dir,
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

        self.emb_dir = os.path.join(root_dir, data_name, model_name)
        self.files = sorted(
            f for f in os.listdir(self.emb_dir) if f.endswith(".pt")
        )

        if not self.files:
            raise RuntimeError("No ESM embeddings found")

        self.repr_layer = repr_layer
        self.include_sequence = include_sequence

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.emb_dir, self.files[idx])
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

