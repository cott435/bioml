import torch
from torch.utils.data import Dataset
from pandas import read_parquet
from .parse import data_dir
from .utils import make_sequence_fasta, cluster_fasta, add_clusters_to_df, missing_esm_ids
from proteins.plotting import plot_seq_info

class SingleSequenceDS(Dataset):

    def __init__(self, data_name, df=None, cluster_coef=0.5, column_map=None, save_dir=data_dir, force=False):
        self.base_dir = save_dir / data_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        data_path = self.base_dir / f'finalized_{cluster_coef}_df.parquet'
        self.data_name = data_name
        self._clstr_path = self.base_dir / f"clustered_{cluster_coef}_sequences.clstr"
        self._fasta_path = self.base_dir / "sequences.fasta"
        self._fasta_path = self._fasta_path if self._fasta_path.exists() else None
        self._clstr_path = self._clstr_path if self._clstr_path.exists() else None
        self.force = force
        self.cluster_coef = cluster_coef

        if data_path.exists() or force:
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
            else make_sequence_fasta(self.data['Sequence'], self.data['ID'], save_dir=self.base_dir, force=self.force)

    @property
    def clstr_path(self):
        return self._clstr_path if self._clstr_path else cluster_fasta(self.fasta_path, force=self.force,
                                                                       cluster_coef=self.cluster_coef)

    def plot_seq_info(self):
        plot_seq_info(self.data['Sequence'], self.data['Y'])


class ESMCEmbeddingDS(SingleSequenceDS):

    def __init__(self, data_name, model_name, df=None, cluster_coef=0.5, column_map=None, save_dir=data_dir,
                 force=False, missing='remove', dtype=torch.float32):
        super().__init__(data_name, df=df, cluster_coef=cluster_coef, column_map=column_map, save_dir=save_dir, force=force)
        assert missing in ['raise', 'remove']
        self.embedding_dir = self.base_dir / model_name
        self.dtype = dtype
        if not self.embedding_dir.exists():
            raise FileNotFoundError(f"Did not find save directory, please create with embed.ESMCForge")
        missing_ids = missing_esm_ids(self.data['ID'].tolist(), self.embedding_dir)
        if len(missing_ids)>0:
            if missing == 'raise':
                raise ValueError(f"Missing ESMC IDs: {missing_ids}")
            elif missing == 'remove':
                self.data = self.data[~self.data['ID'].isin(missing_ids)]
                print('Removed missing ESMC IDs:', missing_ids)
        self.embed_dim = self[0][0].shape[-1]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        active_sites = row['Y']
        emb_file = f"{row['ID']}_embeddings.pt"
        filepath = self.embedding_dir / emb_file
        emb = torch.load(filepath, map_location="cpu").to(self.dtype)
        y = torch.zeros(len(emb)).to(self.dtype)
        y[active_sites] = 1
        return emb, y





