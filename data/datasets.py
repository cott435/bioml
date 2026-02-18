import torch
from .parse import data_dir
from .pre_embed import MultiSequenceDS, SingleSequenceDS
import numpy as np
import h5py


class ESMCSingleDS(SingleSequenceDS):

    def __init__(self, data_name, model_name, df=None, cluster_coef=0.5, column_map=None, save_dir=data_dir,
                 force=False, missing='remove', max_len=5000):
        super().__init__(data_name, df=df, cluster_coef=cluster_coef, column_map=column_map, save_dir=save_dir, force=force)
        assert missing in ['raise', 'remove']
        self.file_path = self.base_dir / f'{model_name}_embeddings.h5'
        if not self.file_path.exists():
            raise FileNotFoundError(f"Did not find save directory, please create with ESMC embedder")
        self.hdf = h5py.File(self.file_path, 'r')
        unique_sequences = {id_: seq for id_, seq in self.unique_sequences.items() if len(seq) < max_len}
        print(f'Dropped {len(self.unique_sequences) - len(unique_sequences)} sequences over len {max_len}')
        stored_ids = set(self.hdf.keys())
        missing = [id_ for id_ in unique_sequences if id_ not in stored_ids]
        n_missing = len(missing)
        if n_missing > 0:
            if missing == 'raise':
                raise ValueError(f"Missing ESMC {n_missing} IDs")
            else:
                self.unique_sequences = {id_: seq for id_, seq in unique_sequences.items() if id_ not in missing}
                print(f'Removed missing {n_missing} ESMC IDs:')
        self.data = self.data[self.data['ID'].isin(unique_sequences.keys())].reset_index(drop=True)
        self.embed_dim = self[0][0].shape[-1]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        active_sites = np.array(row['Y'], copy=True)
        emb_np = self.hdf[row['ID']][:]
        emb = torch.from_numpy(emb_np)
        y = torch.zeros(len(emb))
        y[active_sites] = 1
        return emb, y

    def __len__(self):
        return len(self.data)




