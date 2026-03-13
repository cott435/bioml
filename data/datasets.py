import torch
from .parse import data_dir
from .pre_embed import MultiSequenceDS, SingleSequenceDS
import numpy as np
import h5py
import matplotlib.pyplot as plt

class ESMCSingleDS(SingleSequenceDS):

    def __init__(self, data_name, model_name, df=None, cluster_coef=0.5, column_map=None, save_dir=data_dir,
                 force=False, missing='remove', max_len=5000):
        super().__init__(data_name, df=df, cluster_coef=cluster_coef, column_map=column_map, save_dir=save_dir, force=force)
        assert missing in ['raise', 'remove']
        self.file_path = self.base_dir / model_name / f'{model_name}_embeddings.h5'
        if not self.file_path.exists():
            raise FileNotFoundError(f"Did not find save directory, please create with ESMC embedder")
        self.hdf = None
        unique_sequences = {id_: seq for id_, seq in self.unique_sequences.items() if len(seq) < max_len}
        print(f'Dropped {len(self.unique_sequences) - len(unique_sequences)} sequences over len {max_len}')
        with h5py.File(self.file_path, 'r') as f:
            stored_ids = set(f.keys())
        missing = [id_ for id_ in unique_sequences if id_ not in stored_ids]
        n_missing = len(missing)
        if n_missing > 0:
            if missing == 'raise':
                raise ValueError(f"Missing ESMC {n_missing} IDs")
            else:
                self.unique_sequences = {id_: seq for id_, seq in unique_sequences.items() if id_ not in missing}
                print(f'Removed missing {n_missing} ESMC IDs:')
        self.data = self.data[self.data['ID'].isin(unique_sequences.keys())].reset_index(drop=True)
        self.ids = self.data["ID"].to_numpy()
        self.labels = self.data["Y"].to_list()
        self.embed_dim = self[0][0].shape[-1]


    def _get_hdf(self):
        if self.hdf is None:
            self.hdf = h5py.File(self.file_path, 'r')
        return self.hdf

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        active_sites = np.array(self.labels[idx], copy=True)
        emb_np = self._get_hdf()[id_][:]
        emb = torch.from_numpy(emb_np)
        y = torch.zeros(emb.shape[0], dtype=torch.float32)
        y.index_fill_(0, torch.tensor(active_sites), 1)
        return emb, y

    def __len__(self):
        return len(self.data)

    def save_full_embedding(self, float16=True):
        emb_list = []
        offsets = []
        lengths = []
        labels = []

        offset = 0

        for i in range(len(self.ids)):
            emb = self._get_hdf()[self.ids[i]][:]
            active_sites = np.array(self.labels[i], copy=True)
            y = np.zeros(emb.shape[0], dtype=np.float32)
            y[active_sites] = 1

            emb_list.append(emb)
            labels.append(y)

            offsets.append(offset)
            lengths.append(len(emb))


            offset += len(emb)
        embeddings = np.concatenate(emb_list, axis=0)
        if float16:
            embeddings = embeddings.astype(np.float16)
        offsets = np.array(offsets)
        lengths = np.array(lengths)
        labels = np.concatenate(labels)

        base = self.file_path.parent
        np.save(base / "embeddings.npy", embeddings)
        np.save(base / "offsets.npy", offsets)
        np.save(base / "lengths.npy", lengths)
        np.save(base / "labels.npy", labels)

    def heatmap(self, idx: int | list):
        if isinstance(idx, list):
            emb = np.concatenate([self[i][0] for i in idx], axis=0).T
        else:
            emb = self[idx][0].T
        plt.figure()
        plt.imshow(emb, aspect='auto', cmap='viridis')
        plt.colorbar(label="Value")
        plt.xlabel("Tokens")
        plt.ylabel("Dim")
        plt.title(f"Protein {idx}")
        plt.show()



class PackedSequenceDataset(SingleSequenceDS, torch.utils.data.Dataset):

    def __init__(self, data_name, model_name, cluster_coef=0.5, column_map=None, save_dir=data_dir,
                 force=False, missing='remove', max_len=5000):
        super().__init__(data_name, cluster_coef=cluster_coef, column_map=column_map, save_dir=save_dir, force=force)
        files_dir = save_dir / data_name / model_name
        self.embeddings = np.load(files_dir / "embeddings.npy", mmap_mode="r")
        self.labels = np.load(files_dir / "labels.npy", mmap_mode="r")
        self.offsets = np.load(files_dir / "offsets.npy")
        self.lengths = np.load(files_dir / "lengths.npy")
        unique_sequences = {id_: seq for id_, seq in self.unique_sequences.items() if len(seq) < max_len}
        self.data = self.data[self.data['ID'].isin(unique_sequences.keys())].reset_index(drop=True)
        self.embed_dim = self.embeddings.shape[1]

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):

        start = self.offsets[idx]
        end = start + self.lengths[idx]

        emb = torch.from_numpy(self.embeddings[start:end].copy()).to(torch.float32)
        y = torch.from_numpy(self.labels[start:end].copy())
        return emb, y



