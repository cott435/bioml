import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd  # Assuming your data_dict has pandas DataFrames

def make_binary_labels(sequence_length, active_indices):
    labels = [0] * sequence_length  # Initialize with zeros
    for idx in active_indices:
        if 0 <= idx < sequence_length:  # Safety check to avoid index errors
            labels[idx] = 1
    return np.array(labels)

aa_alphabet = list("ACDEFGHIKLMNPQRSTVWY")
AA_one_hot = OneHotEncoder(sparse_output=False)
AA_one_hot.fit(np.array(aa_alphabet).reshape(-1, 1))


class EpitopeDataset(Dataset):
    """
    Custom PyTorch Dataset for epitope sequences.
    - sequences: list of strings (e.g., ['ACDEFG', 'ACDE'])
    - labels: list of lists (e.g., [[1,0,1,0,0,1], [0,1,0,1]])
    """

    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        self.encoder = AA_one_hot

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_array = np.array(list(seq)).reshape(-1, 1)
        one_hot = self.encoder.transform(seq_array)
        one_hot_tensor = torch.tensor(one_hot, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # (L,)
        return one_hot_tensor, label


class DataProcessor:
    """
    Class to process dataframes into PyTorch DataLoaders for epitope prediction.

    Usage:
    - Initialize with data_dict = {'train': df_train, 'valid': df_valid, 'test': df_test}
      - Each df has 'Antigen' (str sequences) and 'Y' (lists of binary ints, or strings to parse)
    - Optionally set max_len (if None, pads dynamically per batch), batch_size, etc.
    - Call get_dataloaders() to get a dict of {'train': loader, 'valid': loader, 'test': loader}

    In your training loop, batches will be (padded_inputs, padded_labels, mask):
    - padded_inputs: (B, max_L_in_batch, 20) one-hot
    - padded_labels: (B, max_L_in_batch) binary floats
    - mask: (B, max_L_in_batch) bool tensor (True for real residues)
    """

    def __init__(self, data_dict, max_len=None, batch_size=32, shuffle_train=True, num_workers=0):
        self.data = {}
        self.max_len = max_len  # If set, truncate/pad to this; else dynamic per batch
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers

        for split, df in data_dict.items():
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"{split} must be a pandas DataFrame")
            sequences = df['Antigen'].tolist()
            labels = []
            for y, seq in zip(df['Y'], sequences):
                seq_len = len(seq)
                label = [0] * seq_len
                for idx in y:
                    if 0 <= idx < seq_len:  # Safety check to avoid index errors
                        label[idx] = 1
                labels.append(np.array(label))
            self.data[split] = EpitopeDataset(sequences, labels)
        if max_len is None:
            print("Using dynamic padding per batch (recommended for variable lengths).")
        else:
            print(f"Using fixed max_len={max_len}; sequences will be truncated/padded to this.")

    def collate_fn(self, batch):
        """
        Custom collate: pad one-hot sequences and labels.
        Handles truncation if max_len set.
        """
        onehots = [item[0] for item in batch]  # List of (L_i, 20)
        labels = [item[1] for item in batch]  # List of (L_i,)

        if self.max_len is not None:
            onehots = [oh[:self.max_len] for oh in onehots]
            labels = [lab[:self.max_len] for lab in labels]

        padded_onehots = pad_sequence(onehots, batch_first=True, padding_value=0.0)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=0.0)

        # Mask: True where not padding (sum over one-hot dim != 0)
        mask = (padded_onehots.sum(dim=-1) != 0)

        if self.max_len is not None:
            # Further pad/truncate to exact max_len if needed (pad_sequence pads to max in batch)
            current_len = padded_onehots.size(1)
            if current_len < self.max_len:
                pad_diff = self.max_len - current_len
                padded_onehots = torch.nn.functional.pad(padded_onehots, (0, 0, 0, pad_diff), value=0.0)
                padded_labels = torch.nn.functional.pad(padded_labels, (0, pad_diff), value=0.0)
                mask = torch.nn.functional.pad(mask, (0, pad_diff), value=False)

        return padded_onehots, padded_labels, mask

    def get_dataloader(self, split):
        """Get DataLoader for a specific split."""
        if split not in self.data:
            raise ValueError(f"Invalid split: {split}. Available: {list(self.data.keys())}")

        shuffle = self.shuffle_train if split == 'train' else False
        return DataLoader(
            self.data[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def get_dataloaders(self):
        """Get dict of all DataLoaders."""
        return {split: self.get_dataloader(split) for split in self.data}

