import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from proteins.data.utils import pad_collate_fn
from proteins.training.trainers import Trainer


class ClusterPairSplitter:
    """
    Splits paired data based on cluster sets to prevent data leakage.
    Supports both C2 and C3 splitting strategies.

    Parameters:
    -----------
    n_splits : int
        Number of folds for K-Fold.
    split_mode : str
        'C3' (default): Validation pairs must have BOTH clusters unseen.
                        (Hardest - New biology generalization).
                        Discards mixed (Train-Val) pairs.

        'C2':           Validation pairs must have EXACTLY ONE unseen cluster.
                        (Medium - Finding new partners for known proteins).
                        Discards fully unseen (Val-Val) pairs to isolate C2 performance.
    shuffle : bool
        Whether to shuffle clusters before splitting.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(self, n_splits=5, split_mode='C3', shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.split_mode = split_mode.upper()
        self.shuffle = shuffle
        self.random_state = random_state

        if self.split_mode not in ['C2', 'C3']:
            raise ValueError("split_mode must be either 'C2' or 'C3'")

    def split(self, X, y=None, groups=None):
        """
        Yields train_idx and val_idx.

        Args:
            X: Placeholder.
            y: Placeholder.
            groups: Dataframe
        """
        c1, c2 = np.split(groups.values, 2)

        unique_clusters = np.unique(np.concatenate([c1, c2]))
        kfold = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

        for train_clusters_idx, val_clusters_idx in kfold.split(unique_clusters):
            train_cluster_set = set(unique_clusters[train_clusters_idx])
            val_cluster_set = set(unique_clusters[val_clusters_idx])

            c1_in_train = np.isin(c1, list(train_cluster_set))
            c1_in_val = np.isin(c1, list(val_cluster_set))

            c2_in_train = np.isin(c2, list(train_cluster_set))
            c2_in_val = np.isin(c2, list(val_cluster_set))

            train_mask = c1_in_train & c2_in_train

            if self.split_mode == 'C3':  # No group overlap
                val_mask = c1_in_val & c2_in_val

            elif self.split_mode == 'C2':  # One group overlap
                val_mask = (c1_in_train & c2_in_val) | (c1_in_val & c2_in_train)

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]

            yield train_idx, val_idx


def run_cross_validation(model, dataset, split_mode='c3', n_splits=5, device='cpu'):

    """Separate function for full CV – this is the clean place for it."""
    splitter = ClusterPairSplitter(n_splits=n_splits, split_mode=split_mode)
    fold_results = []
    data = dataset.data.dropna(subset=['group_id'])
    for fold, (train_idx, val_idx) in enumerate(splitter.split(data, groups=data[['cluster1', 'cluster2']])):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        loss_weight = data.iloc[train_idx].apply(lambda row: (len(row['Sequence']) - len(row['Y']))/len(row['Y']), axis=1).mean().item()

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=pad_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=64, collate_fn=pad_collate_fn)

        trainer = Trainer(model, train_loader, val_loader, device=device, loss_weight=loss_weight, ckpt_dir=f'./results/fold{fold+1}')
        final_metrics = trainer.train()

        fold_results.append(final_metrics)





