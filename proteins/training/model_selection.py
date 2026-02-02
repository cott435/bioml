import numpy as np
from sklearn.model_selection import KFold


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







