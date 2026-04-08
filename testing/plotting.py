import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ESMPlotter:
    def __init__(
        self,
        dataset,
        n_batch_proteins=None,
        smoothing_window=None,
        whiten=False
    ):
        """
        dataset: dataset[i] -> (x, y)
        n_batch_proteins: number of proteins for batch PCA (None = all)
        smoothing_window: int or None (rolling average window over sequence)
        whiten: whether to use PCA whitening
        """
        self.dataset = dataset
        self.smoothing_window = smoothing_window
        self.whiten = whiten

        self.batch_scaler = StandardScaler()
        self.batch_pca = PCA(n_components=3, whiten=whiten)

        self._fit_batch_pca(n_batch_proteins)

    # -------------------------
    # Preprocessing
    # -------------------------

    def _smooth(self, x):
        if self.smoothing_window is None or self.smoothing_window <= 1:
            return x

        w = self.smoothing_window
        pad = w // 2

        # reflect padding to avoid edge shrinkage
        x_padded = np.pad(x, ((pad, pad), (0, 0)), mode="reflect")

        smoothed = np.zeros_like(x)
        for i in range(len(x)):
            smoothed[i] = x_padded[i:i + w].mean(axis=0)

        return smoothed

    def _preprocess(self, x):
        x = self._smooth(x)
        return x

    # -------------------------
    # Batch PCA fit
    # -------------------------

    def _fit_batch_pca(self, n_batch_proteins):
        xs = []

        n = len(self.dataset) if n_batch_proteins is None else min(n_batch_proteins, len(self.dataset))

        for i in range(n):
            x, _ = self.dataset[i]
            if torch.is_tensor(x):
                x = x.cpu().numpy()

            x = self._preprocess(x)
            xs.append(x)

        X = np.concatenate(xs, axis=0)

        X_scaled = self.batch_scaler.fit_transform(X)
        self.batch_pca.fit(X_scaled)

    # -------------------------
    # Region masks
    # -------------------------

    def _get_regions(self, y):
        if torch.is_tensor(y):
            y = y.cpu().numpy()

        active_idx = np.where(y > 0)[0]

        active_mask = np.zeros_like(y, dtype=bool)
        context_mask = np.zeros_like(y, dtype=bool)

        if len(active_idx) > 0:
            start, end = active_idx[0], active_idx[-1]

            active_mask[start:end + 1] = True

            left = max(0, start - 10)
            right = min(len(y), end + 11)

            context_mask[left:start] = True
            context_mask[end + 1:right] = True

        background_mask = ~(active_mask | context_mask)

        return active_mask, context_mask, background_mask


    def _plot_projection(self, ax, Z, masks, title, x_idx, y_idx):
        active_mask, context_mask, background_mask = masks

        ax.scatter(Z[background_mask, x_idx], Z[background_mask, y_idx],
                   c="lightblue", s=10, alpha=0.6)

        ax.scatter(Z[context_mask, x_idx], Z[context_mask, y_idx],
                   c="purple", s=15, alpha=0.8)

        ax.scatter(Z[active_mask, x_idx], Z[active_mask, y_idx],
                   c="red", s=20, alpha=1.0)

        ax.set_title(title)
        ax.set_xlabel(f"PC{x_idx + 1}")
        ax.set_ylabel(f"PC{y_idx + 1}")

    # -------------------------
    # Main plot
    # -------------------------

    def plot(self, idx):
        x, y = self.dataset[idx]

        if torch.is_tensor(x):
            x = x.cpu().numpy()

        x = self._preprocess(x)

        # Local PCA
        local_scaler = StandardScaler()
        x_local_scaled = local_scaler.fit_transform(x)

        local_pca = PCA(n_components=3, whiten=self.whiten)
        Z_local = local_pca.fit_transform(x_local_scaled)

        # Batch PCA
        x_batch_scaled = self.batch_scaler.transform(x)
        Z_batch = self.batch_pca.transform(x_batch_scaled)

        masks = self._get_regions(y)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # PC1 vs PC2
        self._plot_projection(axes[0, 0], Z_local, masks, "Local PCA: PC1 vs PC2", 0, 1)
        self._plot_projection(axes[0, 1], Z_batch, masks, "Batch PCA: PC1 vs PC2", 0, 1)

        # PC2 vs PC3
        self._plot_projection(axes[1, 0], Z_local, masks, "Local PCA: PC2 vs PC3", 1, 2)
        self._plot_projection(axes[1, 1], Z_batch, masks, "Batch PCA: PC2 vs PC3", 1, 2)

        plt.tight_layout()
        plt.show()