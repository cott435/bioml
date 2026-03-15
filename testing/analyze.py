from sklearn.metrics import f1_score, matthews_corrcoef, average_precision_score
import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import os
import umap


class ProteinEmbeddingAnalyzer:
    def __init__(self, dataset, lengths=None, embedding_key='embedding', n_components=50):
        """
        Initialize the analyzer.

        :param h5_path: Path to the H5 file containing embeddings.
        :param ids: List of protein IDs to load.
        :param labels: Dict of ID to label (0 for negative, 1 for positive).
        :param lengths: Optional dict of ID to sequence length. If None, attempts to load from H5 under 'length'.
        :param embedding_key: Key in H5 group for the embedding dataset.
        :param n_components: Number of PCA components to compute.
        """
        self.dataset = dataset
        self.df = dataset.data.copy().set_index('ID')
        self.lengths = lengths or {}
        self.embedding_key = embedding_key
        self.n_components = n_components
        self.embeddings = None
        self.pca_model = None
        self.pca_transformed = None
        self._load_data()

    def _load_data(self):
        """Load embeddings, labels, and lengths from H5."""
        embeddings_list = []
        labels_list = []
        lengths_list = []

        for id_ in self.df.index:
            if id_ not in self.dataset.hdf:
                raise ValueError(f"ID {id_} not found in H5 file.")
            group = self.dataset.hdf[id_]
            embedding = np.array(group[:])
            embeddings_list.append(embedding)

            label = self.df.loc[id_]['Y']
            if label is None:
                raise ValueError(f"Label not provided for ID {id_}.")
            full_labels = np.zeros(len(embedding))
            full_labels[label] = 1
            labels_list.append(full_labels)

            lengths_list.append(len(embedding))

        self.embeddings = embeddings_list
        self.labels = labels_list
        self.lengths = lengths_list
        self.all_labels = np.concatenate(labels_list)

    def compute_pca(self):
        """Compute PCA on the embeddings."""
        self.pca_model = PCA(n_components=self.n_components)
        all_embeddings = np.concatenate(self.embeddings)
        self.pca_transformed = self.pca_model.fit_transform(all_embeddings)

    def plot_histogram_lengths(self, save_path=None):
        """Plot histogram of protein lengths."""
        plt.figure(figsize=(10, 6))
        sns.histplot(np.array(self.lengths), bins=30, kde=True)
        plt.title('Histogram of Protein Lengths')
        plt.xlabel('Length')
        plt.ylabel('Frequency')
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def plot_histogram_positives(self, save_path=None):
        """Plot bar chart of number of positives and negatives."""
        positives = np.sum(self.all_labels == 1)
        negatives = np.sum(self.all_labels == 0)
        plt.figure(figsize=(6, 6))
        sns.barplot(x=['Negatives', 'Positives'], y=[negatives, positives])
        plt.title('Number of Positives and Negatives')
        plt.ylabel('Count')
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def plot_pca_scatter(self, pc_x=0, pc_y=1, save_path=None):
        """Plot colored scatter plot for two PCA components, colored by label."""
        if self.pca_transformed is None:
            self.compute_pca()

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(self.pca_transformed[:, pc_x], self.pca_transformed[:, pc_y],
                              c=self.all_labels, cmap='viridis', alpha=0.7)
        plt.title(f'PCA Scatter Plot: PC{pc_x + 1} vs PC{pc_y + 1}')
        plt.xlabel(f'PC{pc_x + 1}')
        plt.ylabel(f'PC{pc_y + 1}')
        plt.colorbar(scatter, label='Label (0: Negative, 1: Positive)')
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def find_best_clustering_components(self, n_top=5):
        """
        Find top N PCA components that best cluster the labels using silhouette score.
        Returns list of (component_index, score) sorted by score descending.
        """
        if self.pca_transformed is None:
            self.compute_pca()

        scores = []
        for i in range(self.n_components):
            # Silhouette score for clustering quality
            score = silhouette_score(self.pca_transformed[:, i].reshape(-1, 1), self.all_labels)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n_top]

    def find_best_separating_components(self, n_top=5):
        """
        Find top N PCA components that best separate positives from negatives using t-test p-value.
        Lower p-value means better separation.
        Returns list of (component_index, p_value) sorted by p_value ascending.
        """
        if self.pca_transformed is None:
            self.compute_pca()

        p_values = []
        pos = self.pca_transformed[self.all_labels == 1]
        neg = self.pca_transformed[self.all_labels == 0]

        for i in range(self.n_components):
            _, p = ttest_ind(pos[:, i], neg[:, i])
            p_values.append((i, p))

        p_values.sort(key=lambda x: x[1])
        return p_values[:n_top]

    def plot_best_separating_scatter(self, comp1, comp2, save_path=None):
        """Plot scatter for two specific components, colored by label."""
        self.plot_pca_scatter(pc_x=comp1, pc_y=comp2, save_path=save_path)

    def run_full_analysis(self, output_dir='analysis_results'):
        """Run all analyses and save plots to output_dir."""
        os.makedirs(output_dir, exist_ok=True)

        self.compute_pca()

        self.plot_histogram_lengths(save_path=os.path.join(output_dir, 'lengths_hist.png'))
        self.plot_histogram_positives(save_path=os.path.join(output_dir, 'positives_bar.png'))

        # Plot first two PCA
        self.plot_pca_scatter(0, 1, save_path=os.path.join(output_dir, 'pca_1_vs_2.png'))

        # Best clustering
        best_cluster = self.find_best_clustering_components()
        print("Best clustering components (silhouette):", best_cluster)

        # Best separating
        best_sep = self.find_best_separating_components()
        print("Best separating components (t-test p-value):", best_sep)

        # Plot top separating pair (first two of top)
        if len(best_sep) >= 2:
            comp1, comp2 = best_sep[0][0], best_sep[1][0]
            self.plot_best_separating_scatter(comp1, comp2, save_path=os.path.join(output_dir, 'best_sep_scatter.png'))

        # Plot all top separating as histograms for separation viz
        for i, (comp, p) in enumerate(best_sep):
            plt.figure(figsize=(10, 6))
            sns.histplot(self.pca_transformed[self.all_labels == 0, comp], color='blue', label='Negative', kde=True)
            sns.histplot(self.pca_transformed[self.all_labels == 1, comp], color='orange', label='Positive', kde=True)
            plt.title(f'Histogram of PC{comp + 1} by Label (p={p:.2e})')
            plt.xlabel(f'PC{comp + 1}')
            plt.ylabel('Density')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'sep_hist_pc{comp + 1}.png'))
            plt.close()



class Analysis:
    def __init__(self, train_metrics, val_metrics, trial_names=None):
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        (
            self.train_trials,
            self.val_trials,
            self.trial_names,
        ) = self._normalize_trials(train_metrics, val_metrics, trial_names)

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _is_metric_dict(metrics):
        return isinstance(metrics, dict) and "logits" in metrics and "labels" in metrics

    @staticmethod
    def _normalize_trials(train_metrics, val_metrics, trial_names=None):
        def to_list(metrics):
            if Analysis._is_metric_dict(metrics):
                return [metrics], None
            if isinstance(metrics, dict) and metrics:
                if all(Analysis._is_metric_dict(v) for v in metrics.values()):
                    return list(metrics.values()), list(metrics.keys())
            if isinstance(metrics, (list, tuple)):
                return list(metrics), None
            raise ValueError("Unsupported metrics format.")

        train_list, train_names = to_list(train_metrics)
        val_list, val_names = to_list(val_metrics)

        if len(train_list) != len(val_list):
            raise ValueError("Train and val trials must have the same length.")

        names = trial_names or train_names or val_names
        if names is None:
            names = [f"trial_{i:04d}" for i in range(len(train_list))]
        if len(names) != len(train_list):
            raise ValueError("trial_names length must match number of trials.")

        return train_list, val_list, list(names)

    @staticmethod
    def _normalize_epochs(epochs, total_epochs):
        if isinstance(epochs, (list, tuple, np.ndarray)):
            normalized = []
            for e in list(epochs):
                normalized.append(e if e >= 0 else total_epochs + e)
            return normalized
        return epochs if epochs >= 0 else total_epochs + epochs

    @staticmethod
    def _mean_series(series_list):
        if not series_list:
            return np.array([])
        max_len = max(len(s) for s in series_list)
        padded = np.full((len(series_list), max_len), np.nan, dtype=float)
        for i, series in enumerate(series_list):
            series = np.asarray(series, dtype=float)
            padded[i, : len(series)] = series
        return np.nanmean(padded, axis=0)

    @staticmethod
    def _compute_epoch_scores(metrics):
        labels = metrics["labels"]
        logits = metrics["logits"]
        mcc, f1, auprc = [], [], []

        for epoch in range(len(logits)):
            probs = Analysis._sigmoid(logits[epoch])
            preds = (probs > 0.5).astype(int)
            mcc.append(matthews_corrcoef(labels[epoch], preds))
            f1.append(f1_score(labels[epoch], preds, zero_division=0))
            auprc.append(average_precision_score(labels[epoch], probs))

        return {
            "mcc": np.array(mcc),
            "f1": np.array(f1),
            "auprc": np.array(auprc),
        }

    def plot_epoch_scores(self, show_mean=True):
        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

        if len(self.train_trials) == 1:
            train_scores = self._compute_epoch_scores(self.train_trials[0])
            val_scores = self._compute_epoch_scores(self.val_trials[0])

            axes[0].plot(self.train_trials[0]['avg_loss'], label="Train")
            axes[0].plot(self.val_trials[0]['avg_loss'], label="Val")
            axes[0].set_ylabel("Loss")
            axes[0].legend()

            axes[1].plot(train_scores["mcc"], label="Train")
            axes[1].plot(val_scores["mcc"], label="Val")
            axes[1].set_ylabel("MCC")
            axes[1].legend()

            axes[2].plot(train_scores["f1"], label="Train")
            axes[2].plot(val_scores["f1"], label="Val")
            axes[2].set_ylabel("F1")
            axes[2].legend()

            axes[3].plot(train_scores["auprc"], label="Train")
            axes[3].plot(val_scores["auprc"], label="Val")
            axes[3].set_ylabel("AUPRC")
            axes[3].set_xlabel("Epoch")
            axes[3].legend()

            fig.suptitle("Train vs Val Metrics Over Epochs")
            plt.tight_layout()
            return fig, axes

        train_scores_list = [self._compute_epoch_scores(m) for m in self.train_trials]
        val_scores_list = [self._compute_epoch_scores(m) for m in self.val_trials]

        train_loss = [m.get('avg_loss', []) for m in self.train_trials]
        val_loss = [m.get('avg_loss', []) for m in self.val_trials]

        for series in train_loss:
            axes[0].plot(series, color="tab:blue", alpha=0.25)
        for series in val_loss:
            axes[0].plot(series, color="tab:orange", alpha=0.25)
        if show_mean:
            axes[0].plot(self._mean_series(train_loss), color="tab:blue", label="Train Mean")
            axes[0].plot(self._mean_series(val_loss), color="tab:orange", label="Val Mean")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        for scores in train_scores_list:
            axes[1].plot(scores["mcc"], color="tab:blue", alpha=0.25)
        for scores in val_scores_list:
            axes[1].plot(scores["mcc"], color="tab:orange", alpha=0.25)
        if show_mean:
            axes[1].plot(
                self._mean_series([s["mcc"] for s in train_scores_list]),
                color="tab:blue",
                label="Train Mean",
            )
            axes[1].plot(
                self._mean_series([s["mcc"] for s in val_scores_list]),
                color="tab:orange",
                label="Val Mean",
            )
        axes[1].set_ylabel("MCC")
        axes[1].legend()

        for scores in train_scores_list:
            axes[2].plot(scores["f1"], color="tab:blue", alpha=0.25)
        for scores in val_scores_list:
            axes[2].plot(scores["f1"], color="tab:orange", alpha=0.25)
        if show_mean:
            axes[2].plot(
                self._mean_series([s["f1"] for s in train_scores_list]),
                color="tab:blue",
                label="Train Mean",
            )
            axes[2].plot(
                self._mean_series([s["f1"] for s in val_scores_list]),
                color="tab:orange",
                label="Val Mean",
            )
        axes[2].set_ylabel("F1")
        axes[2].legend()

        for scores in train_scores_list:
            axes[3].plot(scores["auprc"], color="tab:blue", alpha=0.25)
        for scores in val_scores_list:
            axes[3].plot(scores["auprc"], color="tab:orange", alpha=0.25)
        if show_mean:
            axes[3].plot(
                self._mean_series([s["auprc"] for s in train_scores_list]),
                color="tab:blue",
                label="Train Mean",
            )
            axes[3].plot(
                self._mean_series([s["auprc"] for s in val_scores_list]),
                color="tab:orange",
                label="Val Mean",
            )
        axes[3].set_ylabel("AUPRC")
        axes[3].set_xlabel("Epoch")
        axes[3].legend()

        fig.suptitle("Train vs Val Metrics Over Epochs (All Trials)")
        plt.tight_layout()
        return fig, axes

    def plot_logit_separation(self, epoch=-1):
        if len(self.train_trials) == 1:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            self._plot_split_logit_separation(
                axes[0],
                self.train_trials[0],
                epoch,
                title="Train Logit Separation",
            )
            self._plot_split_logit_separation(
                axes[1],
                self.val_trials[0],
                epoch,
                title="Validation Logit Separation",
            )

            plt.tight_layout()
            return fig, axes

        n_trials = len(self.train_trials)
        fig, axes = plt.subplots(
            n_trials,
            2,
            figsize=(12, 4 * n_trials),
            sharex='col',
        )
        axes = np.atleast_2d(axes)

        axes[0, 0].set_title("Train Logit Separation")
        axes[0, 1].set_title("Validation Logit Separation")

        for i, (train_metrics, val_metrics) in enumerate(zip(self.train_trials, self.val_trials)):
            self._plot_split_logit_separation(
                axes[i, 0],
                train_metrics,
                epoch,
                title=None,
            )
            self._plot_split_logit_separation(
                axes[i, 1],
                val_metrics,
                epoch,
                title=None,
            )
            if self.trial_names:
                axes[i, 0].set_ylabel(self.trial_names[i])

        plt.tight_layout()
        return fig, axes
    def _plot_split_logit_separation(self, ax, metrics, epoch, title):
        logits = metrics["logits"]
        labels = metrics["labels"]
        total_epochs = len(logits)
        epochs = self._normalize_epochs(epoch, total_epochs)

        if isinstance(epochs, list):
            epoch_list = sorted(epochs)
            reds = plt.cm.Reds(np.linspace(0.25, 0.9, len(epoch_list)))
            greens = plt.cm.Greens(np.linspace(0.25, 0.9, len(epoch_list)))
            for e, r, g in zip(epoch_list, reds, greens):
                sns.kdeplot(
                    logits[e][labels[e] == 0],
                    fill=True,
                    color=r,
                    label=f"Non-Binding {e}",
                    ax=ax,
                )
                sns.kdeplot(
                    logits[e][labels[e] == 1],
                    fill=True,
                    color=g,
                    label=f"Binding {e}",
                    ax=ax,
                )
        else:
            e = epochs
            sns.kdeplot(
                logits[e][labels[e] == 0],
                fill=True,
                color="red",
                label="Non-Binding",
                ax=ax,
            )
            sns.kdeplot(
                logits[e][labels[e] == 1],
                fill=True,
                color="green",
                label="Binding",
                ax=ax,
            )

        if title:
            ax.set_title(title)
        ax.set_xlabel("Logits")
        ax.legend()


import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap


def plot_global_latent_space(
        train_tensor, val_tensor,
        train_mask, val_mask,
        train_labels, val_labels
):
    """
    Plots PCA and UMAP projections for the entire batch of train and val tensors.

    Args:
        train_tensor, val_tensor: Tensors of shape (B, seq_len, m_dim)
        train_mask, val_mask: Boolean tensors of shape (B, seq_len)
        train_labels, val_labels: Tensors of shape (B, seq_len) with class labels
    """

    # 1. Apply masks to flatten the batch and extract only valid tokens
    # This converts (B, seq_len, m_dim) -> (N_valid_tokens, m_dim)
    X_train = train_tensor[train_mask].detach().cpu().numpy()
    y_train = train_labels[train_mask].detach().cpu().numpy()

    X_val = val_tensor[val_mask].detach().cpu().numpy()
    y_val = val_labels[val_mask].detach().cpu().numpy()

    # Safety check: ensure there is data to plot
    if len(X_train) == 0 or len(X_val) == 0:
        print("No valid tokens found in the batch.")
        return

    # 2. Initialize and Fit ON THE ENTIRE TRAINING BATCH
    pca = PCA(n_components=2)
    pca_train = pca.fit_transform(X_train)
    pca_val = pca.transform(X_val)

    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_train = reducer.fit_transform(X_train)
    umap_val = reducer.transform(X_val)

    sort_idx_train = np.argsort(y_train)
    sort_idx_val = np.argsort(y_val)

    pca_train_sorted = pca_train[sort_idx_train]
    y_train_sorted = y_train[sort_idx_train]

    pca_val_sorted = pca_val[sort_idx_val]
    y_val_sorted = y_val[sort_idx_val]

    umap_train_sorted = umap_train[sort_idx_train]

    umap_val_sorted = umap_val[sort_idx_val]

    # 3. Create a single 2x2 plot for the global batch distributions
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Global Latent Space Projections - Full Batch', fontsize=16)

    # Plot PCA Train (Top Left)
    axs[0, 0].scatter(pca_train_sorted[:, 0], pca_train_sorted[:, 1], c=y_train_sorted, cmap='coolwarm', alpha=0.5, s=10)
    axs[0, 0].set_title('PCA - Train Batch')
    axs[0, 0].set_xlabel('PC 1')
    axs[0, 0].set_ylabel('PC 2')

    # Plot PCA Val (Top Right)
    axs[0, 1].scatter(pca_val_sorted[:, 0], pca_val_sorted[:, 1], c=y_val_sorted, cmap='coolwarm', alpha=0.5, s=10)
    axs[0, 1].set_title('PCA - Validation Batch')
    axs[0, 1].set_xlabel('PC 1')
    axs[0, 1].set_ylabel('PC 2')

    # Plot UMAP Train (Bottom Left)
    axs[1, 0].scatter(umap_train_sorted[:, 0], umap_train_sorted[:, 1], c=y_train_sorted, cmap='coolwarm', alpha=0.5, s=10)
    axs[1, 0].set_title('UMAP - Train Batch')
    axs[1, 0].set_xlabel('UMAP 1')
    axs[1, 0].set_ylabel('UMAP 2')

    # Plot UMAP Val (Bottom Right)
    axs[1, 1].scatter(umap_val_sorted[:, 0], umap_val_sorted[:, 1], c=y_val_sorted, cmap='coolwarm', alpha=0.5, s=10)
    axs[1, 1].set_title('UMAP - Validation Batch')
    axs[1, 1].set_xlabel('UMAP 1')
    axs[1, 1].set_ylabel('UMAP 2')

    plt.tight_layout()
    plt.show()


