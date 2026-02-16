import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, matthews_corrcoef, average_precision_score


class Analysis:
    def __init__(self, train_metrics, val_metrics):
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _normalize_epochs(epochs, total_epochs):
        if isinstance(epochs, list):
            normalized = []
            for e in epochs:
                normalized.append(e if e >= 0 else total_epochs + e)
            return normalized
        return epochs if epochs >= 0 else total_epochs + epochs

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

    def plot_epoch_scores(self):
        train_scores = self._compute_epoch_scores(self.train_metrics)
        val_scores = self._compute_epoch_scores(self.val_metrics)

        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

        axes[0].plot(self.train_metrics['avg_loss'], label="Train")
        axes[0].plot(self.val_metrics['avg_loss'], label="Val")
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

    def plot_logit_separation(self, epoch=-1):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        self._plot_split_logit_separation(
            axes[0],
            self.train_metrics,
            epoch,
            title="Train Logit Separation",
        )
        self._plot_split_logit_separation(
            axes[1],
            self.val_metrics,
            epoch,
            title="Validation Logit Separation",
        )

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

        ax.set_title(title)
        ax.set_xlabel("Logits")
        ax.legend()
