import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, matthews_corrcoef, average_precision_score


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
