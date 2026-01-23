from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Any, List
from torch.utils.data import Subset, DataLoader
from proteins.data.utils import pad_collate_fn
import numpy as np
import optuna
from sklearn.model_selection import GroupKFold


class OptunaGroupedCV:
    """
    Optuna objective wrapper for custom neural network training
    with grouped K-fold cross-validation.
    """

    def __init__(
        self,
        dataset,
        cv_splitter,
        build_model_fn: Callable[[Dict[str, Any]], Any],
        train_fn: Callable[..., float],
        n_splits: int = 5,
        direction: str = "maximize",
        study_name: str | None = None,
        storage: str | None = None,
        save_dir: str | Path = "./optuna_results",
    ):
        self.dataset = dataset
        self.cv_splitter = cv_splitter

        self.build_model_fn = build_model_fn
        self.train_fn = train_fn

        self.cv = GroupKFold(n_splits=n_splits)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )

        self.trial_history: List[Dict[str, Any]] = []

    # -------------------------
    # Hyperparameter definition
    # -------------------------

    def sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Override this to define the hyperparameter space.
        """
        return {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "hidden_dim": trial.suggest_int("hidden_dim", 64, 512),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        }

    # -------------------------
    # Objective
    # -------------------------

    def objective(self, trial: optuna.Trial) -> float:
        params = self.sample_params(trial)

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(
            self.cv_splitter.split(self.dataset.data, groups=self.dataset.get_data_groups())
        ):
            train_ds = Subset(self.dataset, train_idx)
            val_ds = Subset(self.dataset, val_idx)
            loss_weight = self.dataset.data.iloc[train_idx].apply(lambda row: (len(row['Sequence']) - len(row['Y'])) / len(row['Y']),
                                                     axis=1).mean().item()

            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=pad_collate_fn)
            val_loader = DataLoader(val_ds, batch_size=64, collate_fn=pad_collate_fn)

            model = self.build_model_fn(params)

            score = self.train_fn(
                model=model,
                params=params,
                X_train=self.X[train_idx],
                y_train=self.y[train_idx],
                X_val=self.X[val_idx],
                y_val=self.y[val_idx],
                fold=fold,
                trial=trial,
            )

            fold_scores.append(score)

            # Optional pruning hook
            trial.report(score, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_score = float(np.mean(fold_scores))

        self._record_trial(trial, params, fold_scores, mean_score)

        return mean_score

    # -------------------------
    # Running optimization
    # -------------------------

    def optimize(self, n_trials: int, **kwargs):
        self.study.optimize(self.objective, n_trials=n_trials, **kwargs)
        self._save_summary()

    # -------------------------
    # Persistence
    # -------------------------

    def _record_trial(
        self,
        trial: optuna.Trial,
        params: Dict[str, Any],
        fold_scores: List[float],
        mean_score: float,
    ):
        record = {
            "trial": trial.number,
            "params": params,
            "fold_scores": fold_scores,
            "mean_score": mean_score,
        }
        self.trial_history.append(record)

        path = self.save_dir / f"trial_{trial.number:04d}.json"
        with open(path, "w") as f:
            json.dump(record, f, indent=2)

    def _save_summary(self):
        summary = {
            "best_trial": self.study.best_trial.number,
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
        }
        with open(self.save_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # -------------------------
    # Convenience
    # -------------------------

    @property
    def best_params(self) -> Dict[str, Any]:
        return self.study.best_params

    @property
    def best_value(self) -> float:
        return self.study.best_value
