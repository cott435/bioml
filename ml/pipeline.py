from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    average_precision_score,
    accuracy_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ml.datasets import TokenizedMLDataset
from ml.splits import BaseGroupSplitStrategy


@dataclass(slots=True)
class ModelSpec:
    name: str
    factory: Callable[[int], object]
    use_scaler: bool = True


def default_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="logistic_regression",
            factory=lambda seed: LogisticRegression(class_weight='balanced', random_state=seed, max_iter=1000),
            use_scaler=True,
        ),
        ModelSpec(
            name="svm",
            factory=lambda seed: LinearSVC(class_weight='balanced',random_state=seed, max_iter=5000, dual=False),
            use_scaler=True,
        ),
        ModelSpec(
            name="trees",
            factory=lambda seed: RandomForestClassifier(
                class_weight='balanced',
                n_estimators=300,
                random_state=seed,
                n_jobs=-1,
                max_depth=20,
                min_samples_leaf=10,
                min_samples_split=20,
            ),
            use_scaler=False,
        ),
    ]


class MLBaselinePipeline:
    def __init__(
        self,
        model_specs: list[ModelSpec] | None = None,
        seed: int = 42,
    ):
        self.model_specs = model_specs if model_specs is not None else default_model_specs()
        self.seed = seed

    @staticmethod
    def _score_values(model, x_val: np.ndarray) -> np.ndarray | None:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_val)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            return proba.ravel()
        if hasattr(model, "decision_function"):
            return np.asarray(model.decision_function(x_val)).ravel()
        return None

    @staticmethod
    def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None) -> dict[str, float]:
        metrics: dict[str, float] = {
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'accuracy': float(accuracy_score(y_true, y_pred)),
        }
        if scores is not None and np.unique(y_true).size > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
            metrics['AUPRC'] = float(average_precision_score(y_true, scores))
        return metrics

    def _evaluate(self, model, x_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
        y_pred = np.asarray(model.predict(x_val)).ravel().astype(np.int8)
        scores = self._score_values(model, x_val)
        return self._classification_metrics(y_val, y_pred, scores)

    def run(self, dataset: TokenizedMLDataset, split_strategy: BaseGroupSplitStrategy) -> pd.DataFrame:
        frame = dataset.frame
        x_all = frame.loc[:, dataset.feature_columns].to_numpy(dtype=np.float32, copy=False)
        y_all = frame[dataset.target_column].to_numpy(dtype=np.int8, copy=False)

        results: list[dict[str, float | str]] = []
        i=0
        for split in split_strategy.iter_splits(frame):
            i+=1
            print(f'Starting Split {i}')
            x_train = x_all[split.train_idx]
            y_train = y_all[split.train_idx]
            x_val = x_all[split.val_idx]
            y_val = y_all[split.val_idx]

            for model_spec in self.model_specs:
                model = model_spec.factory(self.seed)
                try:
                    if model_spec.use_scaler:
                        scaler = StandardScaler()
                        x_train_fit = scaler.fit_transform(x_train)
                        x_val_eval = scaler.transform(x_val)
                    else:
                        x_train_fit = x_train
                        x_val_eval = x_val

                    model.fit(x_train_fit, y_train)
                    train_metrics = self._evaluate(model, x_train_fit, y_train)
                    val_metrics = self._evaluate(model, x_val_eval, y_val)
                    results.append(
                        {
                            "split": split.name,
                            "model": model_spec.name,
                            "n_train": int(len(split.train_idx)),
                            "n_val": int(len(split.val_idx)),
                            "status": "ok",
                            **val_metrics,
                            **{f'train_{k}': v for k, v in train_metrics.items()},
                        }
                    )
                    print(f'Finished Model {model_spec.name}')
                except ValueError as exc:
                    results.append(
                        {
                            "split": split.name,
                            "model": model_spec.name,
                            "n_train": int(len(split.train_idx)),
                            "n_val": int(len(split.val_idx)),
                            "status": f"failed: {exc}",
                        }
                    )
        return pd.DataFrame(results)
