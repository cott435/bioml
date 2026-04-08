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
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from ml.datasets import TokenizedMLDataset
from ml.splits import BaseGroupSplitStrategy


@dataclass(slots=True)
class ModelSpec:
    name: str
    factory: Callable[[int], object]
    use_scaler: bool = True
    param_grid: dict[str, list[object]] | None = None


def default_model_specs(grid_search: bool = False, include_linear=True, include_trees=True, include_nonlinear_svm: bool = True) -> list[ModelSpec]:
    c_grid = [0.01, 0.1, 1.0, 10.0, 100.0] if grid_search else [1.0]
    tree_depth_grid = [20] if grid_search else [20]
    tree_split_grid = [2000, 4000] if grid_search else [20]

    specs: list[ModelSpec] = []
    if include_linear:
        specs.append(ModelSpec(
                name="logistic_regression",
                factory=lambda seed: LogisticRegression(class_weight='balanced', random_state=seed, max_iter=1000),
                use_scaler=True,
                param_grid={"C": c_grid},
            ))
        specs.append(ModelSpec(
                name="svm_linear",
                factory=lambda seed: LinearSVC(class_weight='balanced',random_state=seed, max_iter=5000, dual=False),
                use_scaler=True,
                param_grid={"C": c_grid},
            )
        )
    if include_trees:
        specs.append(
            ModelSpec(
                name="trees",
                factory=lambda seed: RandomForestClassifier(
                    class_weight='balanced',
                    n_estimators=300,
                    random_state=seed,
                    n_jobs=-1,
                ),
                use_scaler=False,
                param_grid={
                    "max_depth": tree_depth_grid,
                    "min_samples_split": tree_split_grid,
                },
            )
        )
    if include_nonlinear_svm:
        specs.append(
            ModelSpec(
                name="svm_nonlinear",
                factory=lambda seed: SVC(
                    class_weight='balanced',
                    random_state=seed,
                    max_iter=5000,
                ),
                use_scaler=True,
                param_grid={
                    "C": c_grid,
                    "kernel": ["rbf", "poly", "sigmoid"] if grid_search else ["rbf"],
                    "gamma": ["scale", "auto"] if grid_search else ["scale"],
                },
            )
        )
    return specs


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

    @staticmethod
    def _iter_params(model_spec: ModelSpec):
        if model_spec.param_grid:
            yield from ParameterGrid(model_spec.param_grid)
            return
        yield {}

    @staticmethod
    def _param_set_name(params: dict[str, object]) -> str:
        if not params:
            return "default"
        ordered_parts = [f"{k}={params[k]}" for k in sorted(params)]
        return ",".join(ordered_parts)

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
                for params in self._iter_params(model_spec):
                    model = model_spec.factory(self.seed)
                    param_set = self._param_set_name(params)
                    try:
                        if params:
                            if isinstance(model, RandomForestClassifier):
                                params['min_samples_leaf'] = params['min_samples_split'] // 3
                            model.set_params(**params)

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
                                "param_set": param_set,
                                "n_train": int(len(split.train_idx)),
                                "n_val": int(len(split.val_idx)),
                                "status": "ok",
                                **{f"param_{k}": v for k, v in params.items()},
                                **val_metrics,
                                **{f'train_{k}': v for k, v in train_metrics.items()},
                            }
                        )
                        print(f"Finished Model {model_spec.name} ({param_set})")
                    except ValueError as exc:
                        results.append(
                            {
                                "split": split.name,
                                "model": model_spec.name,
                                "param_set": param_set,
                                "n_train": int(len(split.train_idx)),
                                "n_val": int(len(split.val_idx)),
                                "status": f"failed: {exc}",
                                **{f"param_{k}": v for k, v in params.items()},
                            }
                        )
        return pd.DataFrame(results)
