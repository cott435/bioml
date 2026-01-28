from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Dict, Any, List
from torch.utils.data import Subset, DataLoader
from proteins.data.utils import pad_collate_fn, save_params_as_csv
import numpy as np
import torch
import optuna
from dataclasses import fields
from .params import ModelParamSpace, TrainerParamSpace, FloatParam, CategoricalParam, IntParam
from multiprocessing import cpu_count
print('CPU cores:', cpu_count())

class OptunaGroupedCV:

    def __init__(
        self,
        dataset,
        cv_splitter,
        model_class: Callable,
        trainer_class: Callable,
        model_params: ModelParamSpace,
        trainer_params: TrainerParamSpace,
        n_splits: int = 4,
        direction: str = "maximize",
        study_name: str | None = None,
        base_save_dir: str | Path = "./experiments",
        device: torch.device | str='cpu'
    ):
        self.dataset = dataset
        self.cv_splitter = cv_splitter(n_splits=n_splits)

        self.model_class = model_class
        self.trainer_class = trainer_class
        self.model_params = model_params
        self.trainer_params = trainer_params
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(parents=True, exist_ok=True)
        storage = f'sqlite:///{self.base_save_dir/'optuna.db'}'

        self.study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )

        self.save_dir = self.base_save_dir / self.study.study_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.trial_dir = self.save_dir / 'trials'
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.save_dir / 'checkpoints'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.logging_dir = self.save_dir / 'logging'
        self.logging_dir.mkdir(parents=True, exist_ok=True)

        self.trial_history: List[Dict[str, Any]] = []


    def sample_params(self, trial: optuna.Trial, space) -> Dict[str, Any]:
        params = {}
        for f in fields(space):
            spec = getattr(space, f.name)
            if isinstance(spec, FloatParam):
                params[f.name] = trial.suggest_float(
                    f.name, spec.low, spec.high, log=spec.log
                )
            elif isinstance(spec, IntParam):
                params[f.name] = trial.suggest_int(
                    f.name, spec.low, spec.high, log=spec.log
                )
            elif isinstance(spec, CategoricalParam):
                params[f.name] = trial.suggest_categorical(
                    f.name, list(spec.choices)
                )
            elif isinstance(spec, (int, float)):
                params[f.name] = spec
            else:
                raise TypeError(f"Unsupported param type: {type(spec)}")
        if 'kernel_size' in params:
            params['kernel_size'] = 2 * params['kernel_size'] + 1
        return params


    def objective(self, trial: optuna.Trial) -> float:
        model_params = self.sample_params(trial, self.model_params)
        trainer_params = self.sample_params(trial, self.trainer_params)
        all_params=model_params.copy()
        all_params.update(trainer_params)
        trial_number = f'trial_{trial.number:04d}'
        save_params_as_csv(self.ckpt_dir / trial_number, all_params)

        print(f'Running trial{trial.number:04d} with params: {all_params}')
        fold_scores = []
        bs = trainer_params.pop("batch_size")
        for fold, (train_idx, val_idx) in enumerate(
            self.cv_splitter.split(self.dataset.data, groups=self.dataset.get_data_groups())
        ):
            train_ds = Subset(self.dataset, train_idx)
            val_ds = Subset(self.dataset, val_idx)
            num_workers = cpu_count() //2 if self.device.type == 'cuda' else 0
            prefetch_factor = 2 if self.device.type == 'cuda' else None
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=pad_collate_fn,
                                      pin_memory=torch.cuda.is_available(), num_workers=num_workers, prefetch_factor=prefetch_factor)
            val_loader = DataLoader(val_ds, batch_size=bs*3, collate_fn=pad_collate_fn, prefetch_factor=prefetch_factor,
                                    num_workers=num_workers, pin_memory=torch.cuda.is_available())
            # TODO: add bias to final logit output
            model = self.model_class(self.dataset.embed_dim, **model_params)
            run_name = f'fold_{fold}'
            trainer = self.trainer_class(
                model,
                train_loader,
                val_loader,
                device=self.device,
                ckpt_dir=self.ckpt_dir / trial_number,
                log_dir=self.logging_dir / trial_number / run_name,
                run_name=run_name,
                **trainer_params,
            )
            score=trainer.train()
            fold_scores.append(score)

            trial.report(score, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_score = float(np.mean(fold_scores))
        self._record_trial(trial, all_params, fold_scores, mean_score)
        return mean_score

    def optimize(self, n_trials: int, **kwargs):
        self.study.optimize(self.objective, n_trials=n_trials, **kwargs)
        self._save_summary()

    def _record_trial(
        self,
        trial: optuna.Trial,
        params: Dict[str, Any],
        fold_scores: List[float],
        mean_scores: float,
    ):
        record = {
            "trial": trial.number,
            "params": params,
            "fold_scores": fold_scores,
            "mean_scores": mean_scores,
        }
        self.trial_history.append(record)

        path = self.trial_dir / f"trial_{trial.number:04d}.json"
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

    @property
    def best_params(self) -> Dict[str, Any]:
        return self.study.best_params

    @property
    def best_value(self) -> float:
        return self.study.best_value
