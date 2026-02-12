from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Dict, Any, List
from torch.utils.data import Subset, DataLoader, SubsetRandomSampler
from proteins.data.utils import pad_collate_fn, save_params_as_csv, bucket_collate_fn
import numpy as np
from proteins.data.sampling import RandomTokenBatchSampler, SortedTokenBatchSampler
import torch
from sklearn.model_selection import GroupShuffleSplit
import optuna
from dataclasses import fields
from .params import ModelParamSpace, TrainerParamSpace, FloatParam, CategoricalParam, IntParam
from multiprocessing import cpu_count
print('CPU cores:', cpu_count())

class OptunaSearch:

    def __init__(
        self,
        dataset,
        model_class: Callable,
        trainer_class: Callable,
        model_params: ModelParamSpace,
        trainer_params: TrainerParamSpace,
        direction: str = "maximize",
        study_name: str | None = None,
        base_save_dir: str | Path = "./experiments",
        device: torch.device | str='cpu',
        test_size=0.2,
        epochs=25
    ):
        self.dataset = dataset
        groups = self.dataset.get_data_groups() if hasattr(self.dataset, 'get_data_groups') else None
        cv_splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        self.cv_splits = next(cv_splitter.split(self.dataset.data, groups=groups))
        self.model_class = model_class
        self.trainer_class = trainer_class
        self.model_params = model_params
        self.trainer_params = trainer_params
        self.epochs = epochs
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
        self.png_dir = self.save_dir / 'images'
        self.png_dir.mkdir(parents=True, exist_ok=True)

        self.trial_history: List[Dict[str, Any]] = []

    @staticmethod
    def sample_params(trial: optuna.Trial, space) -> Dict[str, Any]:
        params = {}
        for f in fields(space):
            spec = getattr(space, f.name)
            if f.name == 'kernel_size':
                params[f.name] = trial.suggest_int(
                    f.name, spec.low, spec.high, log=spec.log, step=2
                )
            elif isinstance(spec, FloatParam):
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
            elif isinstance(spec, (int, float, bool, str)):
                params[f.name] = spec
            else:
                raise TypeError(f"Unsupported param type: {type(spec)}")
        return params

    def _get_loaders(self, train_idx, val_idx, max_tokens):
        train_ds = Subset(self.dataset, train_idx)
        val_ds = Subset(self.dataset, val_idx)
        num_workers = cpu_count() // 2 if self.device.type == 'cuda' else 0
        prefetch_factor = 2 if self.device.type == 'cuda' else None
        lengths = self.dataset.get_lengths()

        train_loader_sampler = RandomTokenBatchSampler(train_ds, lengths[train_idx], max_tokens=max_tokens, drop_last=True)
        val_loader_sampler = SortedTokenBatchSampler(val_ds, lengths[val_idx], max_tokens=max_tokens * 3)

        train_loader = DataLoader(train_ds,
                                  collate_fn=bucket_collate_fn,
                                  batch_sampler=train_loader_sampler,
                                  pin_memory=torch.cuda.is_available(),
                                  num_workers=num_workers,
                                  prefetch_factor=prefetch_factor,
                                  persistent_workers=self.device.type == 'cuda')
        val_loader = DataLoader(val_ds,
                                collate_fn=bucket_collate_fn,
                                batch_sampler=val_loader_sampler,
                                prefetch_factor=prefetch_factor,
                                num_workers=num_workers,
                                pin_memory=torch.cuda.is_available(),
                                persistent_workers=self.device.type == 'cuda')
        return train_loader, val_loader

    def objective(self, trial: optuna.Trial) -> float:
        model_params = self.sample_params(trial, self.model_params)
        trainer_params = self.sample_params(trial, self.trainer_params)
        all_params=model_params.copy()
        all_params.update(trainer_params)
        trial_number = f'trial_{trial.number:04d}'
        save_params_as_csv(self.ckpt_dir / trial_number, all_params)

        print(f'Running trial{trial.number:04d} with params: {all_params}')
        max_tokens = trainer_params.pop("max_tokens")
        train_idx, val_idx = self.cv_splits
        train_loader, val_loader = self._get_loaders(train_idx, val_idx, max_tokens)
        training_data = self.dataset.data.iloc[train_idx]
        pos = training_data['Y'].apply(lambda x: len(x)).sum()
        neg = training_data['Sequence'].apply(lambda x: len(x)).sum() - pos
        output_bias = np.log(pos/neg)
        model = self.model_class(self.dataset.embed_dim, **model_params, out_bias=output_bias)
        trainer = self.trainer_class(
            model,
            train_loader,
            val_loader,
            device=self.device,
            ckpt_dir=self.ckpt_dir / trial_number,
            log_dir=self.logging_dir / trial_number,
            run_name=trial_number,
            epochs=self.epochs,
            **trainer_params,
        )
        score=trainer.train(trial)
        self._record_trial(trial, all_params, score=score)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score

    def optimize(self, n_trials: int, **kwargs):
        self.study.optimize(self.objective, n_trials=n_trials, **kwargs)
        self._save_summary()

    def _record_trial(
        self,
        trial: optuna.Trial,
        params: Dict[str, Any],
        **kwargs
    ):
        record = {
            "trial": trial.number,
            "params": params,
        }
        record.update(kwargs)
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


class Pruner:

    def __init__(self, trial, start=0):
        self.trial = trial
        self.start = start

class OptunaSearchCV(OptunaSearch):

    def __init__(
        self,
        dataset,
        cv_splitter,
        model_class: Callable,
        trainer_class: Callable,
        model_params: ModelParamSpace,
        trainer_params: TrainerParamSpace,
        n_splits: int = 2,
        direction: str = "maximize",
        study_name: str | None = None,
        base_save_dir: str | Path = "./experiments",
        device: torch.device | str='cpu',
    ):
        super().__init__(dataset, model_class, trainer_class, model_params, trainer_params, direction=direction,
                         study_name=study_name, base_save_dir=base_save_dir, device=device)

        groups = self.dataset.get_data_groups() if hasattr(self.dataset, 'get_data_groups') else None
        cv_splitter = cv_splitter(n_splits=n_splits)
        self.cv_splits = [sp for sp in cv_splitter.split(self.dataset.data, groups=groups)]

    def objective(self, trial: optuna.Trial) -> float:
        model_params = self.sample_params(trial, self.model_params)
        trainer_params = self.sample_params(trial, self.trainer_params)
        all_params=model_params.copy()
        all_params.update(trainer_params)
        trial_number = f'trial_{trial.number:04d}'
        save_params_as_csv(self.ckpt_dir / trial_number, all_params)
        epochs=20

        print(f'Running trial{trial.number:04d} with params: {all_params}')
        fold_scores = []
        batch_size = trainer_params.pop("batch_size")
        for fold, (train_idx, val_idx) in enumerate(self.cv_splits):
            train_loader, val_loader = self._get_loaders(train_idx, val_idx, batch_size)
            training_data = self.dataset.data.iloc[train_idx]
            pos = training_data['Y'].apply(lambda x: len(x)).sum()
            neg = training_data['Sequence'].apply(lambda x: len(x)).sum() - pos
            output_bias = np.log(pos/neg)
            model = self.model_class(self.dataset.embed_dim, **model_params, out_bias=output_bias)
            run_name = f'fold_{fold}'
            trainer = self.trainer_class(
                model,
                train_loader,
                val_loader,
                device=self.device,
                ckpt_dir=self.ckpt_dir / trial_number,
                log_dir=self.logging_dir / trial_number / run_name,
                png_dir=self.png_dir / trial_number,
                run_name=run_name,
                epochs=epochs,
                **trainer_params,
            )
            print(f'Training {run_name}')
            # TODO: implement pruning. Make trial holder to keep track of steps across folds
            score=trainer.train()
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        self._record_trial(trial, all_params, fold_scores=fold_scores, mean_score=mean_score)
        return mean_score
