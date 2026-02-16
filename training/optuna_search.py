import optuna
from dataclasses import fields
from .params import ModelParamSpace, TrainerParamSpace, FloatParam, CategoricalParam, IntParam
from .pipeline import TrainingPipeline
from pathlib import Path
from typing import Any, Dict
import json

class OptunaSearch:
    def __init__(
            self,
            pipeline: TrainingPipeline,
            model_params: ModelParamSpace,
            trainer_params: TrainerParamSpace,
            study_name: str | None = None,
            base_save_dir: str | Path = "./experiments",
            direction: str = "maximize",
    ):
        self.pipeline = pipeline
        self.model_params = model_params
        self.trainer_params = trainer_params

        base_save_dir = Path(base_save_dir)
        base_save_dir.mkdir(parents=True, exist_ok=True)

        storage = f'sqlite:///{base_save_dir}/optuna.db'
        self.study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )

        self.save_dir = base_save_dir if base_save_dir.name == self.study.study_name else base_save_dir / self.study.study_name
        self._setup_directories()
        self.trial_history = []

    def _setup_directories(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.trial_dir = self.save_dir / 'trials'
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.save_dir / 'checkpoints'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.logging_dir = self.save_dir / 'logging'
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.save_dir / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)

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

    def objective(self, trial: optuna.Trial) -> float:
        # 1. Sample Params
        m_params = self.sample_params(trial, self.model_params)
        t_params = self.sample_params(trial, self.trainer_params)
        all_params = {**m_params, **t_params}

        # 2. Setup Paths
        trial_name = f'trial_{trial.number:04d}'
        trial_ckpt_dir = self.ckpt_dir / trial_name
        trial_log_dir = self.logging_dir / trial_name
        trial_data_dir = self.data_dir / trial_name
        trial_data_dir.mkdir(parents=True, exist_ok=True)

        score = self.pipeline.run(
            params=all_params,
            ckpt_dir=trial_ckpt_dir,
            log_dir=trial_log_dir,
            data_dir=trial_data_dir,
            trial=trial
        )

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


