import optuna
import gc
import tracemalloc
import itertools
import random
from dataclasses import fields
from .params import ModelParamSpace, TrainerParamSpace, FloatParam, CategoricalParam, IntParam
from .pipeline import TrainingPipeline
from typing import Any, Dict
import torch
from pathlib import Path
import json


class BaseSearch:
    def __init__(
            self,
            pipeline: TrainingPipeline,
            save_name: str,
            base_save_dir: str | Path = "./experiments",
            memory_debug: bool = False,
            memory_debug_top_n: int = 10,
            max_trial_history: int = 0,
    ):
        self.pipeline = pipeline
        self.memory_debug = memory_debug
        self.memory_debug_top_n = memory_debug_top_n
        self.max_trial_history = max_trial_history
        self.trial_history = []

        self.save_name = save_name
        self.save_dir = self._resolve_save_dir(base_save_dir, save_name)
        self._setup_directories()
        if self.memory_debug and not tracemalloc.is_tracing():
            tracemalloc.start()

    @staticmethod
    def _resolve_save_dir(base_save_dir: str | Path, save_name: str) -> Path:
        base_path = Path(base_save_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path if base_path.name == save_name else base_path / save_name

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

    def _trial_paths(self, trial_num: int):
        trial_name = f'trial_{trial_num:04d}'
        trial_ckpt_dir = self.ckpt_dir / trial_name
        trial_log_dir = self.logging_dir / trial_name
        trial_data_dir = self.data_dir / trial_name
        trial_data_dir.mkdir(parents=True, exist_ok=True)
        return trial_ckpt_dir, trial_log_dir, trial_data_dir

    def _memory_report(
            self,
            mem_before: Dict[str, float | None],
            py_before,
    ) -> Dict[str, Any]:
        mem_after = self._memory_snapshot()
        memory_report = {
            "rss_mb_before": mem_before["rss_mb"],
            "rss_mb_after": mem_after["rss_mb"],
            "rss_mb_delta": (
                mem_after["rss_mb"] - mem_before["rss_mb"]
                if mem_before["rss_mb"] is not None and mem_after["rss_mb"] is not None
                else None
            ),
            "cuda_allocated_mb_before": mem_before["cuda_allocated_mb"],
            "cuda_allocated_mb_after": mem_after["cuda_allocated_mb"],
            "cuda_reserved_mb_before": mem_before["cuda_reserved_mb"],
            "cuda_reserved_mb_after": mem_after["cuda_reserved_mb"],
        }
        if py_before is not None and tracemalloc.is_tracing():
            py_after = tracemalloc.take_snapshot()
            top_stats = py_after.compare_to(py_before, "lineno")[:self.memory_debug_top_n]
            memory_report["python_top_allocations"] = [str(stat) for stat in top_stats]
        return memory_report

    def _run_trial(
            self,
            trial_num: int,
            params: Dict[str, Any],
            trial: optuna.Trial | None = None,
    ) -> tuple[float, Dict[str, Any]]:
        mem_before = self._memory_snapshot()
        py_before = tracemalloc.take_snapshot() if self.memory_debug and tracemalloc.is_tracing() else None
        try:
            trial_ckpt_dir, trial_log_dir, trial_data_dir = self._trial_paths(trial_num)
            score = self.pipeline.run(
                params=params,
                ckpt_dir=trial_ckpt_dir,
                log_dir=trial_log_dir,
                data_dir=trial_data_dir,
                trial=trial,
            )
            memory_report = self._memory_report(mem_before, py_before)
            return score, memory_report
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()

    def _record_trial(self, trial_num: int, params: Dict[str, Any], **kwargs):
        record = {
            "trial": trial_num,
            "params": params,
        }
        record.update(kwargs)
        if self.max_trial_history > 0:
            self.trial_history.append(record)
            if len(self.trial_history) > self.max_trial_history:
                self.trial_history = self.trial_history[-self.max_trial_history:]

        path = self.trial_dir / f"trial_{trial_num:04d}.json"
        with open(path, "w") as f:
            json.dump(record, f, indent=2)

    def _save_summary(self, summary: Dict[str, Any]):
        with open(self.save_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    @staticmethod
    def _memory_snapshot() -> Dict[str, float | None]:
        rss_mb = None
        try:
            status_path = Path("/proc/self/status")
            if status_path.exists():
                for line in status_path.read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        kb = float(line.split()[1])
                        rss_mb = kb / 1024.0
                        break
        except Exception:
            rss_mb = None

        cuda_allocated_mb = None
        cuda_reserved_mb = None
        if torch.cuda.is_available():
            cuda_allocated_mb = torch.cuda.memory_allocated() / (1024.0 ** 2)
            cuda_reserved_mb = torch.cuda.memory_reserved() / (1024.0 ** 2)

        return {
            "rss_mb": rss_mb,
            "cuda_allocated_mb": cuda_allocated_mb,
            "cuda_reserved_mb": cuda_reserved_mb,
        }


class GridSearch(BaseSearch):
    def __init__(
            self,
            pipeline: TrainingPipeline,
            params,
            study_name: str | None = None,
            base_save_dir: str | Path = "./experiments",
            direction: str = "maximize",
            memory_debug: bool = False,
            memory_debug_top_n: int = 10,
            max_trial_history: int = 0,
    ):
        self.direction = direction
        self.study_name = study_name or "grid_search"
        self._best_trial: int | None = None
        self._best_value: float | None = None
        self._best_params: Dict[str, Any] = {}

        self.search_space = {}
        self.fixed_params = {}
        for name, value in params.items():
            if isinstance(value, (list, tuple)):
                options = list(value)
                if not options:
                    raise ValueError(f"GridSearch parameter '{name}' has no options.")
                self.search_space[name] = options
            else:
                self.fixed_params[name] = value

        super().__init__(
            pipeline=pipeline,
            save_name=self.study_name,
            base_save_dir=base_save_dir,
            memory_debug=memory_debug,
            memory_debug_top_n=memory_debug_top_n,
            max_trial_history=max_trial_history,
        )

    def _iter_param_sets(self):
        if not self.search_space:
            yield {}
            return

        keys = list(self.search_space.keys())
        values = [self.search_space[k] for k in keys]
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    def _is_better(self, score: float) -> bool:
        if self._best_value is None:
            return True
        if self.direction == "minimize":
            return score < self._best_value
        return score > self._best_value

    def optimize(
            self,
            max_trials: int | None = None,
            shuffle: bool = False,
            seed: int | None = None
    ):
        param_sets = list(self._iter_param_sets())
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(param_sets)
        if max_trials is not None:
            param_sets = param_sets[:max_trials]

        for trial_num, sampled_params in enumerate(param_sets):
            all_params = {**self.fixed_params, **sampled_params}
            score, memory_report = self._run_trial(trial_num=trial_num, params=all_params)

            if self._is_better(score):
                self._best_trial = trial_num
                self._best_value = score
                self._best_params = dict(all_params)

            self._record_trial(trial_num, all_params, score=score, memory=memory_report)

        self._save_summary({
            "best_trial": self._best_trial,
            "best_value": self._best_value,
            "best_params": self._best_params,
        })

    @property
    def best_params(self) -> Dict[str, Any]:
        return self._best_params

    @property
    def best_value(self) -> float | None:
        return self._best_value


class OptunaSearch(BaseSearch):
    def __init__(
            self,
            pipeline: TrainingPipeline,
            model_params: ModelParamSpace,
            trainer_params: TrainerParamSpace,
            study_name: str | None = None,
            base_save_dir: str | Path = "./experiments",
            direction: str = "maximize",
            memory_debug: bool = False,
            memory_debug_top_n: int = 10,
            max_trial_history: int = 0,
    ):
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

        super().__init__(
            pipeline=pipeline,
            save_name=self.study.study_name,
            base_save_dir=base_save_dir,
            memory_debug=memory_debug,
            memory_debug_top_n=memory_debug_top_n,
            max_trial_history=max_trial_history,
        )

    @staticmethod
    def sample_params(trial: optuna.Trial, space) -> Dict[str, Any]:
        params = {}
        for f in fields(space):
            spec = getattr(space, f.name)
            if f.name == 'kernel_size' and isinstance(spec, IntParam):
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
        all_params={}
        try:
            m_params = self.sample_params(trial, self.model_params)
            t_params = self.sample_params(trial, self.trainer_params)
            all_params = {**m_params, **t_params}

            score, memory_report = self._run_trial(
                trial_num=trial.number,
                params=all_params,
                trial=trial,
            )

            for key, value in memory_report.items():
                trial.set_user_attr(key, value)
            self._record_trial(trial.number, all_params, score=score, memory=memory_report)
            if trial.should_prune():
                raise optuna.TrialPruned()
            return score
        finally:
            del all_params

    def optimize(self, n_trials: int, **kwargs):
        self.study.optimize(self.objective, n_trials=n_trials, **kwargs)
        self._save_summary({
            "best_trial": self.study.best_trial.number,
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
        })

    @property
    def best_params(self) -> Dict[str, Any]:
        return self.study.best_params

    @property
    def best_value(self) -> float:
        return self.study.best_value

