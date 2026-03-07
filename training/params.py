from dataclasses import dataclass
from typing import Any, Sequence, Union, Literal

@dataclass(frozen=True)
class FloatParam:
    low: float
    high: float
    log: bool = False

@dataclass(frozen=True)
class IntParam:
    low: int
    high: int
    log: bool = False

@dataclass(frozen=True)
class CategoricalParam:
    choices: Sequence[Any]

@dataclass(frozen=True)
class ModelParamSpace:
    dropout: float | FloatParam = FloatParam(0.1, 0.5)
    token_dropout: FloatParam = FloatParam(0.00001, 0.00002)
    feature_dropout: FloatParam = FloatParam(0.00001, 0.00002)
    feature_dropout_first: CategoricalParam = CategoricalParam([True, False])

    hidden_dim: int | IntParam = IntParam(64, 256)
    activation: str | CategoricalParam = CategoricalParam(['relu', 'gelu'])
    layers: IntParam | int = IntParam(3, 7)
    kernel_size: IntParam | int = IntParam(3, 7)
    expansion_ratio: IntParam | int = IntParam(1, 3)
    block_type: str | CategoricalParam = CategoricalParam(['Conv1dInvBottleNeck', 'ConvNeXt1DBlock'])
    drop_path_rate: float | FloatParam = FloatParam(0.1, 0.5)
    #inp_norm: CategoricalParam = CategoricalParam(['ln', 'fn', 'bn'])

@dataclass(frozen=True)
class TrainerParamSpace:
    lr: FloatParam = FloatParam(1e-6, 1e-2, log=True)
    weight_decay: FloatParam = FloatParam(1e-2, 0.1, log=True)
    max_tokens: IntParam | int = 30000
    gamma: float | FloatParam = FloatParam(1, 4)
    alpha: float | FloatParam = FloatParam(0.25, 0.75)
    lr_restarts: CategoricalParam = CategoricalParam([True, False])
    jitter: FloatParam = FloatParam(0, 0.01)
    max_norm: FloatParam = FloatParam(0.5, 5)

"""
--- detelte all multi stuff so it starts from scratch
See my scrpts/IEDB_Jespersen.ipynb to see the structure of embedding my data. I need to parse it and pre_embed it and then main.py to see how to run the training pipeline.


build new trainer with codex for 2 if needed. Make general trainer and 2 child classes

"""

from pathlib import Path
import json
import pandas as pd


class TrialExporter:
    """
    Utility for loading Optuna-style trial JSON files and exporting them to Excel.
    """

    def __init__(self, trial_dir: str | Path):
        self.trial_dir = Path(trial_dir)

        if not self.trial_dir.exists():
            raise ValueError(f"Trial directory does not exist: {self.trial_dir}")

        self.trials = []
        self.df = None

    def _flatten_trial(self, trial_dict: dict) -> dict:
        """
        Flatten nested JSON structure into a single row dictionary.
        """
        row = {}

        # basic fields
        row["trial"] = trial_dict.get("trial")
        row["score"] = trial_dict.get("score")

        # parameters
        params = trial_dict.get("params", {})
        for k, v in params.items():
            row[f"param_{k}"] = v

        # memory stats
        memory = trial_dict.get("memory", {})
        for k, v in memory.items():
            row[f"mem_{k}"] = v

        return row

    def load_trials(self):
        """
        Load all trial JSON files from directory.
        """
        rows = []

        for file in sorted(self.trial_dir.glob("*.json")):
            with open(file) as f:
                trial = json.load(f)

            rows.append(self._flatten_trial(trial))

        self.df = pd.DataFrame(rows)

        # Sort by score (best first)
        if "score" in self.df.columns:
            self.df = self.df.sort_values("score", ascending=True)

        return self.df

    def export_excel(self):
        """
        Export loaded trials to Excel.
        """
        if self.df is None:
            raise RuntimeError("Trials not loaded. Run load_trials() first.")

        with pd.ExcelWriter(self.trial_dir/'trials.xlsx', engine="openpyxl") as writer:
            self.df.to_excel(writer, sheet_name="trials", index=False)

