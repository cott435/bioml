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

