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
    hidden_dim: IntParam = IntParam(128, 512)
    dropout: FloatParam = FloatParam(0.1, 0.5)
    activation: CategoricalParam = CategoricalParam(['relu', 'gelu'])
    layers: IntParam = IntParam(3, 7)
    kernel_size: IntParam = IntParam(3, 7)
    expansion_ratio: IntParam = IntParam(1, 3)
    block_type: CategoricalParam = CategoricalParam(['Conv1dInvBottleNeck', 'ConvNeXt1DBlock'])
    layer_scale_init_value: CategoricalParam = CategoricalParam([0.0, 1e-6, 1e-3])
    drop_path_rate: FloatParam = FloatParam(0.1, 0.5)

@dataclass(frozen=True)
class TrainerParamSpace:
    lr: FloatParam = FloatParam(1e-6, 1e-2, log=True)
    weight_decay: FloatParam = FloatParam(1e-4, 5e-2, log=True)
    max_tokens: IntParam | int = 30000
    gamma: FloatParam = FloatParam(1, 4)
    alpha: FloatParam = FloatParam(0.25, 0.8)
    scheduler_type: CategoricalParam = CategoricalParam(['cosine', 'cosine_warmup', 'one_cycle'])
    jitter: FloatParam = FloatParam(0, 0.01)

"""
--- detelte all multi stuff so it starts from scratch
See my scrpts/IEDB_Jespersen.ipynb to see the structure of embedding my data. I need to parse it and pre_embed it and then main.py to see how to run the training pipeline.


build new trainer with codex for 2 if needed. Make general trainer and 2 child classes

"""

