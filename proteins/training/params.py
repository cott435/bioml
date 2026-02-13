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
    dropout: FloatParam = FloatParam(0.00, 0.3)
    activation: CategoricalParam = CategoricalParam(['relu', 'gelu'])
    layers: IntParam = IntParam(3, 7)
    kernel_size: IntParam = IntParam(3, 7)
    block_type: CategoricalParam = CategoricalParam(['Conv1dInvBottleNeck', 'ConvNeXt1DBlock'])

@dataclass(frozen=True)
class TrainerParamSpace:
    lr: FloatParam = FloatParam(1e-5, 5e-3, log=True)
    weight_decay: FloatParam = FloatParam(1e-4, 5e-2, log=True)
    max_tokens: IntParam | int = 30000
    gamma: FloatParam = FloatParam(1, 4)
    alpha: FloatParam = FloatParam(0.25, 0.8)
    scheduler_type: CategoricalParam = CategoricalParam(['cosine', 'cosine_warmup', 'one_cycle'])


