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
    dropout: FloatParam = FloatParam(0.05, 0.35)
    activation: CategoricalParam = CategoricalParam(['relu', 'gelu'])
    batch_norm: CategoricalParam = CategoricalParam([True, False])
    layers: IntParam = IntParam(1, 2)
    kernel_size: IntParam = IntParam(1, 4)
    block_type: CategoricalParam = CategoricalParam(['Conv1dInvBottleNeck', 'ConvNeXt1DBlock'])
    inp_norm: bool = True

@dataclass(frozen=True)
class TrainerParamSpace:
    lr: FloatParam = FloatParam(1e-6, 1e-3, log=True)
    weight_decay: FloatParam = FloatParam(1e-6, 1e-2, log=True)
    batch_size: IntParam | int = 32
    loss_reduction: CategoricalParam = CategoricalParam(['mean', 'sum'])


