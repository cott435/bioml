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
    batch_norm: CategoricalParam | bool = False
    layers: IntParam = IntParam(2, 6)
    kernel_size: IntParam = IntParam(3, 7)
    block_type: CategoricalParam = CategoricalParam(['Conv1dInvBottleNeck', 'ConvNeXt1DBlock'])
    inp_norm: bool = True

@dataclass(frozen=True)
class TrainerParamSpace:
    lr: FloatParam = FloatParam(1e-5, 5e-3, log=True)
    weight_decay: FloatParam = FloatParam(1e-4, 5e-2, log=True)
    max_tokens: IntParam | int = 25000
    loss_reduction: CategoricalParam = CategoricalParam(['mean', 'sum'])
    gamma = FloatParam(1, 4)
    scheduler_type: CategoricalParam = CategoricalParam(['cosine', 'one_cycle'])
    warmup: CategoricalParam = CategoricalParam([True, False])


