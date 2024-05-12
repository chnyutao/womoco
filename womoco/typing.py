from abc import ABC, abstractmethod

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torch.optim import Optimizer
from torchrl.envs import TransformedEnv


class Model(ABC, TensorDictModuleBase):
    @abstractmethod
    def forward(self, x: TensorDictBase) -> TensorDictBase: ...
    @abstractmethod
    def step(self, x: TensorDictBase, opt: Optimizer) -> None: ...


class Env(ABC, TransformedEnv):
    @property
    @abstractmethod
    def name(self) -> str: ...


Device = str | torch.device
