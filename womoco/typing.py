from abc import ABC, abstractmethod

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torch.optim import Optimizer
from torchrl.envs import TransformedEnv

Device = str | torch.device
TensorDict = TensorDictBase


class Model(ABC, TensorDictModuleBase):
    @abstractmethod
    def preprocess(self, x: TensorDict) -> None: ...
    @abstractmethod
    def step(self, x: TensorDict, opt: Optimizer) -> None: ...


class Env(ABC, TransformedEnv):
    @property
    @abstractmethod
    def id(self) -> str: ...
