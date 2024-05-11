from abc import ABC, abstractmethod

import torch
import torchrl.envs
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase


class Model(ABC, TensorDictModuleBase):
    @abstractmethod
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase: ...


Env = torchrl.envs.EnvBase
Device = str | torch.device
