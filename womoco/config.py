from dataclasses import asdict, dataclass, field
from typing import Dict, Literal, Tuple


@dataclass
class EnvConfig:
    names: Tuple[str, ...] = (
        'MiniGrid-SimpleCrossingS9N1-v0',
        'MiniGrid-LavaCrossingS9N1-v0',
        'MiniGrid-DoorKey-8x8-v0',
    )
    """environments for continual learning"""
    n_envs: int = 4
    """number of environments to run in parallel for training"""
    n_frames: int = int(1e6)
    """number of frames allowed in each environment"""
    step_size: int = 128
    """number of frames to collect in  each step"""
    type: Literal['minigrid'] = 'minigrid'
    """environment type"""


@dataclass
class ModelConfig:
    name: Literal['ppo'] = 'ppo'
    """model name"""


@dataclass
class OptConfig:
    lr: float = 1e-4
    """learning rate"""
    name: Literal['Adam', 'SGD'] = 'Adam'
    """optimizer name"""
    weight_decay: float = 0
    """l2 regularization"""


@dataclass
class Config:
    batch_size: int = 64
    """number of samples used for batched gradient descent"""
    buffer_capacity: int = int(1e5)
    """replay buffer capacity"""
    device: str = 'cpu'
    """device to use, typically cpu/cuda"""
    n_updates: int = 8
    """number of parameter updates after each collection step"""

    env: EnvConfig = field(default_factory=EnvConfig)
    """environment configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    """model configuration"""
    opt: OptConfig = field(default_factory=OptConfig)
    """optimizer configuration"""

    def as_dict(self) -> Dict:
        return asdict(self)
