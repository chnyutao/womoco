from dataclasses import asdict, dataclass, field
from typing import Dict, Literal, Tuple


@dataclass
class EnvConfig:
    ids: Tuple[str, ...] = (
        'MiniGrid-SimpleCrossingS9N1-v0',
        'MiniGrid-LavaCrossingS9N1-v0',
        'MiniGrid-DoorKey-8x8-v0',
    )
    """environment ids for continual learning"""
    n_envs: int = 8
    """number of environments to run in parallel for training"""
    n_frames: int = 1_000_000
    """number of frames to collect in each environment"""
    name: Literal['minigrid'] = 'minigrid'
    """environment name"""
    step_size: int = 128
    """number of frames to collect in each environment step"""


@dataclass
class ModelConfig:
    act: str = 'ReLU'
    """activation function"""
    gamma: float = 0.99
    """exponential reward discount factor"""
    lmbda: float = 0.95
    """λ-return discount factor"""
    name: Literal['ppo'] = 'ppo'
    """model name"""


@dataclass
class OptConfig:
    eps: float = 1e-5
    """ε - numerical stability"""
    grad_norm: float = 0.5
    """maximum gradient l2-norm"""
    lr: float = 2e-4
    """learning rate"""
    name: Literal['Adam'] = 'Adam'
    """optimizer name"""
    n_updates: int = 12
    """number of updates after each collection step"""
    weight_decay: float = 0
    """l2 regularization"""


@dataclass
class ReplayConfig:
    buffer_size: int = 1024
    """replay buffer maximum size"""
    batch_size: int = 256
    """number of samples used for batched gradient descent"""


@dataclass
class Config:
    device: str = 'cpu'
    """device to use, typically cpu/cuda"""

    env: EnvConfig = field(default_factory=EnvConfig)
    """environment configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    """model configuration"""
    opt: OptConfig = field(default_factory=OptConfig)
    """optimizer configuration"""
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    """replay configuration"""

    def as_dict(self) -> Dict:
        return asdict(self)
