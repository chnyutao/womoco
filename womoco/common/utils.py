from typing import Any, Dict, List

from torch.optim import SGD, Adam, Optimizer
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from womoco.config import Config
from womoco.envs import MinigridEnv
from womoco.models import PPO
from womoco.typing import Env, Model


def make_collector(env: Env, model: Model, config: Config) -> MultiaSyncDataCollector:
    return MultiaSyncDataCollector(
        [lambda: make_env(env.name, config)] * config.env.n_envs,
        model.get_submodule('policy'),
        frames_per_batch=config.env.step_size * config.env.n_envs,
        total_frames=config.env.n_frames,
    )


def make_env(env_name: str, config: Config) -> Env:
    match config.env.type:
        case 'minigrid':
            return MinigridEnv(env_name=env_name, device=config.device)


def make_envs(config: Config) -> List[Env]:
    return [make_env(env_name, config) for env_name in config.env.names]


def make_model(env: Env, config: Config) -> Model:
    match config.model.name:
        case 'ppo':
            return PPO(env, config.device)


def make_opt(model: Model, config: Config) -> Optimizer:
    params = model.get_submodule('policy').parameters()
    kwargs: Dict[str, Any] = {'lr': config.opt.lr, 'weight_decay': config.opt.weight_decay}
    match config.opt.name:
        case 'Adam':
            return Adam(params, **kwargs)
        case 'SGD':
            return SGD(params, **kwargs)


def make_replay_buffer(config: Config) -> TensorDictReplayBuffer:
    return TensorDictReplayBuffer(
        storage=LazyTensorStorage(config.buffer_capacity),
        sampler=RandomSampler(),
        batch_size=config.batch_size,
    )
