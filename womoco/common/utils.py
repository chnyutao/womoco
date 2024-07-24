from typing import Any, Dict, List

import wandb
from torch.optim import Adam, Optimizer
from torchrl.collectors import MultiSyncDataCollector as DataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyMemmapStorage

from womoco.common.logging import Logger
from womoco.config import Config
from womoco.envs import MinigridEnv
from womoco.models import PPO
from womoco.typing import Env, Model


def make_collector(env: Env, model: Model, config: Config) -> DataCollector:
    return DataCollector(
        [lambda: make_env(env.id, config)] * config.env.n_envs,
        model.get_submodule('policy'),
        frames_per_batch=config.env.step_size * config.env.n_envs,
        total_frames=config.env.n_frames,
        storing_device='cpu',
        env_device=config.device,
        policy_device=config.device,
    )


def make_env(env_id: str, config: Config) -> Env:
    match config.env.name:
        case 'minigrid':
            return MinigridEnv(env_id=env_id, device=config.device)


def make_envs(config: Config) -> List[Env]:
    return [make_env(env_id, config) for env_id in config.env.ids]


def make_logger(config: Config) -> Logger:
    wandb.init(config=config.as_dict(), project='womoco')
    return Logger(config)


def make_model(env: Env, config: Config) -> Model:
    match config.model.name:
        case 'ppo':
            return PPO(env, config)


def make_opt(model: Model, config: Config) -> Optimizer:
    kwargs: Dict[str, Any] = {
        'eps': config.opt.eps,
        'lr': config.opt.lr,
        'weight_decay': config.opt.weight_decay,
    }
    match config.opt.name:
        case 'Adam':
            return Adam(model.parameters(), **kwargs)


def make_replay_buffer(config: Config) -> TensorDictReplayBuffer:
    return TensorDictReplayBuffer(
        storage=LazyMemmapStorage(config.replay.buffer_size),
        sampler=SamplerWithoutReplacement(),
        batch_size=config.replay.batch_size,
    )
