import wandb

from womoco.config import Config
from womoco.typing import Env, TensorDict


class Logger:
    """Generic wandb logger for train/eval metrics, videos, etc."""

    def __init__(self, config: Config) -> None:
        self.step = 0
        self.step_size = config.env.step_size * config.env.n_envs

    def log(self, env: Env, data: TensorDict) -> None:
        self.step += self.step_size
        is_done = data['next', 'done']
        if is_done.sum() > 0:
            reward = data['next', 'reward'][is_done].float().mean()
            length = data['next', 'length'][is_done].float().mean()
            wandb.log({f'{env.id}/reward': reward, f'{env.id}/length': length}, self.step)
