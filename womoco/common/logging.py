from typing import List, cast

import torch

import wandb
from womoco.config import Config
from womoco.typing import Env, Model, TensorDict


class Logger:
    """Logging at every collection step, with periodic evaluation."""

    def __init__(self, envs: List[Env], model: Model, config: Config) -> None:
        self.envs = envs
        self.eval_ckpt = 0
        self.eval_freq = config.eval_freq
        self.policy = model.get_submodule('policy')
        self.step_size = config.env.step_size

    def evaluate(self, *, step: int) -> None:
        for env in self.envs:
            data = cast(TensorDict, env.rollout(self.step_size, self.policy))
            frames = data['pixels'].mul(255).to(torch.uint8).detach().cpu().numpy()
            reward = data['next', 'reward'][-1].item()
            video = wandb.Video(frames, format='gif')
            wandb.log({f'{env.id}/video': video, f'{env.id}/eval': reward}, step=step)

    def log(self, env: Env, data: TensorDict, *, step: int) -> None:
        is_done = data['next', 'done']
        reward = data['next', 'reward'][is_done].mean()
        length = data['next', 'step_count'][is_done].float().mean()
        wandb.log({f'{env.id}/reward': reward, f'{env.id}/length': length}, step=step)
        if step > self.eval_ckpt:
            self.evaluate(step=step)
            self.eval_ckpt += self.eval_freq
