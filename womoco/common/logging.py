from typing import List, cast

import torch

import wandb
from womoco.config import Config
from womoco.typing import Env, Model, TensorDict


class Logger:
    """Logging at every collection step, with periodic evaluation."""

    def __init__(self, envs: List[Env], model: Model, config: Config) -> None:
        self.envs = envs
        self.policy = model.get_submodule('policy')
        # internal states
        self._eval_ckpt = 0
        self._eval_freq = config.eval_freq
        self._rollout_size = config.env.step_size
        self._step = 0
        self._step_size = config.env.step_size * config.env.n_envs

    def evaluate(self) -> None:
        for env in self.envs:
            data = cast(TensorDict, env.rollout(self._rollout_size, self.policy))
            frames = data['pixels'].mul(255).to(torch.uint8).detach().cpu().numpy()
            reward = data['next', 'episode_reward'][-1].item()
            video = wandb.Video(frames, format='gif')
            wandb.log({f'{env.id}/video': video, f'{env.id}/eval': reward}, self._step)

    def log(self, env: Env, data: TensorDict) -> None:
        self._step += self._step_size
        is_done = data['next', 'done']
        if is_done.sum() > 0:
            reward = data['next', 'episode_reward'][is_done].mean()
            length = data['next', 'length'][is_done].float().mean()
            wandb.log({f'{env.id}/reward': reward, f'{env.id}/length': length}, self._step)
        if self._step > self._eval_ckpt:
            self.evaluate()
            self._eval_ckpt += self._eval_freq
