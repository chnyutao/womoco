from typing import Any, Dict, List

import numpy as np
import wandb
from gymnasium.utils.save_video import capped_cubic_video_schedule
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


class MetricsCallback(BaseCallback):
    def __init__(self, envs: List[Monitor], verbose: int = 0) -> None:
        super().__init__(verbose)
        self.envs = envs
        self.n_episodes = 0

    @property
    def env_id(self) -> str:
        return self.training_env.get_attr('spec')[0].id

    def evaluate(self) -> None:
        kwargs = {'n_eval_episodes': 1, 'deterministic': False}
        for env in self.envs:
            # log agent video & reward in the current train env
            if env.spec.id == self.env_id:
                frames = []
                callback = lambda ls, gs: frames.append(env.render().transpose(2, 0, 1))
                reward, _ = evaluate_policy(self.model, env, callback=callback, **kwargs)
                wandb.log({
                    f'{env.spec.id}/eval': reward,
                    f'{self.env_id}/video': wandb.Video(np.array(frames[:-1]))
                }, step=self.num_timesteps)
            # log only reward in other evaluation envs
            else:
                reward, _ = evaluate_policy(self.model, env, **kwargs)
                wandb.log({f'{env.spec.id}/eval': reward}, step=self.num_timesteps)

    def _on_step(self) -> bool:
        infos: List[Dict[str, Any]] = self.locals['infos']
        # log length/reward every episode
        length = np.mean([info['episode']['l'] for info in infos if 'episode' in info])
        reward = np.mean([info['episode']['r'] for info in infos if 'episode' in info])
        if not np.isnan(length) and not np.isnan(reward):
            wandb.log({
                f'{self.env_id}/length': length,
                f'{self.env_id}/reward': reward
            }, step=self.num_timesteps)
        # run evaluation on all envs using cubic schedule
        for info in infos:
            if 'episode' in info:
                self.n_episodes += 1
                if capped_cubic_video_schedule(self.n_episodes):
                    self.evaluate()
        return True

    def _on_training_start(self) -> None:
        self.n_episodes = 0
        return self.evaluate()

    def _on_training_end(self) -> None:
        return self.evaluate()