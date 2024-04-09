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
        for env in self.envs:
            reward, _ = evaluate_policy(self.model, env, n_eval_episodes=5)
            wandb.log({f'{env.spec.id}/eval': reward}, step=self.num_timesteps)
            # log agent video in the current train env
            if env.spec.id == self.env_id:
                frames = []
                callback = lambda ls, gs: frames.append(env.render().transpose(2, 0, 1))
                _, _ = evaluate_policy(self.model, env, n_eval_episodes=1, callback=callback)
                wandb.log({
                    f'{self.env_id}/video': wandb.Video(np.array(frames[:-1]))
                }, step=self.num_timesteps)

    def _on_step(self) -> bool:
        # log current env stats every episode
        infos: Dict[str, Any] = self.locals['infos'][0]
        if 'episode' in infos:
            self.n_episodes += 1
            wandb.log({
                f'{self.env_id}/length': infos['episode']['l'],
                f'{self.env_id}/reward': infos['episode']['r']
            }, step=self.num_timesteps)
         # evaluate on all envs using caped cubic schedule (1, 8, 27, ...)
        if 'episode' in infos and capped_cubic_video_schedule(self.n_episodes):
            self.evaluate()
        return True

    def _on_training_start(self) -> None:
        return self.evaluate()

    def _on_training_end(self) -> None:
        return self.evaluate()