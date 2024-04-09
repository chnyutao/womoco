import multiprocessing

import gymnasium as gym
import wandb
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from callback import MetricsCallback


def make_env(env_key: str) -> gym.Env:
    env = gym.make(env_key, render_mode='rgb_array')
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    return Monitor(env)


run = wandb.init(project='womoco')

env_keys = [
    # 'MiniGrid-Empty-5x5-v0',
    # 'MiniGrid-DoorKey-5x5-v0',
    'MiniGrid-SimpleCrossingS9N1-v0',
    'MiniGrid-LavaCrossingS9N1-v0',
    'MiniGrid-DoorKey-8x8-v0',
]
eval_envs = [make_env(env_key) for env_key in env_keys]

callback=MetricsCallback(eval_envs)
model = None
n_cpus = min(multiprocessing.cpu_count(), 4)
for env_key in env_keys:
    env = SubprocVecEnv([lambda: make_env(env_key)] * n_cpus, start_method='fork')
    if model is None:
        model = PPO('CnnPolicy', env)
    model.set_env(env)
    model.learn(1e6, callback=callback, reset_num_timesteps=False, progress_bar=True)
    env.close()

run.finish()