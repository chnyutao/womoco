import gymnasium as gym
import wandb
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from callback import MetricsCallback


run = wandb.init(project='womoco')

env_keys = [
    # 'MiniGrid-Empty-5x5-v0',
    # 'MiniGrid-DoorKey-5x5-v0',
    'MiniGrid-DoorKey-8x8-v0',
    'MiniGrid-LavaCrossingS9N1-v0',
    'MiniGrid-SimpleCrossingS9N1-v0',
]

envs = []
for env_key in env_keys:
    env = gym.make(env_key, render_mode='rgb_array')
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    envs.append(env)

model = None
for env in envs:
    if model is None:
        model = PPO('CnnPolicy', env)
    model.set_env(env)
    model.learn(1e6, callback=MetricsCallback(envs), reset_num_timesteps=False, progress_bar=True)

run.finish()