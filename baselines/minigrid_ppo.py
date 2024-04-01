import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from stable_baselines3 import PPO



envs = [
    'MiniGrid-DoorKey-8x8-v0',
    'MiniGrid-LavaCrossingS9N1-v0',
    'MiniGrid-SimpleCrossingS9N1-v0'
]

env = gym.make(envs[-1], render_mode='rgb_array')
env = ImgObsWrapper(RGBImgPartialObsWrapper(env))

model = PPO('CnnPolicy', env, verbose=1)
model.learn(10_000)
