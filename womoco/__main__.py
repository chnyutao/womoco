import warnings

import tyro

from .config import Config
from .envs import MinigridEnv

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    config = tyro.cli(Config)
    env = MinigridEnv.make_parallel(
        env_name='MiniGrid-Empty-5x5-v0',
        n_envs=config.n_envs,
        device=config.device,
    )
    print(env.rollout(10))
