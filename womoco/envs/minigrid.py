import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from torchrl.envs import EnvBase, GymWrapper, ParallelEnv
from torchrl.envs.transforms import ToTensorImage, TransformedEnv


class MinigridEnv(GymWrapper):
    """Minigrid environments with partially observable, pixel observations."""

    def __init__(self, env_name: str) -> None:
        env = gym.make(env_name, render_mode='rgb_array')
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)
        _ = env.reset()  # required for init gym wrapper
        super().__init__(env)

    @staticmethod
    def make(env_name: str) -> EnvBase:
        """Make a single minigrid environment."""
        return TransformedEnv(
            env=MinigridEnv(env_name),
            transform=ToTensorImage(in_keys=['pixels']),
        )

    @staticmethod
    def make_parallel(env_name: str, num_envs: int) -> EnvBase:
        """Make `num_envs` parallel minigrid environments."""
        return ParallelEnv(num_envs, lambda: MinigridEnv.make(env_name))
