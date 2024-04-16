import os
from unittest.mock import patch

import torch
from torchrl.envs import EnvBase, GymWrapper, ParallelEnv
from torchrl.envs.transforms import ToTensorImage, TransformedEnv

DeviceType = str | torch.device


class MinigridEnv(TransformedEnv):
    """Minigrid environments with partially observable, pixel observations."""

    def __init__(self, env_name: str, device: DeviceType = 'cpu') -> None:
        # patching to disable pygame messages
        with patch.dict(os.environ, {'PYGAME_HIDE_SUPPORT_PROMPT': 'hide'}):
            import gymnasium
            from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
        env = gymnasium.make(env_name, render_mode='rgb_array')
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)
        _ = env.reset()  # required for init gym wrapper
        env = GymWrapper(env)
        super().__init__(
            env=env,
            transform=ToTensorImage(in_keys=['pixels']),
            device=device,
        )

    @staticmethod
    def make_parallel(
        env_name: str, *, n_envs: int = 4, device: DeviceType = 'cpu'
    ) -> EnvBase:
        """Make `num_envs` parallel minigrid environments."""
        return ParallelEnv(n_envs, lambda: MinigridEnv(env_name, device))
