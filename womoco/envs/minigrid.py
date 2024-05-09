from typing import Any

import gymnasium
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
from torchrl.envs import GymWrapper, ParallelEnv
from torchrl.envs.transforms import Compose, Resize, ToTensorImage, TransformedEnv

from womoco.typing import Device, Env


class MinigridEnv(TransformedEnv):
    """Minigrid environments with fully observable, pixel observations."""

    def __init__(self, env_name: str, *, device: Device = 'cpu') -> None:
        env = gymnasium.make(env_name, render_mode='rgb_array', highlight=False)
        env = PixelObservationWrapper(env, pixels_only=True)
        env = GymWrapper(env, device=device)
        super().__init__(
            env=env,
            transform=Compose(
                ToTensorImage(in_keys=['pixels']),
                Resize(72, 72, in_keys=['pixels']),
            ),
            device=device,
        )

    @staticmethod
    def make_parallel(env_name: str, *, n_envs: int = 4, **kwargs: Any) -> Env:
        """Make `num_envs` parallel minigrid environments."""
        return ParallelEnv(
            n_envs,
            lambda: MinigridEnv(env_name, **kwargs),
            serial_for_single=True,
            mp_start_method='fork',  # spawn is incompatible with wandb
        )
