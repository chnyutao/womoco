import gymnasium
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
from torchrl.envs import GymWrapper
from torchrl.envs.transforms import Resize, StepCounter, ToTensorImage

from womoco.typing import Device, Env


class MinigridEnv(Env):
    """Minigrid environments with fully observable, pixel observations."""

    def __init__(self, env_id: str, *, device: Device = 'cpu') -> None:
        self.env_id = env_id
        env = gymnasium.make(env_id, render_mode='rgb_array', tile_size=8)
        env = PixelObservationWrapper(env, pixels_only=True)
        env = GymWrapper(env, device=device)
        super().__init__(env)
        self.append_transform(ToTensorImage(in_keys=['pixels']))
        self.append_transform(Resize(64, 64, in_keys=['pixels']))
        self.append_transform(StepCounter(step_count_key='length'))

    @property
    def id(self) -> str:
        return self.env_id
