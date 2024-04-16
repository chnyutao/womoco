from dataclasses import dataclass


@dataclass
class Config:
    device: str = 'cpu'
    """device to use, typically cpu/cuda"""

    n_envs: int = 4
    """number of environments to run in parallel"""
