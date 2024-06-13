import warnings

import tyro
from tqdm import tqdm

import wandb

from .common.logging import Logger
from .common.utils import make_collector, make_envs, make_model, make_opt, make_replay_buffer
from .config import Config

warnings.filterwarnings('ignore')

# parse cmd flags
config = tyro.cli(Config)


# init envs
envs = make_envs(config)

# init model & opt
model = make_model(envs[0], config)
opt = make_opt(model, config)

# init logging
wandb.init(config=config.as_dict(), project='womoco')
logger = Logger(envs, model, config)

# init replay buffer
replay_buffer = make_replay_buffer(config)

# main loop
for env in envs:
    collector = make_collector(env, model, config)
    progress = tqdm(total=config.env.n_frames)
    while (data := collector.next()) is not None:
        progress.update(data.numel())
        replay_buffer.extend(data)
        for _ in range(config.n_updates):
            samples = replay_buffer.sample().to(config.device)
            model.step(samples, opt)
        logger.log(env, data, step=progress.n)
    collector.shutdown()
    progress.close()
