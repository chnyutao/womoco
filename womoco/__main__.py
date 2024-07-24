import atexit
import warnings

import tyro
from tqdm import tqdm

from .common.utils import (
    make_collector,
    make_envs,
    make_logger,
    make_model,
    make_opt,
    make_replay_buffer,
)
from .config import Config

warnings.filterwarnings('ignore')

# parse cmd flags
config = tyro.cli(Config)

# init envs
envs = make_envs(config)

# init logging
logger = make_logger(config)

# init model & opt
model = make_model(envs[0], config)
opt = make_opt(model, config)

# init replay buffer
replay_buffer = make_replay_buffer(config)

# main loop
for env in envs:
    collector = make_collector(env, model, config)
    atexit.register(lambda: collector.shutdown())
    progress = tqdm(total=config.env.n_frames)
    for data in collector:
        progress.update(data.numel())
        model.preprocess(data.to(config.device))
        replay_buffer.extend(data.cpu())
        for _ in range(config.opt.n_updates):
            batch = replay_buffer.sample().to(config.device)
            model.step(batch, opt)
        logger.log(env, data)
