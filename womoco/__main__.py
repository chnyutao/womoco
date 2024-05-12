import warnings

import tyro
from tqdm import tqdm

import wandb

from .common.utils import make_collector, make_envs, make_model, make_opt, make_replay_buffer
from .config import Config

warnings.filterwarnings('ignore')

# parse cmd flags
config = tyro.cli(Config)

# init logging
run = wandb.init(config=config.as_dict(), project='womoco')

# init envs
envs = make_envs(config)

# init model & opt
model = make_model(envs[0], config)
opt = make_opt(model, config)

# init replay buffer
replay_buffer = make_replay_buffer(config)

# main loop
for env in envs:
    progress = tqdm(total=config.env.n_frames)
    collector = make_collector(env, model, config)
    while (data := collector.next()) is not None:
        progress.update(data.numel())
        replay_buffer.extend(data)
        wandb.log({f'{env.name}/reward': data['next', 'reward'][data['next', 'done'].flatten()].mean()})
        for _ in range(config.n_updates):
            samples = replay_buffer.sample()
            model.step(samples, opt)
    collector.shutdown()
    progress.close()
