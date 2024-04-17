import warnings

import tyro
from torch.optim import Adam
from torchrl.collectors import SyncDataCollector
from tqdm.auto import tqdm

from .config import Config
from .envs import MinigridEnv
from .models import PPO

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    config = tyro.cli(Config)
    env = MinigridEnv.make_parallel(
        env_name='MiniGrid-Empty-5x5-v0',
        n_envs=config.n_envs,
        device=config.device,
    )
    model = PPO(env, device=config.device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    collector = SyncDataCollector(
        env,
        policy=model.policy,
        frames_per_batch=config.n_envs * 100,
        total_frames=int(1e6),
        split_trajs=False,
        device=config.device,
    )
    for data in tqdm(collector):
        data.to(config.device)
        for _ in range(5):
            model.advantage(data)
            info = model.forward(data)
            loss = info['loss_objective'] + info['loss_critic'] + info['loss_entropy']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'loss: {loss.item()}\treward: {data["next", "reward"].mean()}')
