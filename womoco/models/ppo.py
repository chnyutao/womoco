from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch.distributions import OneHotCategorical
from torchrl.modules import MLP, ConvNet, ProbabilisticActor
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from womoco.typing import Device, Env, Model


class PPO(Model):
    """Proximal policy optimization (PPO)."""

    def __init__(self, env: Env, device: Device = 'cpu') -> None:
        super().__init__()
        # common feature extractor
        common = ConvNet(num_cells=[16, 32, 64], kernel_sizes=3, strides=2).to(device)
        common = TensorDictModule(common, in_keys=['pixels'], out_keys=['hidden'])
        # policy head
        n_actions = env.action_spec.shape[-1]
        policy = MLP(num_cells=[1024], out_features=n_actions).to(device)
        policy = TensorDictModule(policy, in_keys=['hidden'], out_keys=['logits'])
        self.policy = ProbabilisticActor(
            policy,
            in_keys=['logits'],
            out_keys=['action'],
            distribution_class=OneHotCategorical,
            return_log_prob=True,  # required for calculating ppo loss
        )
        self.policy.module.insert(0, common)
        _ = self.policy(env.reset())
        # value head
        value = MLP(num_cells=[1024], out_features=1).to(device)
        value = TensorDictModule(value, in_keys=['hidden'], out_keys=['state_value'])
        self.value = TensorDictSequential(common, value)
        _ = self.value(env.reset())
        # loss function
        # TODO allow config advantage & ppo loss params
        advantage = GAE(gamma=0.99, lmbda=0.95, value_network=self.value, average_gae=True)
        self.loss = ClipPPOLoss(self.policy, self.value)
        self.loss.register_forward_pre_hook(lambda _, args: advantage(args[0]))

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self.loss(tensordict)
