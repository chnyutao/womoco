import torch
from tensordict.nn import TensorDictModule
from torch.distributions import OneHotCategorical
from torch.nn import Sequential
from torch.nn.utils import clip_grad_norm
from torch.optim import Optimizer
from torchrl.modules import MLP, ActorValueOperator, ConvNet, ProbabilisticActor
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from womoco.config import Config
from womoco.typing import Env, Model, TensorDict


class PPO(Model):
    """Proximal policy optimization (PPO)."""

    def __init__(self, env: Env, config: Config) -> None:
        super().__init__()
        self.grad_norm = config.opt.grad_norm
        # common feature extractor
        cnn = ConvNet(
            num_cells=[32, 64, 64],
            kernel_sizes=[8, 4, 3],
            strides=[4, 2, 1],
        )
        mlp = MLP(out_features=512, activate_last_layer=True)
        common = Sequential(cnn, mlp)
        common = TensorDictModule(common, in_keys=['pixels'], out_keys=['hidden'])
        # policy head
        n_actions = env.action_spec.shape[-1]
        policy = MLP(out_features=n_actions)
        policy = TensorDictModule(policy, in_keys=['hidden'], out_keys=['logits'])
        policy = ProbabilisticActor(
            policy,
            in_keys=['logits'],
            out_keys=['action'],
            distribution_class=OneHotCategorical,
            return_log_prob=True,  # required for calculating ppo loss
        )
        # value head
        value = MLP(out_features=1)
        value = TensorDictModule(value, in_keys=['hidden'], out_keys=['state_value'])
        # actor-critic model
        model = ActorValueOperator(
            common_operator=common,
            policy_operator=policy,
            value_operator=value,
        ).to(config.device)
        _ = model(env.reset())
        self.policy = model.get_policy_operator()
        self.value = model.get_value_operator()
        # value estimator
        self.gae = GAE(
            gamma=config.model.gamma,
            lmbda=config.model.lmbda,
            value_network=self.value,
            average_gae=False,
        )
        # loss function
        self.loss = ClipPPOLoss(
            self.policy,  # type: ignore
            self.value,
            loss_critic_type='l2',
            normalize_advantage=True,
        )

    @torch.no_grad
    def preprocess(self, x: TensorDict) -> None:
        return self.gae(x)

    def step(self, x: TensorDict, opt: Optimizer) -> None:
        """Update model params once with graident descent."""
        loss = self.loss(x)
        loss = loss['loss_objective'] + loss['loss_critic'] + loss['loss_entropy']
        loss.backward()
        clip_grad_norm(self.parameters(), self.grad_norm)
        opt.step()
        opt.zero_grad()
