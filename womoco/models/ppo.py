from tensordict.nn import TensorDictModule
from torch.distributions import OneHotCategorical
from torchrl.data import TensorSpec
from torchrl.modules import (
    MLP,
    ActorValueOperator,
    ConvNet,
    ProbabilisticActor,
    SafeProbabilisticTensorDictSequential,
)
from torchrl.modules.tensordict_module.sequence import SafeSequential

from womoco.typing import DeviceType


class PPO(ActorValueOperator):
    """Proximal policy optimization (PPO) modules."""

    def __init__(self, action_spec: TensorSpec, device: DeviceType = 'cpu') -> None:
        # common feature extractor
        common = ConvNet(num_cells=[16, 32, 64], kernel_sizes=3, strides=2).to(device)
        common = TensorDictModule(common, in_keys=['pixels'], out_keys=['hidden'])
        # policy head
        actor = MLP(num_cells=[1024], out_features=action_spec.shape[-1]).to(device)
        actor = TensorDictModule(actor, in_keys=['hidden'], out_keys=['logits'])
        actor = ProbabilisticActor(
            actor,
            in_keys=['logits'],
            out_keys=['action'],
            distribution_class=OneHotCategorical,
        )
        # value head
        critic = MLP(num_cells=[1024], out_features=1).to(device)
        critic = TensorDictModule(critic, in_keys=['hidden'], out_keys=['state_value'])
        super().__init__(common, actor, critic)

    def get_policy_operator(self) -> SafeProbabilisticTensorDictSequential:
        policy = super().get_policy_operator()
        # patching to pass type check
        if not isinstance(policy, SafeProbabilisticTensorDictSequential):
            raise AssertionError()
        policy.register_forward_hook(lambda m, a, o: o.exclude('hidden', 'logits'))
        return policy

    def get_value_operator(self) -> SafeSequential:
        value = super().get_value_operator()
        value.register_forward_hook(lambda m, a, o: o.exclude('hidden'))
        return value
