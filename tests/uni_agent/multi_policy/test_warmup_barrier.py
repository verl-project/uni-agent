from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from examples.multi_policy.trainer.multi_policy_trainer import MultiPolicyTrainer


class _FakePolicyTrainer:
    def __init__(self, *, hybrid=True, enable_switch=False):
        self._enable_hybrid_replicas = hybrid
        self.hybrid_rollout_config = SimpleNamespace(enable_switch=enable_switch)
        self.switch_calls = 0

    def switch_to_trainer(self):
        self.switch_calls += 1


class _FakeReplayBuffer:
    def __init__(self):
        self.calls = []

    def wait_for_sampleable(self, global_steps, partition_id, target_count):
        self.calls.append((global_steps, partition_id, target_count))
        return set(), {}


def _trainer(
    *,
    barrier=None,
    trainers=None,
    trainer_mode="separate_async",
    replay_buffer=None,
    skip_rollout_tq=False,
):
    trainer = MultiPolicyTrainer.__new__(MultiPolicyTrainer)
    config = {"skip": {"rollout_tq": {"enable": skip_rollout_tq}}}
    if barrier is not None:
        config["warmup_barrier"] = barrier
    trainer.config = OmegaConf.create(config)
    trainer.trainer_mode = trainer_mode
    trainer.train_batch_size = 32
    trainer.global_steps = 3
    trainer.policy_trainers = trainers if trainers is not None else {"p1": _FakePolicyTrainer()}
    trainer.replay_buffer = replay_buffer if replay_buffer is not None else _FakeReplayBuffer()

    def dispatch(hook_name, **kwargs):
        for policy_trainer in trainer.policy_trainers.values():
            getattr(policy_trainer, hook_name)()

    trainer._dispatch_policy_hook = dispatch
    return trainer


@pytest.mark.cpu
@pytest.mark.level0
def test_warmup_barrier_waits_then_switches_all_policies():
    replay_buffer = _FakeReplayBuffer()
    policy_trainers = {"p1": _FakePolicyTrainer(), "p2": _FakePolicyTrainer()}
    trainer = _trainer(barrier={"enable": True}, trainers=policy_trainers, replay_buffer=replay_buffer)

    trainer._wait_for_warmup_barrier(num_warmup_batches=8)

    assert replay_buffer.calls == [(3, "train", 32)]
    assert all(policy_trainer.switch_calls == 1 for policy_trainer in policy_trainers.values())


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("barrier", "trainers", "trainer_mode", "num_warmup_batches"),
    [
        ({"enable": False}, None, "separate_async", 8),
        ({"enable": True}, None, "sync", 8),
        ({"enable": True}, {"p1": _FakePolicyTrainer(hybrid=False)}, "separate_async", 8),
        ({"enable": True}, None, "separate_async", 0),
    ],
)
def test_warmup_barrier_skips_when_not_applicable(
    barrier,
    trainers,
    trainer_mode,
    num_warmup_batches,
):
    replay_buffer = _FakeReplayBuffer()
    policy_trainers = trainers if trainers is not None else {"p1": _FakePolicyTrainer()}
    trainer = _trainer(
        barrier=barrier,
        trainers=policy_trainers,
        trainer_mode=trainer_mode,
        replay_buffer=replay_buffer,
    )

    trainer._wait_for_warmup_barrier(num_warmup_batches)

    assert replay_buffer.calls == []
    assert all(policy_trainer.switch_calls == 0 for policy_trainer in policy_trainers.values())


@pytest.mark.cpu
@pytest.mark.level0
def test_warmup_barrier_skips_when_rollout_tq_skip_enabled():
    replay_buffer = _FakeReplayBuffer()
    policy_trainers = {"p1": _FakePolicyTrainer()}
    trainer = _trainer(
        barrier={"enable": True},
        trainers=policy_trainers,
        replay_buffer=replay_buffer,
        skip_rollout_tq=True,
    )

    trainer._wait_for_warmup_barrier(num_warmup_batches=8)

    assert replay_buffer.calls == []
    assert all(policy_trainer.switch_calls == 0 for policy_trainer in policy_trainers.values())


@pytest.mark.cpu
@pytest.mark.level0
def test_warmup_barrier_skips_below_one_warmup_batch():
    replay_buffer = _FakeReplayBuffer()
    policy_trainers = {"p1": _FakePolicyTrainer()}
    trainer = _trainer(
        barrier={"enable": True},
        trainers=policy_trainers,
        replay_buffer=replay_buffer,
    )

    trainer._wait_for_warmup_barrier(num_warmup_batches=0.5)

    assert replay_buffer.calls == []
    assert all(policy_trainer.switch_calls == 0 for policy_trainer in policy_trainers.values())
