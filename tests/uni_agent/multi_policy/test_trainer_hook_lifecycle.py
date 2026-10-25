"""CPU tests for the outer trainer's delegated per-policy lifecycle hooks.

``MultiPolicyTrainer`` replaces ``PPOTrainer.fit()`` and drives the per-policy
separate-async trainers directly. ``PPOTrainer.fit()`` normally initializes the
policy runtime state that the delegated hooks read, so the outer trainer must
replicate that initialization. These tests pin the lifecycle contract without
starting Ray or vLLM.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from omegaconf import OmegaConf
import pytest

from examples.multi_policy.trainer.multi_policy_trainer import MultiPolicyTrainer


class _LifecyclePolicyTrainer:
    """Mimic the state contract of ``PPOTrainerSeparateAsync``'s hooks."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.timing_raw: dict[str, float] = {}
        self.global_steps = 0

    def on_step_begin(self) -> None:
        self.calls.append("step_begin")
        # trainer_separate_async.on_step_begin() resets the wait state and
        # records the hybrid switch timer in timing_raw.
        self._step_sample_wait_seconds = 0.0
        self._step_wait_samples = 0
        self.timing_raw["switch_to_trainer"] = 0.0

    def on_sample_begin(self) -> None:
        self.calls.append("sample_begin")
        # trainer_separate_async.on_sample_begin() records the sample start.
        self._sample_start = time.perf_counter()

    def on_sample_end(self) -> None:
        self.calls.append("sample_end")
        # trainer_separate_async.on_sample_end() reads both attributes that
        # only the begin hooks above produce.
        self._step_sample_wait_seconds += time.perf_counter() - self._sample_start

    def on_step_end(self) -> None:
        self.calls.append("step_end")
        # trainer_separate_async.on_step_end() reads the profiling flag that
        # PPOTrainer.fit() normally creates.
        assert self.prev_step_profile is False


def _make_outer_trainer() -> MultiPolicyTrainer:
    trainer = MultiPolicyTrainer.__new__(MultiPolicyTrainer)
    trainer.trainer_mode = "separate_async"
    trainer.config = OmegaConf.create(
        {
            "skip": None,
            "trainer": {"v1": {"separate_async": {"num_warmup_batches": 0}}},
        }
    )
    trainer.policy_trainers = {
        "p1": _LifecyclePolicyTrainer(),
        "p2": _LifecyclePolicyTrainer(),
    }
    trainer._policy_pool = ThreadPoolExecutor(max_workers=2)
    trainer.global_steps = 1
    trainer.timing_raw = {}
    return trainer


@pytest.mark.cpu
@pytest.mark.level0
def test_separate_async_forwards_step_and_sample_begin_hooks():
    trainer = _make_outer_trainer()

    trainer.on_train_begin()
    trainer.on_step_begin()
    trainer.on_sample_begin()
    trainer.on_sample_end()

    for policy_trainer in trainer.policy_trainers.values():
        assert policy_trainer.calls == ["step_begin", "sample_begin", "sample_end"]
    # Step-begin timers are surfaced under the policy prefix.
    assert trainer.timing_raw["p1/switch_to_trainer"] == 0.0
    assert trainer.timing_raw["p2/switch_to_trainer"] == 0.0


@pytest.mark.cpu
@pytest.mark.level0
def test_train_begin_initializes_policy_profiling_state():
    trainer = _make_outer_trainer()

    trainer.on_train_begin()

    for policy_trainer in trainer.policy_trainers.values():
        assert policy_trainer.prev_step_profile is False
        assert policy_trainer.curr_step_profile is False
        assert policy_trainer.next_step_profile is False

    trainer.on_step_end()

    for policy_trainer in trainer.policy_trainers.values():
        assert policy_trainer.calls == ["step_end"]
