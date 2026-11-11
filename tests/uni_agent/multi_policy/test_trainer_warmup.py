"""CPU tests for the outer trainer's async warmup semantics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from examples.multi_policy.trainer.multi_policy_checkpoint_manager import CheckpointLoadResult
from examples.multi_policy.trainer.multi_policy_trainer import MultiPolicyTrainer


def _make_warmup_trainer(
    *,
    num_warmup_batches: int | float,
    restored_prompts: int = 0,
    skip_enabled: bool = False,
    dataloader_batch_size: int | None = 1,
) -> tuple[MultiPolicyTrainer, list[int]]:
    trainer = MultiPolicyTrainer.__new__(MultiPolicyTrainer)
    trainer.config = OmegaConf.create(
        {
            "skip": {"rollout_tq": {"enable": skip_enabled}},
            "trainer": {"v1": {"separate_async": {"num_warmup_batches": num_warmup_batches}}},
        }
    )
    trainer.train_batch_size = 32
    trainer.train_dataloader = SimpleNamespace(batch_size=dataloader_batch_size)
    trainer._restored_tq_prompt_count = restored_prompts
    added: list[int] = []
    trainer._add_prompts_to_generate = added.append
    return trainer, added


@pytest.mark.cpu
@pytest.mark.level0
def test_add_async_warmup_batches_accepts_fractional_batches():
    trainer, added = _make_warmup_trainer(num_warmup_batches=1.5)

    trainer._add_async_warmup_batches(1.5)

    assert added == [48]


@pytest.mark.cpu
@pytest.mark.level0
def test_add_async_warmup_batches_fresh_start_fills_full_window():
    trainer, added = _make_warmup_trainer(num_warmup_batches=8)

    trainer._add_async_warmup_batches(8)

    assert added == [256]


@pytest.mark.cpu
@pytest.mark.level0
def test_add_async_warmup_batches_skips_complete_restored_window():
    trainer, added = _make_warmup_trainer(num_warmup_batches=8, restored_prompts=256)

    trainer._add_async_warmup_batches(8)

    assert added == []


@pytest.mark.cpu
@pytest.mark.level0
def test_add_async_warmup_batches_tops_up_partial_restored_window():
    trainer, added = _make_warmup_trainer(num_warmup_batches=8, restored_prompts=100)

    trainer._add_async_warmup_batches(8)

    assert added == [156]


@pytest.mark.cpu
@pytest.mark.level0
def test_add_async_warmup_batches_respects_skip():
    trainer, added = _make_warmup_trainer(num_warmup_batches=8, skip_enabled=True)

    trainer._add_async_warmup_batches(8)

    assert added == []


@pytest.mark.cpu
@pytest.mark.level0
def test_add_async_warmup_batches_zero_returns_without_dataloader_access():
    trainer = MultiPolicyTrainer.__new__(MultiPolicyTrainer)
    trainer.config = OmegaConf.create({"skip": None, "trainer": {"v1": {"separate_async": {"num_warmup_batches": 0}}}})

    trainer._add_async_warmup_batches(0)


@pytest.mark.cpu
@pytest.mark.level0
def test_add_async_warmup_batches_negative_returns_without_dataloader_access():
    trainer = MultiPolicyTrainer.__new__(MultiPolicyTrainer)
    trainer.config = OmegaConf.create({"skip": None, "trainer": {"v1": {"separate_async": {"num_warmup_batches": -1}}}})

    trainer._add_async_warmup_batches(-1)


@pytest.mark.cpu
@pytest.mark.level0
def test_add_async_warmup_batches_falls_back_to_train_batch_size():
    trainer, added = _make_warmup_trainer(num_warmup_batches=1, dataloader_batch_size=None)

    trainer._add_async_warmup_batches(1)

    assert added == [32]


@pytest.mark.cpu
@pytest.mark.level0
def test_count_tq_prompt_groups_counts_all_prompt_states(monkeypatch):
    from examples.multi_policy.trainer import multi_policy_checkpoint_manager as checkpoint_manager

    monkeypatch.setattr(
        checkpoint_manager.tq,
        "kv_list",
        lambda partition_id: {
            partition_id: {
                "u1": {"is_prompt": True, "status": "pending"},
                "u2": {"is_prompt": True, "status": "finished"},
                "u3": {"is_prompt": False, "status": "finished"},
            }
        },
    )

    assert checkpoint_manager._count_tq_prompt_groups("train") == 2


@pytest.mark.cpu
@pytest.mark.level0
def test_load_checkpoint_records_restored_prompt_count():
    trainer = MultiPolicyTrainer.__new__(MultiPolicyTrainer)
    trainer.policy_trainers = {}
    trainer.train_dataloader = SimpleNamespace()
    trainer.global_steps = 0
    trainer._restored_tq_prompt_count = 0
    trainer._sync_policy_runtime_context = lambda: None
    trainer.checkpoint_manager = SimpleNamespace(
        load=lambda **_: CheckpointLoadResult(global_step=7, restored_prompt_count=123)
    )

    trainer._load_checkpoint()

    assert trainer.global_steps == 7
    assert trainer._restored_tq_prompt_count == 123


@pytest.mark.cpu
@pytest.mark.level0
def test_load_checkpoint_resets_restored_prompt_count_without_checkpoint():
    trainer = MultiPolicyTrainer.__new__(MultiPolicyTrainer)
    trainer.policy_trainers = {}
    trainer.train_dataloader = SimpleNamespace()
    trainer.global_steps = 5
    trainer._restored_tq_prompt_count = 123
    trainer.checkpoint_manager = SimpleNamespace(load=lambda **_: None)

    trainer._load_checkpoint()

    assert trainer.global_steps == 0
    assert trainer._restored_tq_prompt_count == 0
