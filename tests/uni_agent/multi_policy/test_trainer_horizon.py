"""CPU tests for the outer trainer's data and optimizer horizon setup."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import examples.multi_policy.trainer.multi_policy_trainer as trainer_module
from examples.multi_policy.trainer.multi_policy_trainer import MultiPolicyTrainer


def _recipe_config():
    return OmegaConf.load(Path(__file__).parents[3] / "examples" / "multi_policy" / "config" / "multi_policy.yaml")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("parameter_sync_step", [1, 2])
def test_policy_configs_receive_optimizer_horizon(parameter_sync_step):
    config = _recipe_config()
    config.trainer.v1.separate_async.parameter_sync_step = parameter_sync_step

    trainer = MultiPolicyTrainer.__new__(MultiPolicyTrainer)
    trainer.config = config
    trainer.total_training_steps = int(config.trainer.total_training_steps)
    trainer.parameter_sync_step = parameter_sync_step
    policy_configs = trainer._compose_policy_configs()

    expected = int(config.trainer.total_training_steps) * parameter_sync_step
    for policy_config in policy_configs.values():
        assert policy_config.actor_rollout_ref.actor.optim.total_training_steps == expected
        assert policy_config.critic.optim.total_training_steps == expected


@pytest.mark.cpu
@pytest.mark.level0
def test_require_total_training_steps_accepts_positive_int():
    assert MultiPolicyTrainer._require_total_training_steps(100) == 100


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("value", [None, 0, -1, "100", 100.5, True])
def test_require_total_training_steps_rejects_invalid(value):
    with pytest.raises(ValueError, match="total_training_steps"):
        MultiPolicyTrainer._require_total_training_steps(value)


@pytest.mark.cpu
@pytest.mark.level0
def test_build_dataloader_sets_steps_per_epoch_and_shuffles_validation(monkeypatch):
    class FakeDataset:
        dataframe = list(range(10))

        def __len__(self):
            return 10

    class FakeLoader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(trainer_module, "create_rl_dataset", lambda *args, **kwargs: FakeDataset())
    monkeypatch.setattr(trainer_module, "create_rl_sampler", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer_module, "StatefulDataLoader", FakeLoader)

    trainer = MultiPolicyTrainer.__new__(MultiPolicyTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {
                "train_files": [],
                "val_files": [],
                "train_batch_size": 4,
                "val_batch_size": 2,
                "dataloader_num_workers": 0,
            },
            "algorithm": {},
            "sampler": {"sync_refill_failed_groups": False},
            "trainer": {
                "v1": {
                    "trainer_mode": "separate_async",
                    "separate_async": {"parameter_sync_step": 1},
                }
            },
        }
    )
    trainer.trainer_mode = "separate_async"
    trainer.train_batch_size = 4
    trainer.total_training_steps = 100
    trainer.policy_trainers = {"p1": SimpleNamespace(tokenizer=None, processor=None)}

    trainer._build_dataloader()

    assert trainer.steps_per_epoch == 2
    assert trainer.val_dataloader.kwargs["shuffle"] is True
