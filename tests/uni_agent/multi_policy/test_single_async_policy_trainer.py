from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from examples.multi_policy.trainer import single_async_policy_trainer as trainer_module
from examples.multi_policy.trainer.single_async_policy_trainer import SingleAsyncPPOTrainer


class _FakeServerManager:
    def get_replicas(self):
        return []


def _trainer(config, *, enable_hybrid):
    trainer = SingleAsyncPPOTrainer.__new__(SingleAsyncPPOTrainer)
    trainer.config = config
    trainer.llm_server_manager = SimpleNamespace(rollout_replicas=[])
    trainer.actor_rollout_wg = object()
    trainer._enable_hybrid_replicas = enable_hybrid
    trainer.balancer_calls = 0
    trainer.add_replicas_to_balancer = lambda: setattr(trainer, "balancer_calls", trainer.balancer_calls + 1)
    return trainer


@pytest.fixture
def captured_runtime(monkeypatch):
    captured = {}

    class FakeLLMServerManager:
        @classmethod
        def create(cls, *, config, start_rank):
            captured["config"] = config
            captured["start_rank"] = start_rank
            return _FakeServerManager()

    class FakeCheckpointEngineManager:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(trainer_module, "LLMServerManager", FakeLLMServerManager)
    monkeypatch.setattr(trainer_module, "CheckpointEngineManager", FakeCheckpointEngineManager)
    monkeypatch.setattr(trainer_module, "omega_conf_to_dataclass", lambda _config: SimpleNamespace())
    return captured


def _config(*, gpu_memory_utilization=0.6, standalone_gpu_memory_utilization=None):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "standalone_gpu_memory_utilization": standalone_gpu_memory_utilization,
                    "checkpoint_engine": {},
                }
            }
        }
    )


@pytest.mark.cpu
@pytest.mark.level0
def test_standalone_runtime_applies_memory_override(captured_runtime):
    config = _config(standalone_gpu_memory_utilization=0.9)
    trainer = _trainer(config, enable_hybrid=True)

    trainer.init_standalone_rollout_runtime()

    runtime_config = captured_runtime["config"]
    assert runtime_config is not config
    assert runtime_config.actor_rollout_ref.rollout.gpu_memory_utilization == 0.9
    assert config.actor_rollout_ref.rollout.gpu_memory_utilization == 0.6
    assert trainer.current_mode == trainer_module.HybridEngineMode.ROLLOUT
    assert trainer.balancer_calls == 1


@pytest.mark.cpu
@pytest.mark.level0
def test_standalone_runtime_stays_in_trainer_mode_without_hybrid_replicas(captured_runtime):
    config = _config()
    trainer = _trainer(config, enable_hybrid=False)

    trainer.init_standalone_rollout_runtime()

    assert captured_runtime["start_rank"] == 0
    assert trainer.current_mode == trainer_module.HybridEngineMode.TRAINER
    assert trainer.balancer_calls == 0
