from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from uni_agent.metrics import PromptMetricsSummary


class _FakeReplayBuffer:
    def __init__(self, *, sampler_kwargs, **_kwargs) -> None:
        self.sampler_kwargs = sampler_kwargs

    def _select_prompt_uids(self, _partition_id, sampleable_keys, batch_size):
        selected = sorted(sampleable_keys)[:batch_size]
        return selected, {}, {}

    def _materialize_batch(self, partition_id, selected_prompt_uids, _partition_snapshot):
        return {"partition_id": partition_id, "uids": selected_prompt_uids}

    def sample(self, _global_steps, partition_id, batch_size):
        selected, partition_snapshot, _ = self._select_prompt_uids(partition_id, {"uid-1"}, batch_size)
        return self._materialize_batch(partition_id, selected, partition_snapshot), {"upstream": 1}


class _FakeReplayBufferAsync(_FakeReplayBuffer):
    pass


def _load_adapter(monkeypatch: pytest.MonkeyPatch, prompt_tag: dict):
    calls = {"kv_list": 0}

    def kv_list():
        calls["kv_list"] += 1
        return {"train": {"uid-1": prompt_tag}}

    fake_replay_buffer = ModuleType("verl.trainer.ppo.v1.replay_buffer")
    fake_replay_buffer.ReplayBuffer = _FakeReplayBuffer
    fake_replay_buffer.ReplayBufferAsync = _FakeReplayBufferAsync
    fake_replay_buffer.tq = SimpleNamespace(kv_list=kv_list)
    fake_v1 = ModuleType("verl.trainer.ppo.v1")
    fake_v1.replay_buffer = fake_replay_buffer
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.v1", fake_v1)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.v1.replay_buffer", fake_replay_buffer)

    module_path = Path(__file__).parents[3] / "uni_agent" / "metrics" / "replay_buffer.py"
    spec = importlib.util.spec_from_file_location("_phase6_replay_buffer_adapter", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


def _summary_tag() -> dict:
    summary = PromptMetricsSummary(
        episode_count=1,
        successful_episodes=1,
        empty_episodes=0,
        failed_episodes=0,
        fragment_count=0,
        complete=False,
        metrics={},
        incomplete_reasons=("missing metrics fragment for episode one",),
    )
    return {
        "agent_metrics_summary": summary.to_dict(),
        "agent_metrics_export_owner": "trainer",
    }


def test_primary_adapter_exports_selected_prompt_summary_once(monkeypatch):
    module, calls = _load_adapter(monkeypatch, _summary_tag())
    replay_buffer = module.PromptMetricsReplayBuffer(
        sampler_kwargs={"agent_metrics": {"mode": "primary"}},
    )

    batch, metrics = replay_buffer.sample(1, "train", 1)

    assert batch == {"partition_id": "train", "uids": ["uid-1"]}
    assert metrics["upstream"] == 1
    assert metrics["training/agent_metrics/summary/sum/prompts"] == 1
    assert metrics["training/agent_metrics/summary/sum/incomplete_prompts"] == 1
    assert replay_buffer.last_agent_metrics_incomplete_reasons == ("uid-1: missing metrics fragment for episode one",)
    assert calls["kv_list"] == 1


@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_non_primary_adapter_preserves_upstream_metrics_without_metadata_read(monkeypatch, mode):
    module, calls = _load_adapter(monkeypatch, _summary_tag())
    replay_buffer = module.PromptMetricsReplayBufferAsync(
        sampler_kwargs={"agent_metrics": {"mode": mode}},
    )

    _, metrics = replay_buffer.sample(1, "train", 1)

    assert metrics == {"upstream": 1}
    assert replay_buffer.last_agent_metrics_incomplete_reasons == ()
    assert calls["kv_list"] == 0


def test_primary_adapter_ignores_summary_without_trainer_ownership(monkeypatch):
    tag = _summary_tag()
    del tag["agent_metrics_export_owner"]
    module, calls = _load_adapter(monkeypatch, tag)
    replay_buffer = module.PromptMetricsReplayBuffer(
        sampler_kwargs={"agent_metrics": {"mode": "primary"}},
    )

    _, metrics = replay_buffer.sample(1, "train", 1)

    assert metrics == {"upstream": 1}
    assert calls["kv_list"] == 1
