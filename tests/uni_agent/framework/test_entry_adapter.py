from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from uni_agent.framework import entry


def _recording_worker(worker_id: int, dispatched: list[tuple[int, list[int]]]):
    def remote(chunk):
        dispatched.append((worker_id, chunk["index"].tolist()))
        return worker_id

    return SimpleNamespace(generate_sequences=SimpleNamespace(remote=remote))


def _prompts(*indices: int) -> TensorDict:
    return TensorDict({"index": torch.tensor(list(indices))}, batch_size=[len(indices)])


@pytest.mark.cpu
@pytest.mark.level0
def test_rollout_adapter_create_spreads_framework_workers(monkeypatch):
    option_calls = []
    remote_calls = []
    workers = []

    class _Builder:
        def __init__(self, options):
            self.options = options

        def remote(self, **kwargs):
            worker = SimpleNamespace(worker_id=len(workers), kwargs=kwargs, options=self.options)
            workers.append(worker)
            remote_calls.append(kwargs)
            return worker

    class _WorkerClass:
        @staticmethod
        def options(**kwargs):
            option_calls.append(kwargs)
            return _Builder(kwargs)

    monkeypatch.setattr(entry, "AgentFrameworkWorker", _WorkerClass)
    monkeypatch.setattr(entry, "build_gateway_manager", lambda **_: "gateway")
    node_a, node_b = "a" * 56, "b" * 56
    monkeypatch.setattr(
        entry.ray,
        "nodes",
        lambda: [
            {"NodeID": node_a, "Alive": True, "Resources": {"CPU": 8}},
            {"NodeID": node_b, "Alive": True, "Resources": {"CPU": 8}},
        ],
    )
    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"agent": {"num_workers": 5}}}})

    adapter = entry.AgentFrameworkRolloutAdapter.create(config=config, llm_client=object())

    assert len(adapter.framework_workers) == 5
    assert adapter.framework_worker is adapter.framework_workers[0]
    assert [call["num_cpus"] for call in option_calls] == [0] * 5
    assert [call["scheduling_strategy"].node_id for call in option_calls] == [node_a, node_b, node_a, node_b, node_a]
    assert [call["scheduling_strategy"].soft for call in option_calls] == [False] * 5
    assert [call["gateway_manager"] for call in remote_calls] == ["gateway"] * 5


@pytest.mark.cpu
@pytest.mark.level0
def test_rollout_adapter_rotates_workers_across_small_refills(monkeypatch):
    dispatched = []
    adapter = entry.AgentFrameworkRolloutAdapter()
    adapter.framework_workers = [_recording_worker(index, dispatched) for index in range(3)]

    for start in range(4):
        adapter.generate_sequences(_prompts(start))
    adapter.generate_sequences(_prompts(10, 11, 12))

    assert dispatched == [(0, [0]), (1, [1]), (2, [2]), (0, [3]), (1, [10]), (2, [11]), (0, [12])]

    monkeypatch.setattr(entry.ray, "get", lambda object_refs: object_refs)
    assert adapter.generate_sequences_and_wait(_prompts(20, 21)) is None
    assert dispatched[-2:] == [(1, [20]), (2, [21])]
    assert adapter._dispatch(_prompts()) == []


@pytest.mark.cpu
@pytest.mark.level0
def test_rollout_adapter_splits_uneven_batches_across_all_workers():
    dispatched = []
    adapter = entry.AgentFrameworkRolloutAdapter()
    adapter.framework_workers = [_recording_worker(index, dispatched) for index in range(8)]

    adapter.generate_sequences(_prompts(*range(10)))

    assert dispatched == [
        (0, [0, 1]),
        (1, [2, 3]),
        (2, [4]),
        (3, [5]),
        (4, [6]),
        (5, [7]),
        (6, [8]),
        (7, [9]),
    ]


def _runner_caps_config(**caps: int):
    runners = {name: {"runner_fqn": "pkg.runner", "max_concurrent_sessions": cap} for name, cap in caps.items()}
    agent_framework = {"agent_runners": runners}
    return OmegaConf.create({"actor_rollout_ref": {"rollout": {"custom": {"agent_framework": agent_framework}}}})


def _worker_caps(worker_configs, name: str) -> list[int]:
    return [
        cfg.actor_rollout_ref.rollout.custom.agent_framework.agent_runners[name].max_concurrent_sessions
        for cfg in worker_configs
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_split_session_caps_preserves_global_cap_per_runner():
    config = _runner_caps_config(swe=10, unlimited=0)

    worker_configs = entry.split_session_caps(config, 4)

    assert _worker_caps(worker_configs, "swe") == [3, 3, 2, 2]
    assert _worker_caps(worker_configs, "unlimited") == [0, 0, 0, 0]
    assert config.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.swe.max_concurrent_sessions == 10


@pytest.mark.cpu
@pytest.mark.level0
def test_split_session_caps_uses_fewer_workers_than_the_smallest_cap():
    worker_configs = entry.split_session_caps(_runner_caps_config(swe=2, other=9), 4)

    assert _worker_caps(worker_configs, "swe") == [1, 1]
    assert _worker_caps(worker_configs, "other") == [5, 4]


@pytest.mark.cpu
@pytest.mark.level0
def test_split_session_caps_without_caps_reuses_config():
    config = _runner_caps_config(swe=0)

    worker_configs = entry.split_session_caps(config, 3)

    assert len(worker_configs) == 3
    assert all(worker_config is config for worker_config in worker_configs)


@pytest.mark.cpu
@pytest.mark.level0
def test_rollout_adapter_rejects_nonpositive_worker_count(monkeypatch):
    monkeypatch.setattr(entry, "build_gateway_manager", lambda **_: "gateway")
    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"agent": {"num_workers": 0}}}})

    with pytest.raises(ValueError, match="num_workers must be positive"):
        entry.AgentFrameworkRolloutAdapter.create(config=config, llm_client=object())
