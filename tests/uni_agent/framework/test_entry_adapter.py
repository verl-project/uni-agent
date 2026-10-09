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
    monkeypatch.setattr(
        entry.ray,
        "nodes",
        lambda: [
            {"NodeID": "node-a", "Alive": True, "Resources": {"CPU": 8}},
            {"NodeID": "node-b", "Alive": True, "Resources": {"CPU": 8}},
        ],
    )
    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"agent": {"num_workers": 5}}}})

    adapter = entry.AgentFrameworkRolloutAdapter.create(config=config, llm_client=object())

    assert len(adapter.framework_workers) == 5
    assert adapter.framework_worker is adapter.framework_workers[0]
    assert [call["num_cpus"] for call in option_calls] == [0] * 5
    assert [call["scheduling_strategy"].node_id for call in option_calls] == [
        "node-a",
        "node-b",
        "node-a",
        "node-b",
        "node-a",
    ]
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
def test_rollout_adapter_rejects_nonpositive_worker_count(monkeypatch):
    monkeypatch.setattr(entry, "build_gateway_manager", lambda **_: "gateway")
    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"agent": {"num_workers": 0}}}})

    with pytest.raises(ValueError, match="num_workers must be positive"):
        entry.AgentFrameworkRolloutAdapter.create(config=config, llm_client=object())
