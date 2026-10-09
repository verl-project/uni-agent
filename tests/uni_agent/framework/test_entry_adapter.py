from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from uni_agent.framework import entry


class _FakeRemoteMethod:
    def __init__(self, worker_id: int, calls: list[tuple[int, object]]) -> None:
        self.worker_id = worker_id
        self.calls = calls

    def remote(self, chunk):
        result = (self.worker_id, chunk)
        self.calls.append(result)
        return result


class _FakeWorker:
    def __init__(self, worker_id: int, calls: list[tuple[int, object]]) -> None:
        self.generate_sequences = _FakeRemoteMethod(worker_id, calls)


class _FakePrompts:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def chunk(self, count: int):
        return [f"chunk-{index}" for index in range(count)]


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
def test_rollout_adapter_dispatch_chunks_across_workers(monkeypatch):
    calls = []
    adapter = entry.AgentFrameworkRolloutAdapter()
    adapter.framework_workers = [_FakeWorker(index, calls) for index in range(3)]

    refs = adapter._dispatch(_FakePrompts(5))

    assert refs == [(0, "chunk-0"), (1, "chunk-1"), (2, "chunk-2")]
    assert calls == refs
    monkeypatch.setattr(entry.ray, "get", lambda object_refs: ("waited", object_refs))
    assert adapter.generate_sequences_and_wait(_FakePrompts(2)) is None
    assert calls[-2:] == [(0, "chunk-0"), (1, "chunk-1")]


@pytest.mark.cpu
@pytest.mark.level0
def test_rollout_adapter_rejects_nonpositive_worker_count(monkeypatch):
    monkeypatch.setattr(entry, "build_gateway_manager", lambda **_: "gateway")
    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"agent": {"num_workers": 0}}}})

    with pytest.raises(ValueError, match="num_workers must be positive"):
        entry.AgentFrameworkRolloutAdapter.create(config=config, llm_client=object())
