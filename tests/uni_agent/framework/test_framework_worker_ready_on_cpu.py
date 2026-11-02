"""Tests for the framework-actor readiness handshake in `uni_agent.framework.entry`.

The bug these pin down: `AgentFrameworkWorker.remote(...)` returns a handle even
when the actor's `__init__` raised, the adapter never probed it, and
`generate_sequences` is fire-and-forget — so a broken framework config surfaced
as the trainer polling TransferQueue forever instead of failing at start-up.
"""

from __future__ import annotations

import os

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import pytest
import ray

from uni_agent.framework import entry


class _FakeRemoteMethod:
    """Stand-in for a Ray actor method handle; records calls and returns sentinels."""

    def __init__(self, name):
        self.name = name
        self.calls = 0

    def remote(self):
        self.calls += 1
        return f"{self.name}-ref"


class _FakeWorker:
    def __init__(self):
        self.ready = _FakeRemoteMethod("ready")


@pytest.fixture
def fake_ray(monkeypatch):
    """Patch `ray.get` so the unit tests never need a cluster."""
    recorded: list[tuple[object, float | None]] = []

    def fake_get(ref, timeout=None):
        recorded.append((ref, timeout))
        return "ready-ok"

    monkeypatch.setattr(entry.ray, "get", fake_get)
    return recorded


@pytest.mark.cpu
@pytest.mark.level0
def test_ready_probe_calls_ready_with_timeout(fake_ray):
    worker = _FakeWorker()
    entry._wait_until_framework_ready(worker)

    assert worker.ready.calls == 1
    assert fake_ray == [("ready-ref", entry._FRAMEWORK_READY_TIMEOUT_SECONDS)]


@pytest.mark.cpu
@pytest.mark.level0
def test_ready_probe_honours_explicit_timeout(fake_ray):
    entry._wait_until_framework_ready(_FakeWorker(), timeout=1.5)
    assert fake_ray == [("ready-ref", 1.5)]


@pytest.mark.cpu
@pytest.mark.level0
def test_ready_probe_does_not_swallow_actor_death(monkeypatch):
    """An actor that died in __init__ must surface, not be swallowed."""
    worker = _FakeWorker()

    def boom(ref, timeout=None):
        raise ray.exceptions.RayActorError(
            actor_id="0" * 16, error_msg="actor died in __init__", actor_init_failed=True
        )

    monkeypatch.setattr(entry.ray, "get", boom)

    with pytest.raises(ray.exceptions.RayActorError, match="actor died in __init__"):
        entry._wait_until_framework_ready(worker)
    assert worker.ready.calls == 1


@pytest.mark.cpu
@pytest.mark.level0
def test_ready_probe_turns_stuck_construction_into_runtime_error(monkeypatch):
    """A hung __init__ must not hang the caller: it becomes a RuntimeError."""

    def timeout(ref, timeout=None):
        raise ray.exceptions.GetTimeoutError("timed out")

    monkeypatch.setattr(entry.ray, "get", timeout)

    with pytest.raises(RuntimeError, match="did not finish initializing"):
        entry._wait_until_framework_ready(_FakeWorker(), timeout=2.0)


@pytest.mark.cpu
@pytest.mark.level0
def test_ready_probe_preserves_underlying_cause(monkeypatch):
    """The RuntimeError must chain to the Ray timeout so logs show both."""

    def timeout(ref, timeout=None):
        raise ray.exceptions.GetTimeoutError("timed out")

    monkeypatch.setattr(entry.ray, "get", timeout)

    with pytest.raises(RuntimeError) as excinfo:
        entry._wait_until_framework_ready(_FakeWorker(), timeout=2.0)
    assert isinstance(excinfo.value.__cause__, ray.exceptions.GetTimeoutError)


@pytest.mark.cpu
@pytest.mark.level0
def test_framework_worker_exposes_ready():
    """The handshake is useless if the actor has no method to answer it."""
    assert "ready" in dir(entry.AgentFrameworkWorker)


@pytest.fixture(scope="module")
def ray_runtime():
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    yield
    ray.shutdown()


@pytest.mark.cpu
@pytest.mark.level1
def test_ready_probe_succeeds_against_a_live_actor(ray_runtime):
    """End-to-end: a real actor answering `ready` proves construction finished."""

    @ray.remote
    class Healthy:
        def __init__(self):
            self.value = 42

        def ready(self):
            return None

    entry._wait_until_framework_ready(Healthy.remote(), timeout=60)


@pytest.mark.cpu
@pytest.mark.level1
def test_ready_probe_raises_for_an_actor_that_died_in_init(ray_runtime):
    """End-to-end: the failure the fix exists for — and it must carry the cause."""

    @ray.remote
    class Broken:
        def __init__(self):
            raise ValueError("bad framework config")

        def ready(self):
            return None

    with pytest.raises(ray.exceptions.RayActorError, match="bad framework config"):
        entry._wait_until_framework_ready(Broken.remote(), timeout=60)
