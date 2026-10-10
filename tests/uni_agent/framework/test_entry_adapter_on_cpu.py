"""CPU coverage for worker efficiency reporting."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from uni_agent.framework import entry as entry_module

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def _build_framework():
    from uni_agent.framework.framework import GatewayAgentFramework, _RunnerConfig

    return GatewayAgentFramework(None, runner_registry={"runner": _RunnerConfig("unused", {}, "ray_task", 1)})


def test_real_ray_snapshot_while_default_generation_slots_are_saturated(tmp_path):
    import ray

    default_slots = 1000  # Ray's default capacity for async actors.

    class BusyWorker(entry_module.AgentFrameworkWorker.__ray_metadata__.modified_class):
        def __init__(self):
            self.framework = _build_framework()
            self.framework._run_agent_episode = self._hold_episode
            self._starts = 0

        async def _hold_episode(self, **_kwargs):
            await asyncio.Event().wait()

        async def generate_sequences(self, marker):
            self._starts += 1
            if self._starts == default_slots:
                Path(marker).touch()
            await self.framework._run_agent_episode_with_concurrency_limit(
                sample_fields={}, sample_index=0, session_index=0, global_steps=0, sampling_params={}
            )

    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(address="local", num_cpus=1, num_gpus=0, include_dashboard=False)
    worker = None
    try:
        pythonpath = str(Path(__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
        # Retain production actor groups and Ray's default generation capacity.
        actor_options = {"num_cpus": 0, **entry_module.AgentFrameworkWorker._default_options}
        worker = (
            ray.remote(**actor_options)(BusyWorker)
            .options(runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})
            .remote()
        )
        marker = tmp_path / "all-generation-slots-started"
        for _ in range(default_slots):
            worker.generate_sequences.remote(str(marker))
        deadline = time.monotonic() + 45
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists(), "generation calls did not occupy the default actor slots"
        metrics = ray.get(worker.get_efficiency_metrics.remote(), timeout=10)
        assert metrics["sessions/admitted"] == metrics["sessions/in_flight"] == 1
        assert metrics["sessions/completed"] == 0
    finally:
        if worker is not None:
            ray.kill(worker)
        if owns_ray:
            ray.shutdown()


@pytest.mark.asyncio
async def test_adapter_combines_session_and_live_model_metrics():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    adapter = entry_module.AgentFrameworkRolloutAdapter()
    adapter.framework_worker = SimpleNamespace(
        get_efficiency_metrics=SimpleNamespace(
            remote=AsyncMock(return_value={"sessions/admitted": 3.0, "admission/wait_s": 12.0})
        )
    )
    adapter.gateway_manager = SimpleNamespace(
        get_efficiency_metrics=AsyncMock(return_value={"model/output_tokens": 100.0, "model/requests_in_flight": 2.0})
    )
    assert await adapter.get_efficiency_metrics() == {
        "sessions/admitted": 3.0,
        "admission/wait_s": 12.0,
        "model/output_tokens": 100.0,
        "model/requests_in_flight": 2.0,
    }


@pytest.mark.asyncio
async def test_session_metrics_exclude_cancelled_capacity_waiters(monkeypatch):
    framework = _build_framework()
    started = asyncio.Event()

    async def episode(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(framework, "_run_agent_episode", episode)
    kwargs = dict(sample_fields={}, sample_index=0, session_index=0, global_steps=0, sampling_params={})
    running = asyncio.create_task(framework._run_agent_episode_with_concurrency_limit(**kwargs))
    waiting = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        waiting = asyncio.create_task(framework._run_agent_episode_with_concurrency_limit(**kwargs))
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        metrics = framework.get_efficiency_metrics()
        assert metrics["sessions/admitted"] == metrics["sessions/in_flight"] == 1
        assert metrics["sessions/completed"] == 0
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        metrics = framework.get_efficiency_metrics()
        assert metrics["sessions/admitted"] == metrics["sessions/completed"] == 1
        assert metrics["sessions/in_flight"] == 0
    finally:
        for task in (running, waiting):
            if task is not None:
                task.cancel()
        await asyncio.gather(*(task for task in (running, waiting) if task is not None), return_exceptions=True)
