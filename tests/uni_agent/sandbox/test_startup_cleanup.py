import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from uni_agent.sandbox.base import Sandbox

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.mark.asyncio
async def test_failed_start_preserves_original_error_when_cleanup_fails():
    failure = RuntimeError("startup failed")
    sandbox = SimpleNamespace(
        _run_start=AsyncMock(side_effect=failure),
        stop=AsyncMock(side_effect=RuntimeError("cleanup failed")),
    )
    with pytest.raises(RuntimeError) as caught:
        await Sandbox.__aenter__(sandbox, retry=1)
    assert caught.value is failure
    sandbox.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_start_waits_for_cleanup_despite_repeated_cancel():
    started = asyncio.Event()
    stopping = asyncio.Event()
    release_stop = asyncio.Event()
    stopped = asyncio.Event()

    async def start():
        started.set()
        await asyncio.Event().wait()

    async def stop():
        stopping.set()
        await release_stop.wait()
        stopped.set()

    sandbox = SimpleNamespace(_run_start=AsyncMock(side_effect=start), stop=stop)
    task = asyncio.create_task(Sandbox.__aenter__(sandbox, retry=3))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        await asyncio.wait_for(stopping.wait(), timeout=5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release_stop.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
    assert stopped.is_set()
    sandbox._run_start.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_start_cleanup_times_out(monkeypatch):
    monkeypatch.setenv("SANDBOX_STOP_TIMEOUT", "0.01")
    failure = RuntimeError("startup failed")
    stop_cancelled = asyncio.Event()

    async def stop():
        try:
            await asyncio.Event().wait()
        finally:
            stop_cancelled.set()

    sandbox = SimpleNamespace(_run_start=AsyncMock(side_effect=failure), stop=stop)
    with pytest.raises(RuntimeError) as caught:
        await asyncio.wait_for(Sandbox.__aenter__(sandbox, retry=1), timeout=5)
    assert caught.value is failure
    assert stop_cancelled.is_set()
