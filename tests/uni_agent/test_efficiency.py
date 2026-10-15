import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from uni_agent.efficiency import collect_efficiency, measure_efficiency, record_efficiency
from uni_agent.sandbox.base import Sandbox
from uni_agent.tools.base import Tool, Toolbox, ToolError, ToolResult

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_measure_accumulates_and_restores_nested_collectors(monkeypatch):
    ticks = iter([1.0, 3.0, 5.0, 8.0])
    monkeypatch.setattr("uni_agent.efficiency.time.perf_counter", lambda: next(ticks))
    with collect_efficiency() as outer:
        with measure_efficiency("model"):
            pass
        with collect_efficiency() as inner:
            record_efficiency("only_inner")
        with measure_efficiency("model"):
            pass
    record_efficiency("outside")
    assert {k: v for k, v in outer.items() if k.startswith("model/")} == {
        "model/count": 2,
        "model/total_s": 5,
        "model/error_count": 0,
        "model/cancelled_count": 0,
    }
    assert "only_inner" not in outer
    assert "outside" not in outer
    assert inner["only_inner"] == 1
    assert "model/count" not in inner


@pytest.mark.asyncio
async def test_overlapping_episode_collectors_are_isolated():
    async def episode(stage):
        with collect_efficiency() as metrics:
            with measure_efficiency(stage):
                await asyncio.sleep(0)
            return metrics

    one, two = await asyncio.gather(episode("first"), episode("second"))
    assert one["first/count"] == two["second/count"] == 1
    assert "second/count" not in one
    assert "first/count" not in two


@pytest.mark.asyncio
async def test_sandbox_startup_includes_retries_and_cleanup(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("uni_agent.efficiency.time.perf_counter", lambda: now[0])
    calls = []

    async def start():
        calls.append("start")
        now[0] += 3
        if len(calls) == 1:
            raise RuntimeError("retry")

    async def stop():
        calls.append("stop")
        now[0] += 1

    async def sleep(delay):
        now[0] += delay

    monkeypatch.setattr(asyncio, "sleep", sleep)
    sandbox = SimpleNamespace(_run_start=start, stop=stop)
    with collect_efficiency() as metrics:
        assert await Sandbox.__aenter__(sandbox, retry=2) is sandbox
    assert calls == ["start", "stop", "start"]
    assert metrics["sandbox_startup/count"] == 1
    assert metrics["sandbox_startup/total_s"] == 9
    assert metrics["sandbox_startup/error_count"] == 0


class ReturningTool(Tool):
    name = "shell"

    def schema(self):
        return {"type": "function", "function": {"name": "shell", "parameters": {"type": "object"}}}

    async def run(self, args, *, timeout=None):
        return ToolResult(status="timeout")


@pytest.mark.asyncio
async def test_tool_returned_failure_and_thrown_error_are_distinct():
    tool = ReturningTool(object())
    box = Toolbox([tool])
    with collect_efficiency() as metrics:
        assert (await box.call("shell")).status == "timeout"
        tool.run = AsyncMock(side_effect=ToolError("bad command"))
        assert (await box.call("shell")).status == "error"
        assert (await box.call("missing")).status == "format_error"
        failure = RuntimeError("bug")
        tool.run = AsyncMock(side_effect=failure)
        with pytest.raises(RuntimeError) as caught:
            await box.call("shell")
    assert caught.value is failure
    assert metrics["tool/count"] == 4
    assert metrics["tool/error_count"] == 1
    for status in ("timeout", "error", "format_error"):
        assert metrics[f"tool/result_{status}_count"] == 1


@pytest.mark.asyncio
async def test_startup_metrics_preserve_cleanup_through_repeated_cancellation():
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
    with collect_efficiency() as metrics:
        task = asyncio.create_task(Sandbox.__aenter__(sandbox))
        try:
            await asyncio.wait_for(started.wait(), 5)
            task.cancel()
            await asyncio.wait_for(stopping.wait(), 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release_stop.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
    assert stopped.is_set()
    sandbox._run_start.assert_awaited_once()
    assert metrics["sandbox_startup/count"] == metrics["sandbox_startup/cancelled_count"] == 1
    assert metrics["sandbox_startup/error_count"] == 0
