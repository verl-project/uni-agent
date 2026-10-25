"""CPU tests for the external MAS local process backend."""

from __future__ import annotations

import asyncio
import signal

import pytest

from examples.multi_policy.external_mas import process_backend


class _FakeTransport:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _NeverEndingProcess:
    pid = 43210
    returncode = 0

    def __init__(self) -> None:
        self._transport = _FakeTransport()

    async def communicate(self):
        await asyncio.Event().wait()

    async def wait(self):
        return self.returncode


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_timeout_does_not_wait_for_inherited_pipe_holders(monkeypatch):
    process = _NeverEndingProcess()

    async def fake_create(*args, **kwargs):
        return process

    async def fake_terminate(_process):
        return None

    monkeypatch.setattr(
        process_backend.asyncio, "create_subprocess_exec", fake_create
    )
    monkeypatch.setattr(process_backend, "_terminate_process", fake_terminate)
    monkeypatch.setattr(
        process_backend,
        "_COMMUNICATE_DRAIN_TIMEOUT_SECONDS",
        0.05,
        raising=False,
    )

    with pytest.raises(process_backend.LocalProcessTimeout):
        await asyncio.wait_for(
            process_backend.run_local_process(["fake-mas"], timeout_seconds=0.01),
            timeout=5.0,
        )

    assert process._transport.closed


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_terminate_process_group_kills_after_leader_exits(monkeypatch):
    calls = []
    alive = True
    sigkill = getattr(signal, "SIGKILL", 9)

    monkeypatch.setattr(
        process_backend.signal, "SIGKILL", sigkill, raising=False
    )

    def fake_killpg(pgid, signum):
        nonlocal alive
        calls.append(signum)
        if signum == 0:
            if not alive:
                raise ProcessLookupError
            return
        if signum == sigkill:
            alive = False

    monkeypatch.setattr(process_backend, "_killpg", fake_killpg)

    await process_backend._terminate_process_group(
        12345, grace_seconds=0.01
    )

    assert signal.SIGTERM in calls
    assert sigkill in calls


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_terminate_process_cleans_group_when_leader_already_exited(monkeypatch):
    calls = []

    async def fake_terminate_group(pgid, grace_seconds=2.0):
        calls.append(pgid)

    monkeypatch.setattr(
        process_backend, "_terminate_process_group", fake_terminate_group
    )

    class FakeProcess:
        pid = 12345
        returncode = 0

    await process_backend._terminate_process(FakeProcess())

    assert calls == [12345]

    assert calls == [12345]
