"""Tests for HostRuntime: concurrency, timeout recovery, and session rebuild.

Requires swerex to be installed (HostRuntime depends on its abstract types).
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

pytest.importorskip("swerex")

from swerex.exceptions import CommandTimeoutError, SessionNotInitializedError  # noqa: E402
from swerex.runtime.abstract import (  # noqa: E402
    BashAction,
    BashInterruptAction,
    CreateSessionRequest,
)

from uni_agent.deployment.host.deployment import HostRuntime  # noqa: E402


@pytest_asyncio.fixture
async def runtime():
    rt = HostRuntime(run_id="test")
    await rt.create_session(CreateSessionRequest(startup_source=[], startup_timeout=10))
    try:
        yield rt
    finally:
        await rt.close()


@pytest.mark.asyncio
async def test_sequential_commands_keep_session_state(runtime: HostRuntime) -> None:
    """Persistent session: state from one command must be visible in the next."""
    r1 = await runtime.run_in_session(BashAction(command="export FOO=bar", timeout=10))
    assert r1.exit_code == 0

    r2 = await runtime.run_in_session(BashAction(command="echo $FOO", timeout=10))
    assert r2.exit_code == 0
    assert r2.output.strip() == "bar"


@pytest.mark.asyncio
async def test_concurrent_commands_are_serialized(runtime: HostRuntime) -> None:
    """Concurrent run_in_session calls must not interleave their stdout/stdin."""
    commands = [f"echo line_{i}" for i in range(8)]
    results = await asyncio.gather(*(runtime.run_in_session(BashAction(command=c, timeout=10)) for c in commands))
    outputs = [r.output.strip() for r in results]
    # Each result must match exactly one input, with no cross-contamination.
    assert sorted(outputs) == sorted(f"line_{i}" for i in range(8))
    for r in results:
        assert r.exit_code == 0


@pytest.mark.asyncio
async def test_timeout_does_not_pollute_next_command(runtime: HostRuntime) -> None:
    """After a timeout, the next command must see a clean stdout stream."""
    with pytest.raises(CommandTimeoutError):
        await runtime.run_in_session(BashAction(command="sleep 30", timeout=1))

    # Next command should return exactly its own output, no leftover markers
    # or "sleep" output bleeding in.
    r = await runtime.run_in_session(BashAction(command="echo recovered", timeout=10))
    assert r.exit_code == 0
    assert r.output.strip() == "recovered"


@pytest.mark.asyncio
async def test_interrupt_unblocks_running_command(runtime: HostRuntime) -> None:
    """BashInterruptAction must be deliverable while another command holds
    the IO lock. The interrupted run_in_session should return cleanly with
    a non-zero exit code (SIGINT → 130), not hang or pollute the next call."""
    result: dict[str, object] = {}

    async def long_running() -> None:
        # Use a generous timeout: we expect the interrupt to finish this
        # well before the deadline.
        result["obs"] = await runtime.run_in_session(BashAction(command="sleep 30", timeout=15))

    task = asyncio.create_task(long_running())
    await asyncio.sleep(0.3)
    await runtime.run_in_session(BashInterruptAction(timeout=5))
    await asyncio.wait_for(task, timeout=5)

    obs = result["obs"]
    assert obs.exit_code == 130, f"expected SIGINT exit code, got {obs.exit_code}"

    r = await runtime.run_in_session(BashAction(command="echo after_interrupt", timeout=10))
    assert r.exit_code == 0
    assert r.output.strip() == "after_interrupt"


@pytest.mark.asyncio
async def test_interrupt_kills_entire_pipeline(runtime: HostRuntime) -> None:
    """A pipeline (e.g. `sleep 30 | cat | cat`) spawns multiple processes
    in one process group. Interrupt must signal the whole group, otherwise
    bash stays blocked waiting for the surviving stages to finish."""
    result: dict[str, object] = {}

    async def long_pipeline() -> None:
        result["obs"] = await runtime.run_in_session(BashAction(command="sleep 30 | cat | cat", timeout=15))

    task = asyncio.create_task(long_pipeline())
    await asyncio.sleep(0.3)
    await runtime.run_in_session(BashInterruptAction(timeout=5))
    await asyncio.wait_for(task, timeout=5)

    obs = result["obs"]
    assert obs.exit_code != 0, "pipeline should not exit cleanly after SIGINT"

    r = await runtime.run_in_session(BashAction(command="echo still_alive", timeout=10))
    assert r.exit_code == 0
    assert r.output.strip() == "still_alive"


@pytest.mark.asyncio
async def test_dead_session_raises_instead_of_silent_rebuild(runtime: HostRuntime) -> None:
    """If the bash process exits, the next call must surface the failure so
    the upper layer can fail the episode, rather than silently respawning a
    fresh shell with lost state."""
    proc = runtime._process
    assert proc is not None
    proc.kill()
    await proc.wait()

    with pytest.raises(SessionNotInitializedError):
        await runtime.run_in_session(BashAction(command="echo never_runs", timeout=10))

    alive = await runtime.is_alive()
    assert not alive.is_alive
