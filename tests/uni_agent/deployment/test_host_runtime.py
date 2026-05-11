"""Tests for HostRuntime: concurrency, timeout recovery, and session rebuild.

Requires swerex to be installed (HostRuntime depends on its abstract types).
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

pytest.importorskip("swerex")

from swerex.exceptions import CommandTimeoutError  # noqa: E402
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
    results = await asyncio.gather(
        *(runtime.run_in_session(BashAction(command=c, timeout=10)) for c in commands)
    )
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
    """BashInterruptAction must be deliverable while another command holds the
    IO lock, and the next command must still work."""

    async def long_running() -> None:
        with pytest.raises(CommandTimeoutError):
            await runtime.run_in_session(BashAction(command="sleep 30", timeout=2))

    task = asyncio.create_task(long_running())
    await asyncio.sleep(0.3)
    # Interrupt while `sleep` is blocking inside the lock holder.
    await runtime.run_in_session(BashInterruptAction(timeout=5))
    await task

    r = await runtime.run_in_session(BashAction(command="echo after_interrupt", timeout=10))
    assert r.output.strip() == "after_interrupt"


@pytest.mark.asyncio
async def test_session_rebuilds_after_bash_dies(runtime: HostRuntime) -> None:
    """If the bash process exits, the next call should transparently rebuild."""
    proc = runtime._process
    assert proc is not None
    proc.kill()
    await proc.wait()

    r = await runtime.run_in_session(BashAction(command="echo alive_again", timeout=10))
    assert r.exit_code == 0
    assert r.output.strip() == "alive_again"
    assert runtime._process is not None and runtime._process.returncode is None
