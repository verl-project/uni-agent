"""Execution backends for command-driven external MAS rollouts (Linux only)."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class LocalProcessResult:
    returncode: int
    stdout: str
    stderr: str


class LocalProcessError(RuntimeError):
    def __init__(self, returncode: int, stdout: str, stderr: str):
        super().__init__(
            f"External MAS process exited with exit code {returncode}: {stderr.strip()}"
        )
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class LocalProcessTimeout(TimeoutError):
    pass


_TERMINATE_GRACE_SECONDS = 2.0
_COMMUNICATE_DRAIN_TIMEOUT_SECONDS = 2.0


def _killpg(pgid: int, signum: int) -> None:
    os.killpg(pgid, signum)


def _process_group_alive(pgid: int) -> bool:
    try:
        _killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


async def _terminate_process_group(
    pgid: int, grace_seconds: float = _TERMINATE_GRACE_SECONDS
) -> None:
    """Terminate every process in the group, even after its leader exited."""
    if not _process_group_alive(pgid):
        return
    try:
        _killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while _process_group_alive(pgid) and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    if not _process_group_alive(pgid):
        return
    try:
        _killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return


async def _wait_for_exit(
    process: asyncio.subprocess.Process, timeout: float
) -> None:
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        pass


def _close_subprocess_pipes(process: asyncio.subprocess.Process) -> None:
    transport = getattr(process, "_transport", None)
    if transport is not None:
        transport.close()


async def _drain_communicate(
    process: asyncio.subprocess.Process,
    communicate_task: asyncio.Task[tuple[bytes, bytes]],
    timeout: float | None = None,
) -> tuple[bytes, bytes]:
    """Drain output without blocking forever on inherited pipe holders."""
    if timeout is None:
        timeout = _COMMUNICATE_DRAIN_TIMEOUT_SECONDS
    try:
        return await asyncio.wait_for(communicate_task, timeout=timeout)
    except asyncio.TimeoutError:
        communicate_task.cancel()
        with contextlib.suppress(
            asyncio.CancelledError, BrokenPipeError, ConnectionResetError
        ):
            await communicate_task
        _close_subprocess_pipes(process)
        return b"", b""
    except (BrokenPipeError, ConnectionResetError):
        _close_subprocess_pipes(process)
        return b"", b""


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    """Terminate the child and descendants holding its stdio pipes."""
    # The leader may already have exited while grandchildren keep the pipes
    # open, so always clean up the Linux process group.
    await _terminate_process_group(process.pid)
    if process.returncode is None:
        await _wait_for_exit(process, timeout=_TERMINATE_GRACE_SECONDS)


async def run_local_process(
    command: str | Sequence[str],
    *,
    work_dir: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
    shell: bool = False,
) -> LocalProcessResult:
    """Run one MAS command and terminate its process group on timeout/cancellation."""
    process_env = os.environ.copy()
    if env:
        process_env.update({str(key): str(value) for key, value in env.items()})

    process_kwargs = {
        "cwd": work_dir,
        "env": process_env,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "start_new_session": True,
    }

    if shell:
        if not isinstance(command, str):
            raise TypeError("shell commands must be strings")
        process = await asyncio.create_subprocess_shell(command, **process_kwargs)
    else:
        if isinstance(command, str):
            raise TypeError("non-shell commands must be an argv sequence")
        process = await asyncio.create_subprocess_exec(
            *[str(part) for part in command], **process_kwargs
        )

    communicate_task = asyncio.create_task(process.communicate())
    try:
        if timeout_seconds is None:
            stdout_bytes, stderr_bytes = await communicate_task
        else:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                asyncio.shield(communicate_task),
                timeout=float(timeout_seconds),
            )
    except asyncio.TimeoutError as exc:
        await _terminate_process(process)
        await _drain_communicate(process, communicate_task)
        raise LocalProcessTimeout(
            f"External MAS process timed out after {timeout_seconds} seconds"
        ) from exc
    except asyncio.CancelledError:
        await _terminate_process(process)
        await _drain_communicate(process, communicate_task)
        raise

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    result = LocalProcessResult(
        returncode=int(process.returncode or 0), stdout=stdout, stderr=stderr
    )
    if result.returncode != 0:
        raise LocalProcessError(result.returncode, result.stdout, result.stderr)
    return result
