from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from .base import ExecResult, Sandbox, _to_str
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import SandboxConfig

logger = logging.getLogger(__name__)

# Retain deferred spawn cleanups even after their cancelled caller has returned.
_cleanup_tasks: set[asyncio.Future[None]] = set()


@register_sandbox("local")
class LocalSandbox(Sandbox):
    """Runs commands on the host via ``asyncio`` subprocesses (no container).

    File operations use the host filesystem directly.
    """

    @classmethod
    def from_config(cls, config: SandboxConfig) -> LocalSandbox:
        if config.image is not None:
            raise ValueError("LocalSandbox does not accept an image; use provider='docker' to run one locally")
        if config.image_mounts:
            raise NotImplementedError("LocalSandbox does not support image mounts")
        if config.executable_paths:
            raise NotImplementedError("LocalSandbox does not support executable path overrides")
        return cls()

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def read_file(self, path: str) -> bytes:
        """Read directly from the host filesystem without base64 transport."""
        return await asyncio.to_thread(Path(path).read_bytes)

    async def write_file(self, path: str, content: bytes | str) -> None:
        """Write directly to the host filesystem, creating parent directories."""
        data = content.encode("utf-8") if isinstance(content, str) else content
        target = Path(path)

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        await asyncio.to_thread(_write)

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        # A new session gives the command its own process group, so a timeout or
        # cancellation also kills its background children.
        spawn = asyncio.ensure_future(
            asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env={**os.environ, **env} if env else None,
                start_new_session=True,
            )
        )
        try:
            # A cancelled spawn would kill only the child and then block until
            # every pipe closes, which a background grandchild can delay.
            proc = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            spawn.add_done_callback(_kill_spawned)
            raise
        # No proc.wait() after the kill: it also waits for the pipes, which a
        # process that left the group may hold. The event loop reaps the child.
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            await _await_cleanup(proc)
            return ExecResult(exit_code=-1, stdout="", stderr=f"local exec timed out after {timeout}s")
        except BaseException:
            await _await_cleanup(proc)
            raise
        return ExecResult(exit_code=proc.returncode or 0, stdout=_to_str(out), stderr=_to_str(err))


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _kill_command(proc: asyncio.subprocess.Process) -> None:
    """Kill the original group and groups found in its still-active process tree.

    Snapshot before killing the leader: detached tools retain PPID ancestry until
    their parents exit. This is best-effort cleanup, not security containment:
    already-reparented daemons and children created after the snapshot can escape.
    Extra groups need a recorded member still present; unobserved forks after
    recorded members exit can be missed, and PID/PGID reuse is not fully fenced.
    Called in the default executor: the one-second ps timeout does not block the
    event loop. Successful commands do not pay this cost. No pipes are awaited.
    """
    groups: dict[int, set[int]] = {proc.pid: {proc.pid}}
    if proc.returncode is None:
        try:
            snapshot = subprocess.run(
                ["ps", "-A", "-o", "pid=,ppid=,pgid="],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=True,
                timeout=1,
            )
            children: dict[int, list[tuple[int, int]]] = {}
            for line in snapshot.stdout.splitlines():
                child, parent, group = map(int, line.split())
                children.setdefault(parent, []).append((child, group))
            pending, seen = [proc.pid], {proc.pid}
            while pending:
                for child, group in children.get(pending.pop(), []):
                    if child not in seen:
                        seen.add(child)
                        pending.append(child)
                        groups.setdefault(group, set()).add(child)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            logger.warning("local exec descendant snapshot failed; killing original group only: %s", exc)
    own_group = os.getpgrp()
    for group in [proc.pid, *sorted(groups.keys() - {proc.pid})]:
        if group > 1 and group != own_group:
            if group != proc.pid and not any(_still_in_group(pid, group) for pid in groups[group]):
                continue
            try:
                _kill_group(group)
            except OSError as exc:
                logger.warning("local exec could not terminate process group %s: %s", group, exc)


def _still_in_group(pid: int, group: int) -> bool:
    try:
        return os.getpgid(pid) == group
    except (ProcessLookupError, PermissionError):
        return False


def _kill_spawned(spawn: asyncio.Future[asyncio.subprocess.Process]) -> None:
    if not spawn.cancelled() and spawn.exception() is None:
        _start_cleanup(spawn.result())


def _start_cleanup(proc: asyncio.subprocess.Process) -> asyncio.Future[None]:
    # Submit immediately: a deferred coroutine could be cancelled by loop
    # shutdown before ever submitting its worker. asyncio.run drains this
    # executor on shutdown, including cleanups detached from spawn callers.
    task = asyncio.get_running_loop().run_in_executor(None, _kill_command, proc)
    _cleanup_tasks.add(task)
    task.add_done_callback(_cleanup_finished)
    return task


def _cleanup_finished(task: asyncio.Future[None]) -> None:
    _cleanup_tasks.discard(task)
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.warning("local exec deferred cleanup failed: %s", error)


async def _await_cleanup(proc: asyncio.subprocess.Process) -> None:
    # shield prevents a second cancellation from cancelling the wrapper task
    # while its thread still owns live process cleanup. Keep waiting for that
    # same task; do not start duplicate snapshots or return before signalling.
    task = _start_cleanup(proc)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError
