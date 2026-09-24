from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from .base import ExecResult, Sandbox, _to_str
from .registry import register_sandbox


@register_sandbox("local")
class LocalSandbox(Sandbox):
    """Runs commands on the host via ``asyncio`` subprocesses (no container).

    File operations use the host filesystem directly. Constructed with no args,
    so it uses the base :meth:`Sandbox.from_config` (which ignores the config
    fields).
    """

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
            _kill_group(proc.pid)
            return ExecResult(exit_code=-1, stdout="", stderr=f"local exec timed out after {timeout}s")
        except BaseException:
            _kill_group(proc.pid)
            raise
        return ExecResult(exit_code=proc.returncode or 0, stdout=_to_str(out), stderr=_to_str(err))


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _kill_spawned(spawn: asyncio.Future[asyncio.subprocess.Process]) -> None:
    if not spawn.cancelled() and spawn.exception() is None:
        _kill_group(spawn.result().pid)
