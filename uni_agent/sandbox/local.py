"""Host-local sandbox (no container).

Runs commands as host subprocesses; for tests and host-local runs. ``start`` /
``stop`` are no-ops (the host is always there). Depends only on the standard
library, so it is always importable.
"""

from __future__ import annotations

import asyncio

from .base import ExecResult, Sandbox, _to_str


class LocalSandbox(Sandbox):
    """Runs commands on the host via ``asyncio`` subprocesses (no container).

    File operations use the inherited exec-based floor, which transparently
    round-trips through the host shell.
    """

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        import os

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            env={**os.environ, **env} if env else None,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return ExecResult(exit_code=124, stdout="", stderr=f"local exec timed out after {timeout}s")
        return ExecResult(exit_code=proc.returncode or 0, stdout=_to_str(out), stderr=_to_str(err))
