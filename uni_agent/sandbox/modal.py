"""Modal sandbox: one class that creates a Modal sandbox and drives it.

Merges lifecycle (``start`` creates a ``sleep infinity`` sandbox, ``stop``
terminates it) with the data plane (``exec`` over ``modal.Sandbox.exec``, native
binary-safe ``upload`` / ``download``). ``modal`` is imported lazily inside
:meth:`start`, so this module imports fine where ``modal`` isn't installed; an
already-created handle can be wrapped via :meth:`from_handle` (warm pools, or a
sandbox booted by another service) without owning its lifecycle.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from .base import ExecResult, Sandbox, _to_str

if TYPE_CHECKING:
    import modal


class ModalSandbox(Sandbox):
    """Creates a Modal sandbox (``sleep infinity``) and drives it via exec."""

    def __init__(
        self,
        *,
        image: str = "python:3.12-slim",
        app_name: str = "agent-sandbox",
        runtime_timeout: float = 3600.0,
        modal_sandbox_kwargs: dict | None = None,
    ):
        self.image = image
        self.app_name = app_name
        self.runtime_timeout = runtime_timeout
        self.modal_sandbox_kwargs = dict(modal_sandbox_kwargs or {})
        self._app = None
        self._sandbox: modal.Sandbox | None = None
        self._owns_sandbox = True

    @classmethod
    def from_handle(cls, sandbox: modal.Sandbox) -> ModalSandbox:
        """Wrap an already-created Modal sandbox without owning its lifecycle.

        ``start`` becomes a no-op and ``stop`` will not terminate the sandbox --
        use this to attach to a warm-pool / externally-booted sandbox and drive
        it (exec + files + sessions) while leaving teardown to its owner.
        """
        inst = cls()
        inst._sandbox = sandbox
        inst._owns_sandbox = False
        return inst

    # ----- control plane -----
    async def start(self) -> None:
        if self._sandbox is not None:
            return  # already attached (from_handle) or started
        import modal

        self._app = await modal.App.lookup.aio(self.app_name, create_if_missing=True)
        image = modal.Image.from_registry(self.image)
        self._sandbox = await modal.Sandbox.create.aio(
            "sleep",
            "infinity",
            image=image,
            app=self._app,
            timeout=int(self.runtime_timeout),
            **self.modal_sandbox_kwargs,
        )

    async def stop(self) -> None:
        if self._sandbox is not None and self._owns_sandbox:
            try:
                if await self._sandbox.poll.aio() is None:
                    await self._sandbox.terminate.aio()
            finally:
                self._sandbox = None
        self._app = None

    def _require_sandbox(self) -> modal.Sandbox:
        if self._sandbox is None:
            raise RuntimeError("ModalSandbox not started; call start() or use from_handle()")
        return self._sandbox

    # ----- data plane -----
    async def exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        sandbox = self._require_sandbox()
        # Modal exec has no portable env kwarg across versions; inject via the
        # shell-free ``env`` binary so quoting stays the caller's concern.
        if env:
            argv = ["env", *(f"{key}={value}" for key, value in env.items()), *argv]
        proc = await sandbox.exec.aio(
            *argv,
            workdir=workdir,
            timeout=int(timeout) if timeout else None,
        )

        async def _read(stream) -> str:
            try:
                return _to_str(await stream.read.aio())
            except Exception:
                return ""

        out_task = asyncio.create_task(_read(proc.stdout))
        err_task = asyncio.create_task(_read(proc.stderr))
        exit_code = await proc.wait.aio()
        return ExecResult(exit_code=int(exit_code or 0), stdout=await out_task, stderr=await err_task)

    async def upload(self, local_path: Path | str, remote_path: str) -> None:
        # Native Modal filesystem copy: streams the file, creates the remote
        # parent dirs, and overwrites. ``remote_path`` must be absolute.
        # (Directories go through the base ``upload_dir`` tar path.)
        src = Path(local_path)
        if src.is_dir():
            raise IsADirectoryError(f"upload expects a file; use upload_dir() for {src}")
        await self._require_sandbox().filesystem.copy_from_local.aio(str(src), remote_path)

    async def download(self, remote_path: str, local_path: Path | str) -> None:
        # Native Modal filesystem copy: streams to a temp file then atomically
        # renames, and creates local parent dirs. ``remote_path`` must be absolute.
        await self._require_sandbox().filesystem.copy_to_local.aio(remote_path, str(local_path))

    async def expose_port(self, port: int) -> str:
        # Modal requires ports to be declared via ``encrypted_ports`` at sandbox
        # creation (pass through ``modal_sandbox_kwargs``); a running
        # ``sleep infinity`` sandbox cannot open one on demand. Once declared,
        # implement this via ``self._sandbox.tunnels()``.
        raise NotImplementedError(
            "ModalSandbox.expose_port requires encrypted_ports at sandbox creation time"
        )
