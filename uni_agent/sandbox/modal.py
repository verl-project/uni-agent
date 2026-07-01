from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from .base import ExecResult, Sandbox, _to_str
from .registry import register_sandbox

if TYPE_CHECKING:
    import modal

    from .base import SandboxConfig


@register_sandbox("modal")
class ModalSandbox(Sandbox):
    """Creates a Modal sandbox (``sleep infinity``) and drives it via exec."""

    def __init__(
        self,
        *,
        image: str = "python:3.12-slim",
        app_name: str = "agent-sandbox",
        runtime_timeout: float = 3600.0,
        **modal_sandbox_kwargs,
    ):
        self.image = image
        self.app_name = app_name
        self.runtime_timeout = runtime_timeout
        self.modal_sandbox_kwargs = dict(modal_sandbox_kwargs)
        self._app = None
        self._sandbox: modal.Sandbox | None = None

    @classmethod
    def from_config(cls, config: SandboxConfig) -> ModalSandbox:
        return cls(image=config.image, runtime_timeout=config.runtime_timeout, **config.sandbox_kwargs)

    # ----- control plane -----
    async def start(self) -> None:
        if self._sandbox is not None:
            return  # already started
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
        if self._sandbox is not None:
            try:
                if await self._sandbox.poll.aio() is None:
                    await self._sandbox.terminate.aio()
            finally:
                self._sandbox = None
        self._app = None

    def _require_sandbox(self) -> modal.Sandbox:
        if self._sandbox is None:
            raise RuntimeError("ModalSandbox not started; call start() first")
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
        proc = await self._require_sandbox().exec.aio(
            *argv,
            timeout=int(timeout) if timeout else None,
            workdir=workdir,
            env=env or None,
        )

        async def _read(stream) -> str:
            try:
                return _to_str(await stream.read.aio())
            except Exception:
                return ""

        stdout, stderr = await asyncio.gather(_read(proc.stdout), _read(proc.stderr))
        exit_code = await proc.wait.aio()
        return ExecResult(exit_code=int(exit_code or 0), stdout=stdout, stderr=stderr)

    async def read_file(self, path: str) -> bytes:
        # Native streamed read; absolute paths only -- relative paths use the exec floor.
        if not path.startswith("/"):
            return await super().read_file(path)
        return await self._require_sandbox().filesystem.read_bytes.aio(path)

    async def write_file(self, path: str, content: bytes | str) -> None:
        # Native streamed write (creates parent dirs); absolute paths only -- relative use the exec floor.
        if not path.startswith("/"):
            return await super().write_file(path, content)
        data = content.encode("utf-8") if isinstance(content, str) else content
        await self._require_sandbox().filesystem.write_bytes.aio(data, path)

    async def upload_file(self, local_file: Path | str, remote_file: str) -> None:
        # Native streamed copy (creates remote dirs, overwrites); trees route here via base upload's tar.
        await self._require_sandbox().filesystem.copy_from_local.aio(str(local_file), remote_file)

    async def download_file(self, remote_file: str, local_file: Path | str) -> None:
        # Native streamed copy (creates local parent dirs); ``remote_file`` must be absolute.
        await self._require_sandbox().filesystem.copy_to_local.aio(remote_file, str(local_file))

    async def expose_port(self, port: int) -> str:
        # Modal needs ports declared via ``encrypted_ports`` at creation (in
        # ``sandbox_kwargs``); once declared, implement via ``self._sandbox.tunnels()``.
        raise NotImplementedError("ModalSandbox.expose_port requires encrypted_ports at sandbox creation time")
