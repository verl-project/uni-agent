from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import ExecResult, Sandbox, _to_str
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import ImageMount, SandboxConfig

logger = logging.getLogger(__name__)


def _ensure_template(image: str, cpu_count: int, memory_mb: int) -> str:
    """Build the image-backed template once and return its deterministic name."""
    import e2b

    image = image.strip()
    if not image:
        raise ValueError("image must be a non-empty string")

    slug = re.sub(r"[^a-z0-9_-]+", "-", image.lower()).strip("-_")[:64] or "image"
    digest = hashlib.sha256(image.encode()).hexdigest()[:12]
    name = f"uni-agent-{slug}-{cpu_count}c-{memory_mb}mb-{digest}"
    if not e2b.Template.exists(name):
        e2b.Template.build(
            e2b.Template().from_image(image),
            name,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
            on_build_logs=None,
        )
    return name


@register_sandbox("e2b")
class E2BSandbox(Sandbox):
    """Create an E2B sandbox from an image-backed template."""

    def __init__(
        self,
        *,
        image: str | None = None,
        template: str | None = None,
        cpu_count: int = 2,
        memory_mb: int = 4096,
        runtime_timeout: float = 3600.0,
        user: str | None = "root",
        metadata: dict[str, str] | None = None,
        envs: dict[str, str] | None = None,
        secure: bool = True,
        allow_internet_access: bool = True,
        network: dict[str, Any] | None = None,
        lifecycle: dict[str, Any] | None = None,
        file_request_timeout: float = 600.0,
        image_mounts: list[ImageMount] | None = None,
        executable_paths: dict[str, str] | None = None,
    ) -> None:
        if runtime_timeout <= 0:
            raise ValueError("runtime_timeout must be positive")
        if file_request_timeout <= 0:
            raise ValueError("file_request_timeout must be positive")
        if cpu_count < 1:
            raise ValueError("cpu_count must be at least 1")
        if memory_mb < 512:
            raise ValueError("memory_mb must be at least 512")
        if image is not None and not image.strip():
            raise ValueError("image must be non-empty when provided")
        if template is not None and not template.strip():
            raise ValueError("template must be non-empty when provided")
        if image is None and template is None:
            raise ValueError("either image or template must be provided")
        if image_mounts:
            raise NotImplementedError(
                "E2BSandbox does not support image mounts; build a composite E2B template instead"
            )

        self.image = image.strip() if image is not None else None
        self.template = template.strip() if template is not None else None
        self.cpu_count = cpu_count
        self.memory_mb = memory_mb
        self.runtime_timeout = float(runtime_timeout)
        self.user = user
        self.metadata = dict(metadata or {})
        self.envs = dict(envs or {})
        self.secure = secure
        self.allow_internet_access = allow_internet_access
        self.network = network
        self.lifecycle = lifecycle
        self.file_request_timeout = float(file_request_timeout)
        self.image_mounts = list(image_mounts or [])
        self.executable_paths = dict(executable_paths or {})

        self._sandbox: Any | None = None
        self.sandbox_id: str | None = None

    @classmethod
    def from_config(cls, config: SandboxConfig) -> E2BSandbox:
        return cls(
            image=config.image,
            runtime_timeout=config.runtime_timeout,
            image_mounts=config.image_mounts,
            executable_paths=config.executable_paths,
            **config.sandbox_kwargs,
        )

    async def start(self) -> None:
        if self._sandbox is not None:
            return

        from e2b import AsyncSandbox

        template = self.template
        if template is None:
            assert self.image is not None
            template = await asyncio.to_thread(
                _ensure_template,
                self.image,
                self.cpu_count,
                self.memory_mb,
            )

        create_kwargs: dict[str, Any] = {
            "template": template,
            "timeout": int(self.runtime_timeout),
            "metadata": self.metadata,
            "envs": self.envs,
            "secure": self.secure,
            "allow_internet_access": self.allow_internet_access,
        }
        if self.network is not None:
            create_kwargs["network"] = self.network
        if self.lifecycle is not None:
            create_kwargs["lifecycle"] = self.lifecycle

        sandbox = await AsyncSandbox.create(**create_kwargs)
        self._sandbox = sandbox
        self.sandbox_id = str(sandbox.sandbox_id)
        await self._setup_executable_paths(self.executable_paths)

    async def stop(self) -> None:
        sandbox, self._sandbox = self._sandbox, None
        sandbox_id, self.sandbox_id = self.sandbox_id, None
        if sandbox is None:
            return
        try:
            await sandbox.kill()
        except Exception as exc:
            logger.warning("Failed to kill E2B sandbox %s: %s", sandbox_id or "?", exc)

    def _require_sandbox(self) -> Any:
        if self._sandbox is None:
            raise RuntimeError("E2BSandbox not started; call start() first")
        return self._sandbox

    async def is_alive(self) -> bool:
        if self._sandbox is None:
            return False
        try:
            return bool(await self._sandbox.is_running())
        except Exception:
            return False

    def _is_timeout_error(self, exc: BaseException) -> bool:
        return type(exc).__name__ == "TimeoutException" or super()._is_timeout_error(exc)

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        from e2b import CommandExitException

        # E2B executes command strings through a shell. ``exec`` replaces that
        # wrapper with the requested process so handle.kill() targets the real
        # command rather than leaving an orphaned child behind.
        command = f"exec {shlex.join(argv)}"
        command_timeout = timeout if timeout is not None and timeout > 0 else 0
        handle = await self._require_sandbox().commands.run(
            command,
            background=True,
            timeout=command_timeout,
            user=self.user,
            envs=env,
            cwd=workdir,
        )
        try:
            result = await handle.wait()
        except CommandExitException as exc:
            return ExecResult(
                exit_code=int(exc.exit_code),
                stdout=_to_str(exc.stdout),
                stderr=_to_str(exc.stderr),
            )
        except BaseException:
            try:
                await asyncio.shield(handle.kill())
            except Exception:
                logger.warning("Failed to kill E2B command in sandbox %s", self.sandbox_id)
            raise
        return ExecResult(
            exit_code=int(result.exit_code or 0),
            stdout=_to_str(result.stdout),
            stderr=_to_str(result.stderr),
        )

    async def read_file(self, path: str) -> bytes:
        data = await self._require_sandbox().files.read(
            path,
            format="bytes",
            user=self.user,
            request_timeout=self.file_request_timeout,
        )
        return bytes(data)

    async def write_file(self, path: str, content: bytes | str) -> None:
        await self._require_sandbox().files.write(
            path,
            content,
            user=self.user,
            request_timeout=self.file_request_timeout,
            use_octet_stream=isinstance(content, bytes),
        )

    async def upload_file(self, local_file: Path | str, remote_file: str) -> None:
        with Path(local_file).open("rb") as stream:
            await self._require_sandbox().files.write(
                remote_file,
                stream,
                user=self.user,
                request_timeout=self.file_request_timeout,
                use_octet_stream=True,
            )

    async def download_file(self, remote_file: str, local_file: Path | str) -> None:
        data = await self.read_file(remote_file)
        destination = Path(local_file)

        def _write() -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)

        await asyncio.to_thread(_write)

    async def expose_port(self, port: int) -> str:
        host = self._require_sandbox().get_host(port)
        if inspect.isawaitable(host):
            host = await host
        host = str(host)
        return host if "://" in host else f"https://{host}"
