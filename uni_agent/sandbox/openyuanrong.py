"""openYuanrong remote sandbox command execution.

This sandbox infra is developed by the OpenYuanrong & Ant Akernel team.

Wraps remote sandbox lifecycle (create, run commands, cleanup and etc.)"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import ExecResult, Sandbox, _to_str
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import SandboxConfig

logger = logging.getLogger(__name__)


def _resolve_sandbox_name() -> str | None:
    """Return ``{prefix}{random}`` when ``SANDBOX_NAME_PREFIX`` env is set."""
    prefix = os.getenv("SANDBOX_NAME_PREFIX")
    if not prefix:
        return None
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def _load_sdk() -> Any:
    """Import ``openyuanrong_sandbox`` lazily so this provider stays importable without it."""
    try:
        import yr_sandbox
    except ImportError as exc:
        raise ImportError(
            "the openyuanrong sandbox provider requires the openYuanrong sandbox SDK; "
            "install it with: pip install openyuanrong-sandbox"
        ) from exc
    return yr_sandbox


def _connection_config(sdk: Any) -> Any:
    """Build an SDK ``ConnectionConfig`` from the ``OPENYUANRONG_*`` env vars.

    Only overrides SDK defaults when the matching env var is set:

    * ``OPENYUANRONG_TLS`` → ``use_tls`` (SDK default ``True``)
    * ``OPENYUANRONG_GATEWAY_ADDRESS`` → ``gateway_address`` (SDK default ``None``)
    * ``OPENYUANRONG_GATEWAY_TLS`` → ``gateway_use_tls`` (SDK default ``False``)
    * ``OPENYUANRONG_TLS_VERIFY`` → ``verify_tls`` (SDK default ``False``)
    * ``OPENYUANRONG_TUNNEL_SSL_VERIFY`` → ``YR_TUNNEL_SSL_VERIFY`` (tunnel
      client default ``"1"``; process-env only, no ``ConnectionConfig`` field)
    """
    server = os.getenv("OPENYUANRONG_SERVER_ADDRESS")
    token = os.getenv("OPENYUANRONG_TOKEN")
    if not server or not token:
        raise ValueError(
            "OPENYUANRONG_SERVER_ADDRESS and OPENYUANRONG_TOKEN environment variables must be set for sandbox"
        )
    kwargs: dict[str, Any] = {"server_address": server, "token": token}
    tls = os.getenv("OPENYUANRONG_TLS")
    if tls:
        kwargs["use_tls"] = tls != "0"
    gateway_address = os.getenv("OPENYUANRONG_GATEWAY_ADDRESS")
    if gateway_address:
        kwargs["gateway_address"] = gateway_address
    gateway_tls = os.getenv("OPENYUANRONG_GATEWAY_TLS")
    if gateway_tls:
        kwargs["gateway_use_tls"] = gateway_tls != "0"
    tls_verify = os.getenv("OPENYUANRONG_TLS_VERIFY")
    if tls_verify:
        kwargs["verify_tls"] = tls_verify != "0"
    tunnel_ssl_verify = os.getenv("OPENYUANRONG_TUNNEL_SSL_VERIFY")
    if tunnel_ssl_verify:
        os.environ["YR_TUNNEL_SSL_VERIFY"] = tunnel_ssl_verify
    return sdk.ConnectionConfig(**kwargs)


class _OpenyuanrongShell:
    """Adapt an openyuanrong_sandbox shell to the uni-agent sandbox shell handle.

    Converts the provider shell protocol (``shell.run`` / ``shell.kill``) into
    uni-agent's ``open_shell()`` contract: ``run`` → :class:`ExecResult`,
    ``close`` to release the session. Not killed between ``run`` calls.
    """

    def __init__(self, shell: Any) -> None:
        self._shell = shell

    async def run(self, command: str, *, timeout: float | None = None) -> ExecResult:
        result = await self._shell.run(command, timeout=int(timeout) if timeout else 60)
        return ExecResult(
            exit_code=getattr(result, "exit_code", -1),
            stdout=getattr(result, "stdout", "") or "",
            stderr=getattr(result, "stderr", "") or "",
        )

    async def close(self) -> None:
        try:
            await self._shell.kill()
        except Exception:
            pass


@register_sandbox("openyuanrong")
class OpenyuanrongSandbox(Sandbox):
    """Command execution via remote sandbox."""

    supports_shell = True

    def __init__(
        self,
        *,
        image: str,
        runtime_timeout: float = 3600.0,
        cpu: int = 2000,
        memory: int = 4096,
        cpu_limit: int = 8000,
        mem_limit: int = 12288,
        idle_timeout: int = 7200,
        env: dict[str, str] | None = None,
        add_to_path: list[str] | None = None,
        cwd: str | None = None,
        name: str | None = None,
        mounts: list[Any] | None = None,
        upstream: str | None = None,
        proxy_port: int | None = None,
        port_forwardings: list[int] | None = None,
        **extra_kwargs: Any,
    ) -> None:
        self.image = image
        self.runtime_timeout = runtime_timeout
        self.cpu = cpu
        self.memory = memory
        self.cpu_limit = cpu_limit
        self.mem_limit = mem_limit
        self.idle_timeout = idle_timeout
        self.env = env
        if add_to_path is not None and not isinstance(add_to_path, list):
            raise ValueError("add_to_path must be a list of non-empty strings")
        self.add_to_path = tuple(add_to_path or [])
        if any(not isinstance(path, str) or not path for path in self.add_to_path):
            raise ValueError("add_to_path must be a list of non-empty strings")
        self.cwd = cwd
        self.name = name
        self.mounts = mounts or []
        self.upstream = upstream
        self.proxy_port = proxy_port
        self.port_forwardings = port_forwardings or []
        self.extra_kwargs = extra_kwargs
        self._sandbox: Any = None

    @classmethod
    def from_config(cls, config: SandboxConfig) -> OpenyuanrongSandbox:
        return cls(image=config.image, runtime_timeout=config.runtime_timeout, **config.sandbox_kwargs)

    # ----- public: control plane -----
    async def start(self) -> None:
        if self._sandbox is not None:
            return
        sdk = _load_sdk()
        sb_kwargs: dict[str, Any] = {
            "image": self.image,
            "cpu": self.cpu,
            "memory": self.memory,
            "cpu_limit": self.cpu_limit,
            "mem_limit": self.mem_limit,
            "idle_timeout": self.idle_timeout,
        }
        if self.mounts:
            sb_kwargs["mounts"] = [self._coerce_mount(m, sdk) for m in self.mounts]
        if self.env:
            sb_kwargs["env"] = self.env
        if self.cwd:
            sb_kwargs["cwd"] = self.cwd
        if self.upstream:
            sb_kwargs["upstream"] = self.upstream
        if self.proxy_port:
            sb_kwargs["proxy_port"] = self.proxy_port
        if self.port_forwardings:
            sb_kwargs["port_forwardings"] = list(self.port_forwardings)
        name = _resolve_sandbox_name()
        if name is not None:
            sb_kwargs["name"] = name
        sb_kwargs.update(self.extra_kwargs)
        # An explicit ``connection`` in sandbox_kwargs wins over the env-derived one.
        if "connection" not in sb_kwargs:
            sb_kwargs["connection"] = _connection_config(sdk)
        self._sandbox = await asyncio.to_thread(lambda: sdk.Sandbox(**sb_kwargs))

    async def stop(self) -> None:
        """Kill the sandbox if still running."""
        if self._sandbox is not None:
            sid = getattr(self._sandbox, "sandbox_id", "?")
            try:
                await asyncio.to_thread(self._sandbox.kill)
                logger.info("openyuanrong sandbox %s killed", sid)
            except Exception as e:
                logger.warning("Failed to kill openyuanrong sandbox %s: %s", sid, e)
            self._sandbox = None

    async def is_alive(self) -> bool:
        sb = self._sandbox
        if sb is None:
            return False
        try:
            return bool(await asyncio.to_thread(sb.is_running))
        except Exception:
            return False

    async def open_shell(
        self,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> _OpenyuanrongShell:
        """Return a long-lived SDK shell (cwd/env persist across ``run`` calls)."""
        sb = self._require()
        shell = _OpenyuanrongShell(await sb.shells.create(cwd=cwd, envs=env))
        if (path_setup := self._path_setup_command()) is not None:
            result = await shell.run(path_setup)
            if result.exit_code != 0:
                await shell.close()
                detail = result.stderr or result.stdout or f"exit code {result.exit_code}"
                raise RuntimeError(f"failed to initialize OpenYuanrong shell PATH: {detail}")
        return shell

    # ----- public: data plane (files / ports) -----
    async def read_file(self, path: str) -> bytes:
        """Read via SDK ``files.read(..., format='bytes')``."""
        data = await asyncio.to_thread(lambda: self._require().files.read(path, format="bytes"))
        return data if isinstance(data, bytes) else bytes(data)

    async def write_file(self, path: str, content: bytes | str) -> None:
        """Write via SDK ``files.write``."""
        data: bytes | str = content.encode("utf-8") if isinstance(content, str) else content
        await asyncio.to_thread(self._require().files.write, path, data)

    async def upload(self, local_path: Path | str, remote_path: str) -> None:
        """Upload file or directory via SDK ``files.copy_from_local``."""
        await asyncio.to_thread(self._require().files.copy_from_local, str(local_path), str(remote_path))

    async def download(self, remote_path: str, local_path: Path | str) -> None:
        """Download file or directory via SDK ``files.copy_to_local``."""
        await asyncio.to_thread(self._require().files.copy_to_local, str(remote_path), str(local_path))

    async def expose_port(self, port: int) -> str:
        """Return gateway URL for a port declared in ``port_forwardings``."""
        return await asyncio.to_thread(self._require().get_port_url, port)

    def get_port_url(self, port: int) -> str:
        return self._require().get_port_url(port)

    def get_tunnel_url(self) -> str:
        return self._require().get_tunnel_url()

    # ----- private helpers -----
    def _require(self) -> Any:
        if self._sandbox is None:
            raise RuntimeError("OpenyuanrongSandbox not started; call start() first")
        return self._sandbox

    @staticmethod
    def _coerce_mount(m: Any, sdk: Any) -> Any:
        """Accept a ``Mount`` instance or a dict (``target`` + ``image_url``/``s3_config``).

        The SDK validates types strictly, so a nested ``s3_config`` dict is
        reified into ``S3Config`` before constructing the ``Mount``.
        """
        if isinstance(m, dict):
            m = dict(m)
            if isinstance(m.get("s3_config"), dict):
                m["s3_config"] = sdk.S3Config(**m["s3_config"])
            return sdk.Mount(**m)
        return m

    def _is_timeout_error(self, exc: BaseException) -> bool:
        # openyuanrong_sandbox reports an expired command budget in-band ("Command timed
        # out after ..."); server-side failures may still raise with that wording.
        return "timed out after" in str(exc) or super()._is_timeout_error(exc)

    def _path_setup_command(self) -> str | None:
        """Return a shell command that prepends configured directories to PATH.

        ``Sandbox(env=...)`` has ordinary environment-assignment semantics, so
        setting its ``PATH`` key would replace the task image's original PATH.
        Expanding ``PATH`` in the remote command/shell preserves image tools
        such as conda while giving mounted sidecars precedence.
        """
        if not self.add_to_path:
            return None
        prefix = ":".join(self.add_to_path)
        return f'export PATH={shlex.quote(prefix)}:"${{PATH:-}}"'

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run ``argv`` once via openyuanrong_sandbox ``Commands.run``."""
        sb = self._require()
        timeout_i = int(timeout) if timeout else 60
        command = shlex.join(argv)
        if (path_setup := self._path_setup_command()) is not None:
            command = f"{path_setup}; {command}"
        # commands.run is a blocking SDK poll; run it off the event loop.
        result = await asyncio.to_thread(sb.commands.run, command, envs=env, cwd=workdir, timeout=timeout_i)
        exit_code = int(result.exit_code)
        stdout = _to_str(getattr(result, "stdout", ""))
        stderr = _to_str(getattr(result, "stderr", ""))
        # openyuanrong_sandbox surfaces command timeouts as a result (exit_code=-1), not
        # an exception; re-raise so the shared exec() policy classifies it.
        if exit_code == -1 and "timed out" in stderr:
            raise TimeoutError(stderr)
        if exit_code != 0:
            raise RuntimeError(stderr or f"command exited with {exit_code}")
        return ExecResult(exit_code=0, stdout=stdout, stderr=stderr)
