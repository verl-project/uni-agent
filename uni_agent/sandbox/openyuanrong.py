"""openYuanrong remote sandbox command execution.

This sandbox infra is developed by the OpenYuanrong & Ant Akernel team.

Wraps remote sandbox lifecycle (create, run commands, cleanup and etc.)"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shlex
import subprocess
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
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


def _legacy_sdk_requested(extra_kwargs: dict[str, Any]) -> bool:
    value = extra_kwargs.get("sdk_mode") or os.getenv("OPENYUANRONG_SDK_MODE")
    if value is None and os.getenv("USE_OPENYUANRONG_SDK") == "0":
        value = "legacy"
    return str(value or "").strip().lower() in {"legacy", "akernel", "old", "v0"}


class _LegacyRPC:
    """Serialize legacy ``akernel_sdk`` calls through its Python 3.11 runtime."""

    def __init__(self, python_path: str) -> None:
        worker_path = Path(__file__).with_name("legacy_openyuanrong_worker.py")
        if not worker_path.is_file():
            raise FileNotFoundError(f"legacy OpenYuanRong worker is missing: {worker_path}")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self._process = subprocess.Popen(
            [python_path, str(worker_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
        )
        self._lock = threading.RLock()
        self._request_id = 0

    def call(self, op: str, **fields: Any) -> Any:
        with self._lock:
            if self._process.poll() is not None or self._process.stdin is None or self._process.stdout is None:
                raise RuntimeError("legacy OpenYuanRong worker exited before the RPC completed")
            self._request_id += 1
            request = {"id": self._request_id, "op": op, **fields}
            self._process.stdin.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
            self._process.stdin.flush()
            while True:
                line = self._process.stdout.readline()
                if not line:
                    raise RuntimeError("legacy OpenYuanRong worker closed its RPC stream")
                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    # Older yr dependencies may print a diagnostic during an
                    # RPC. The helper redirects normal noise, but tolerate one
                    # stray line without desynchronizing the protocol.
                    continue
                if response.get("id") != self._request_id:
                    continue
                break
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error") or "legacy OpenYuanRong RPC failed"))
            return response.get("result")

    def close(self) -> None:
        with self._lock:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
            for stream in (self._process.stdin, self._process.stdout):
                if stream is not None:
                    stream.close()


class _LegacyCommands:
    def __init__(self, rpc: _LegacyRPC) -> None:
        self._rpc = rpc

    def run(
        self, command: str, *, envs: dict[str, str] | None = None, cwd: str | None = None, timeout: int = 60
    ) -> Any:
        result = self._rpc.call("exec_shell", command=command, env=envs, workdir=cwd, timeout=timeout)
        return SimpleNamespace(**result)

    def run_argv(
        self,
        argv: list[str],
        *,
        envs: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: int = 60,
    ) -> Any:
        result = self._rpc.call("exec", argv=argv, env=envs, workdir=cwd, timeout=timeout)
        return SimpleNamespace(**result)


class _LegacyFilesystem:
    def __init__(self, rpc: _LegacyRPC) -> None:
        self._rpc = rpc

    def read(self, path: str, format: str = "text") -> str | bytes:
        result = self._rpc.call("read_file", path=path)
        data = base64.b64decode(result["data"])
        return data if format == "bytes" else data.decode("utf-8")

    def write(self, path: str, data: str | bytes) -> Any:
        raw = data.encode("utf-8") if isinstance(data, str) else data
        return self._rpc.call("write_file", path=path, data=base64.b64encode(raw).decode("ascii"))

    def copy_from_local(self, local_path: str, remote_path: str) -> Any:
        return self._rpc.call("upload", local_path=local_path, remote_path=remote_path)

    def copy_to_local(self, remote_path: str, local_path: str) -> Any:
        return self._rpc.call("download", remote_path=remote_path, local_path=local_path)


class _LegacyShell:
    def __init__(self, rpc: _LegacyRPC, shell_id: str) -> None:
        self._rpc = rpc
        self._shell_id = shell_id

    async def run(
        self,
        command: str,
        *,
        envs: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: int = 60,
    ) -> Any:
        result = await asyncio.to_thread(
            self._rpc.call,
            "shell_run",
            shell_id=self._shell_id,
            command=command,
            env=envs,
            workdir=cwd,
            timeout=timeout,
        )
        return SimpleNamespace(**result)

    async def kill(self) -> None:
        await asyncio.to_thread(self._rpc.call, "shell_close", shell_id=self._shell_id)


class _LegacyShells:
    def __init__(self, rpc: _LegacyRPC) -> None:
        self._rpc = rpc

    async def create(self, *, cwd: str | None = None, envs: dict[str, str] | None = None) -> _LegacyShell:
        result = await asyncio.to_thread(self._rpc.call, "shell_create", cwd=cwd, env=envs)
        return _LegacyShell(self._rpc, result["shell_id"])


class _LegacySandboxProxy:
    """Old-SDK-shaped object consumed by the provider's existing data plane."""

    def __init__(self, python_path: str) -> None:
        self._rpc = _LegacyRPC(python_path)
        self.commands = _LegacyCommands(self._rpc)
        self.files = _LegacyFilesystem(self._rpc)
        self.shells = _LegacyShells(self._rpc)
        self.sandbox_id = ""

    def start(self, kwargs: dict[str, Any]) -> None:
        result = self._rpc.call("start", kwargs=kwargs)
        self.sandbox_id = str(result.get("sandbox_id") or "")

    def is_running(self) -> bool:
        return bool(self._rpc.call("is_alive"))

    def get_port_url(self, port: int) -> str:
        return str(self._rpc.call("get_port_url", port=port))

    def get_tunnel_url(self) -> str:
        return str(self._rpc.call("get_tunnel_url"))

    def kill(self) -> None:
        try:
            self._rpc.call("stop")
        finally:
            self._rpc.close()


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
        legacy = _legacy_sdk_requested(self.extra_kwargs)
        sb_kwargs: dict[str, Any] = {
            "image": self.image,
            "cpu": self.cpu,
            "memory": self.memory,
            "cpu_limit": self.cpu_limit,
            "mem_limit": self.mem_limit,
            "idle_timeout": self.idle_timeout,
        }
        if self.mounts:
            sb_kwargs["mounts"] = (
                list(self.mounts) if legacy else [self._coerce_mount(m, _load_sdk()) for m in self.mounts]
            )
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
        if legacy:
            legacy_python = str(
                self.extra_kwargs.get("legacy_python")
                or os.getenv("OPENYUANRONG_LEGACY_PYTHON")
                or "/home/zxh/miniconda3/envs/uni-agent/bin/python"
            )
            sb_kwargs.update(
                {key: value for key, value in self.extra_kwargs.items() if key not in {"sdk_mode", "legacy_python"}}
            )
            proxy = _LegacySandboxProxy(legacy_python)
            try:
                await asyncio.to_thread(proxy.start, sb_kwargs)
            except BaseException:
                try:
                    proxy.kill()
                except BaseException:
                    pass
                raise
            self._sandbox = proxy
            return

        sdk = _load_sdk()
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
        if isinstance(sb, _LegacySandboxProxy):
            if (path_setup := self._path_setup_command()) is not None:
                argv = ["bash", "-c", f"{path_setup}; {shlex.join(argv)}"]
            result = await asyncio.to_thread(
                sb.commands.run_argv,
                argv,
                envs=env,
                cwd=workdir,
                timeout=timeout_i,
            )
            exit_code = int(getattr(result, "exit_code", -1))
            stdout = _to_str(getattr(result, "stdout", ""))
            stderr = _to_str(getattr(result, "stderr", ""))
            if exit_code == -1 and "timed out" in stderr:
                raise TimeoutError(stderr)
            if exit_code != 0:
                raise RuntimeError(stderr or f"command exited with {exit_code}")
            return ExecResult(exit_code=0, stdout=stdout, stderr=stderr)

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
        return ExecResult(exit_code=exit_code, stdout=stdout, stderr=stderr)
