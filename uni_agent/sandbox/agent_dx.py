"""Agent DX Environment SDK adapter for Uni-Agent sandboxes."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import shlex
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from .base import ExecResult, ImageMount, Sandbox, _to_str
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import SandboxConfig

logger = logging.getLogger(__name__)
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_REQUIRED_CAPABILITIES = {"local-close", "explicit-connection-config", "recoverable-commands", "tunnel-proxy-port"}
_COMMAND_RESULT_GRACE_SECONDS = 30.0
_CREATE_RECONCILE_ATTEMPTS = 3
_CREATE_RECONCILE_INTERVAL_SECONDS = 1.0


class AgentDxSandboxError(RuntimeError):
    """An infrastructure failure whose result must not be scored as a command exit."""

    def __init__(
        self,
        code: str,
        *,
        environment_id: str | None = None,
        command_id: str | None = None,
        request_id: str | None = None,
        outcome: str | None = None,
        retry: str | None = None,
        operation_id: str | None = None,
    ) -> None:
        self.code = code
        self.environment_id = environment_id
        self.command_id = command_id
        self.request_id = request_id
        self.outcome = outcome
        self.retry = retry
        self.operation_id = operation_id
        super().__init__(
            f"Agent DX {code} (environment_id={environment_id or '?'}, "
            f"command_id={command_id or '?'}, request_id={request_id or '?'})"
        )


class _CommandTimedOut(TimeoutError):
    """A confirmed remote execution timeout, distinct from a transport timeout."""


def _load_sdk() -> Any:
    try:
        import adx_environment
    except ImportError as exc:
        raise ImportError(
            "the agent_dx provider requires adx-environment; install the verified Agent DX Environment SDK "
            "with: pip install /path/to/agent-dx/platform/sdk/environment/python"
        ) from exc
    return adx_environment


def _connection_config(sdk: Any) -> Any:
    verify = os.getenv("ADX_TLS_VERIFY", "1").strip().lower()
    if verify not in {"0", "1", "false", "true", "no", "yes"}:
        raise ValueError("ADX_TLS_VERIFY must be 0/1, false/true, or no/yes")
    return sdk.ConnectionConfig.from_env(verify_tls=verify in {"1", "true", "yes"})


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _seconds(value: float | None) -> int | None:
    return None if value is None else math.ceil(_positive(value, "timeout"))


def _validate_env(env: dict[str, str] | None) -> None:
    if env is not None and (
        not isinstance(env, dict)
        or any(
            not isinstance(key, str) or not _ENV_NAME.fullmatch(key) or not isinstance(value, str) or "\0" in value
            for key, value in env.items()
        )
    ):
        raise ValueError("env must map POSIX environment variable names to strings without NUL")


def _validate_cwd(cwd: str | None) -> None:
    if cwd is not None and (not isinstance(cwd, str) or not cwd.startswith("/") or "\0" in cwd):
        raise ValueError("cwd must be an absolute POSIX path without NUL")


async def _settle(task: asyncio.Task) -> Any:
    """Collect an already-started operation, including through repeated cancellation."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


@register_sandbox("agent_dx")
class AgentDxSandbox(Sandbox):
    """One task-owned Environment; SDK imports and network access start in start()."""

    supports_shell = True
    supports_reverse_tunnel = True

    @classmethod
    def tunnel_upstream(cls, base_url: str) -> str:
        """Format the HTTP(S) service address expected by the Environment SDK."""
        parsed = urlparse(base_url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("Agent DX gateway tunnel requires an HTTP(S) base_url without embedded credentials")
        host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        address = f"{host}:{port}"
        return f"https://{address}" if parsed.scheme == "https" else address

    def __init__(
        self,
        *,
        image: str,
        runtime_timeout: float = 3600,
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
        image_mounts: list[ImageMount] | None = None,
        executable_paths: dict[str, str] | None = None,
        upstream: str | None = None,
        proxy_port: int | None = None,
        port_forwardings: list[int] | None = None,
        runtime: str | None = None,
        rootfs_readonly: bool = False,
        schedule_timeout: int = 30,
        create_timeout: int = 120,
        init_timeout: int = 30,
        tunnel_connect_timeout: float = 60,
        storage_mb: int | None = None,
        storage_limit_mb: int = 0,
        labels: dict[str, str] | None = None,
        node_id: str | None = None,
        connection: Any = None,
    ) -> None:
        if not isinstance(image, str) or not image.strip():
            raise ValueError("image must be a non-empty string")
        self.image = image.strip()
        self.runtime_timeout = _positive(runtime_timeout, "runtime_timeout")
        for key, value in {
            "cpu": cpu,
            "memory": memory,
            "schedule_timeout": schedule_timeout,
            "create_timeout": create_timeout,
            "init_timeout": init_timeout,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key, value, request in (("cpu_limit", cpu_limit, cpu), ("mem_limit", mem_limit, memory)):
            if isinstance(value, bool) or not isinstance(value, int) or (value != 0 and value < request):
                raise ValueError(f"{key} must be 0 or at least its resource request")
        if isinstance(idle_timeout, bool) or not isinstance(idle_timeout, int) or idle_timeout < 0:
            raise ValueError("idle_timeout must be a non-negative integer")
        if create_timeout < schedule_timeout + 30:
            raise ValueError("create_timeout must leave at least 30 seconds after schedule_timeout")
        _validate_env(env)
        _validate_cwd(cwd)
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError("name must be a non-empty string")
        if add_to_path is not None and (
            not isinstance(add_to_path, list)
            or any(not isinstance(path, str) or not path or "\0" in path for path in add_to_path)
        ):
            raise ValueError("add_to_path must be a list of non-empty strings without NUL")
        if proxy_port is not None and (
            isinstance(proxy_port, bool) or not isinstance(proxy_port, int) or not 2 <= proxy_port <= 65535
        ):
            raise ValueError("proxy_port must be an integer between 2 and 65535")
        if upstream is not None and (not isinstance(upstream, str) or not upstream.strip()):
            raise ValueError("upstream must be a non-empty address")
        ports = list(port_forwardings or [])
        if any(isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535 for port in ports):
            raise ValueError("port_forwardings must contain integers between 1 and 65535")
        if len(ports) != len(set(ports)) or 50090 in ports:
            raise ValueError("port_forwardings must be unique and must not expose the Execd control port 50090")
        effective_proxy = 8766 if proxy_port is None else proxy_port
        if upstream and set(ports).intersection({effective_proxy - 1, effective_proxy}):
            raise ValueError("port_forwardings must not overlap reverse tunnel ports")
        self.env = dict(env or {})
        self.cwd = cwd
        self.name = name
        self.add_to_path = tuple(add_to_path or [])
        self.mounts = list(mounts or [])
        self.image_mounts = list(image_mounts or [])
        self.executable_paths = dict(executable_paths or {})
        self.upstream = upstream
        self.proxy_port = effective_proxy
        self.port_forwardings = ports
        self._create_kwargs = {
            "cpu": cpu,
            "memory": memory,
            "cpu_limit": cpu_limit,
            "mem_limit": mem_limit,
            "idle_timeout": idle_timeout,
            "runtime": runtime,
            "rootfs_readonly": rootfs_readonly,
            "schedule_timeout": schedule_timeout,
            "create_timeout": create_timeout,
            "init_timeout": init_timeout,
            "tunnel_connect_timeout": _positive(tunnel_connect_timeout, "tunnel_connect_timeout"),
            "storage_mb": storage_mb,
            "storage_limit_mb": storage_limit_mb,
            "labels": dict(labels or {}),
            "node_id": node_id,
        }
        self._configured_connection = connection
        self._connection: Any = None
        self._sdk: Any = None
        self._environment: Any = None
        self.environment_id: str | None = None
        self._attempted = False
        self._delete_pending = False
        self._create_error: Exception | None = None
        self._create_unconfirmed = False
        self._create_seen = False
        self._initializing = False
        self._fatal_error: AgentDxSandboxError | None = None
        self._startup_error: BaseException | None = None
        self._timer: asyncio.Task | None = None
        self._lifecycle_lock = asyncio.Lock()

    @classmethod
    def from_config(cls, config: SandboxConfig) -> AgentDxSandbox:
        return cls(
            image=config.image,
            runtime_timeout=config.runtime_timeout,
            image_mounts=config.image_mounts,
            executable_paths=config.executable_paths,
            **config.sandbox_kwargs,
        )

    def _error(self, code: str, command_id: str | None = None) -> AgentDxSandboxError:
        return AgentDxSandboxError(
            code,
            environment_id=self.environment_id,
            command_id=command_id,
            request_id=getattr(self._create_error, "request_id", None),
            outcome=getattr(self._create_error, "outcome", None),
            retry=getattr(self._create_error, "retry", None),
            operation_id=getattr(self._create_error, "operation_id", None),
        )

    def _fail(self, code: str, command_id: str | None = None) -> AgentDxSandboxError:
        error = self._error(code, command_id)
        if self._fatal_error is None:
            self._fatal_error = error
        return error

    def _record_create_error(self, exc: Exception) -> None:
        self._create_error = exc
        if getattr(exc, "outcome", None) == "unknown":
            environment_id = getattr(exc, "environment_id", None)
            if isinstance(environment_id, str) and environment_id.strip():
                self.environment_id = environment_id
            self._create_unconfirmed = self._delete_pending = True

    def raise_if_unavailable(self) -> None:
        if self._fatal_error is not None:
            raise self._fatal_error

    async def __aexit__(self, *exc) -> None:
        await super().__aexit__(*exc)
        if not exc or exc[0] is None:
            self.raise_if_unavailable()

    @asynccontextmanager
    async def entered(self, **start_kwargs: Any) -> AsyncIterator[AgentDxSandbox]:
        async with super().entered(**start_kwargs):
            yield self
        self.raise_if_unavailable()

    def _require(self) -> Any:
        self.raise_if_unavailable()
        if self._environment is None:
            raise RuntimeError("AgentDxSandbox not started; call start() first and use a fresh instance after stop()")
        return self._environment

    def _is_fatal_exec_error(self, exc: BaseException) -> bool:
        return isinstance(exc, AgentDxSandboxError | ValueError | TypeError)

    def _mounts(self, sdk: Any) -> list[Any]:
        mounts = []
        for value in self.mounts:
            if isinstance(value, dict):
                kwargs = dict(value)
                if isinstance(kwargs.get("s3_config"), dict):
                    kwargs["s3_config"] = sdk.S3Config(**kwargs["s3_config"])
                value = sdk.Mount(**kwargs)
            mounts.append(value)
        mounts.extend(sdk.Mount(target=m.mount_path, image_url=m.image) for m in self.image_mounts)
        targets = [str(PurePosixPath("/" + mount.target.lstrip("/"))) for mount in mounts]
        if len(targets) != len(set(targets)):
            raise ValueError("mounts and image_mounts must use unique target paths")
        return mounts

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._startup_error is not None:
                raise self._startup_error
            self.raise_if_unavailable()
            if self._environment is not None:
                return
            if self._attempted:
                raise RuntimeError("AgentDxSandbox cannot restart; construct a fresh instance")
            sdk = self._sdk = _load_sdk()
            if not _REQUIRED_CAPABILITIES.issubset(getattr(sdk, "SDK_CAPABILITIES", ())):
                raise RuntimeError("agent_dx requires an Environment SDK with recoverable commands and local close")
            self._connection = self._configured_connection or _connection_config(sdk)
            if not isinstance(self._connection, sdk.ConnectionConfig):
                raise TypeError("connection must be an adx_environment.ConnectionConfig instance")
            name = self.name
            if name is None and (prefix := os.getenv("SANDBOX_NAME_PREFIX")):
                name = f"{prefix}{uuid.uuid4().hex[:8]}"
            kwargs = dict(
                self._create_kwargs,
                image=self.image,
                env=self.env or None,
                cwd=self.cwd,
                name=name,
                mounts=self._mounts(sdk),
                upstream=self.upstream,
                proxy_port=self.proxy_port,
                port_forwardings=self.port_forwardings,
                connection=self._connection,
            )
            self._attempted = True
            creation = asyncio.create_task(asyncio.to_thread(sdk.Environment, **kwargs))
            try:
                self._environment = await asyncio.shield(creation)
            except asyncio.CancelledError:
                try:
                    self._environment = await _settle(creation)
                except Exception as exc:
                    self._record_create_error(exc)
                else:
                    self.environment_id = self._environment.environment_id
                    self._delete_pending = True
                self._fail("CREATE_CANCELLED")
                try:
                    await _settle(asyncio.create_task(self._delete()))
                except Exception:
                    logger.warning(
                        "Agent DX cancelled creation cleanup failed (environment_id=%s, request_id=%s)",
                        self.environment_id,
                        getattr(self._create_error, "request_id", None),
                    )
                raise
            except Exception as exc:
                self._record_create_error(exc)
                raise self._fail("CREATE_FAILED") from exc
            self.environment_id = self._environment.environment_id
            self._delete_pending = True
            self._timer = asyncio.create_task(self._expire())
            self._initializing = True
            try:
                await self._setup_executable_paths(self.executable_paths)
                self.raise_if_unavailable()
            except BaseException as exc:
                self._startup_error = exc
                self._cancel_timer()
                try:
                    await _settle(asyncio.create_task(self._delete()))
                except Exception:
                    logger.warning("Agent DX startup rollback failed (environment_id=%s)", self.environment_id)
                raise
            finally:
                self._initializing = False

    def _cancel_timer(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None and timer is not asyncio.current_task():
            timer.cancel()

    async def _expire(self) -> None:
        await asyncio.sleep(self.runtime_timeout)
        self._timer = None
        self._fail("RUNTIME_TIMEOUT")
        try:
            await self.stop()
        except Exception:
            logger.warning("Agent DX lifetime cleanup failed (environment_id=%s)", self.environment_id)

    def _created_environment_exists(self) -> bool | None:
        try:
            environment = self._sdk.Environment.from_id(self.environment_id, connection=self._connection)
        except self._sdk.EnvironmentNotFound:
            return False
        except Exception:
            # SDK attachment also fails for non-running instances; teardown must still run.
            return None
        try:
            return True
        finally:
            environment.close()

    async def _reconcile_create(self) -> None:
        """Require observed creation before accepting deletion; otherwise retain pending cleanup."""
        if self.environment_id is None:
            raise self._error("CLEANUP_UNCONFIRMED")
        for attempt in range(_CREATE_RECONCILE_ATTEMPTS):
            self._create_seen |= (await asyncio.to_thread(self._created_environment_exists)) is True
            await asyncio.to_thread(self._sdk.Environment.delete, self.environment_id, connection=self._connection)
            # A pre-create 404 is not final: the original create can still arrive.
            if self._create_seen and (await asyncio.to_thread(self._created_environment_exists)) is False:
                self._create_unconfirmed = False
                return
            if attempt + 1 < _CREATE_RECONCILE_ATTEMPTS:
                await asyncio.sleep(_CREATE_RECONCILE_INTERVAL_SECONDS)
        # Keep the identity pending so a later stop() can retry reconciliation.
        raise self._error("CLEANUP_UNCONFIRMED")

    async def _delete(self) -> None:
        environment, self._environment = self._environment, None
        if not self._delete_pending:
            return
        try:
            if self._create_unconfirmed:
                await self._reconcile_create()
            elif environment is not None:
                try:
                    await asyncio.to_thread(environment.kill)
                except Exception:
                    await asyncio.to_thread(
                        self._sdk.Environment.delete, self.environment_id, connection=self._connection
                    )
            else:
                await asyncio.to_thread(self._sdk.Environment.delete, self.environment_id, connection=self._connection)
            self._delete_pending = False
        except AgentDxSandboxError:
            raise
        except Exception as exc:
            raise self._error("CLEANUP_FAILED") from exc
        finally:
            if environment is not None:
                await asyncio.to_thread(environment.close)

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            self._cancel_timer()
            deletion = asyncio.create_task(self._delete())
            try:
                await asyncio.shield(deletion)
            except asyncio.CancelledError:
                try:
                    await _settle(deletion)
                except Exception:
                    logger.warning(
                        "Agent DX cancelled cleanup failed (environment_id=%s, request_id=%s)",
                        self.environment_id,
                        getattr(self._create_error, "request_id", None),
                    )
                raise

    async def _stop_poisoned(self) -> None:
        if not self._initializing:
            try:
                await _settle(asyncio.create_task(self.stop()))
            except Exception:
                logger.warning(
                    "Agent DX invalidated environment cleanup failed (environment_id=%s)", self.environment_id
                )

    async def is_alive(self) -> bool:
        if self._environment is None or self._fatal_error is not None:
            return False
        try:
            return bool(await asyncio.to_thread(self._environment.is_running))
        except Exception:
            return False

    def _path_setup(self) -> str | None:
        if not self.add_to_path:
            return None
        return f'export PATH={shlex.quote(":".join(self.add_to_path))}:"${{PATH:-}}"'

    def _result(self, result: Any, command_id: str | None = None) -> ExecResult:
        if any(getattr(result, key, False) for key in ("truncated", "stdout_truncated", "stderr_truncated")):
            raise self._fail("OUTPUT_TRUNCATED", command_id)
        status = getattr(result, "status", "SUCCEEDED")
        status = getattr(status, "value", status)
        if status == "TIMED_OUT":
            raise _CommandTimedOut("Agent DX command execution timed out")
        if status not in ("SUCCEEDED", "FAILED") or result.exit_code is None or result.exit_code < 0:
            raise self._fail("COMMAND_RESULT_LOST", command_id)
        if getattr(result, "error_code", None):
            raise self._fail("COMMAND_FAILED", command_id)
        return ExecResult(int(result.exit_code), _to_str(result.stdout), _to_str(result.stderr))

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        environment = self._require()
        budget = _seconds(timeout)
        _validate_env(env)
        _validate_cwd(workdir)
        if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or "\0" in arg for arg in argv):
            raise ValueError("argv must contain strings without NUL")
        command = f"exec {shlex.join(argv)}"
        if path_setup := self._path_setup():
            command = f"{path_setup}; {command}"
        command_id = f"uni-agent-{uuid.uuid4().hex}"
        submission = asyncio.create_task(
            asyncio.to_thread(
                environment.commands.run,
                command,
                background=True,
                command_id=command_id,
                envs=env,
                cwd=workdir if workdir is not None else self.cwd,
                timeout=budget,
            )
        )
        try:
            handle = await asyncio.shield(submission)
            # The server owns the execution budget; allow its terminal result to arrive.
            grace = max(
                _COMMAND_RESULT_GRACE_SECONDS,
                getattr(self._connection, "communication_margin_seconds", 0.0),
            )
            deadline = None if budget is None else asyncio.get_running_loop().time() + budget + grace
            while True:
                remaining = None if deadline is None else deadline - asyncio.get_running_loop().time()
                if remaining is not None and remaining <= 0:
                    error = self._fail("COMMAND_DEADLINE_EXCEEDED", command_id)
                    await self._stop_poisoned()
                    raise error
                value = await handle.wait_async(timeout=10 if remaining is None else min(10, remaining))
                status = getattr(value.status, "value", value.status)
                if status not in ("RUNNING", "PENDING"):
                    return self._result(value, command_id)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self._fail("COMMAND_CANCELLED", command_id)
            try:
                await _settle(submission)
            except Exception:
                pass
            await self._stop_poisoned()
            raise
        except (AgentDxSandboxError, _CommandTimedOut):
            raise
        except Exception as exc:
            raise self._fail("COMMAND_UNAVAILABLE", command_id) from exc

    async def open_shell(self, *, cwd: str | None = None, env: dict[str, str] | None = None) -> _AgentDxShell:
        environment = self._require()
        _validate_cwd(cwd)
        _validate_env(env)
        # Override the SDK's default cwd and initialize quoted values ourselves.
        creation = asyncio.create_task(environment.shells.create(cwd="/", envs=None))
        try:
            native = await asyncio.shield(creation)
        except asyncio.CancelledError:
            self._fail("SHELL_STATE_LOST")
            try:
                await _settle(creation)
            except Exception:
                pass
            await self._stop_poisoned()
            raise
        except Exception as exc:
            raise self._fail("SHELL_STATE_LOST") from exc
        shell = _AgentDxShell(self, native)
        setup = []
        if effective_cwd := cwd if cwd is not None else self.cwd:
            setup.append(f"cd -- {shlex.quote(effective_cwd)}")
        setup.extend(f"export {key}={shlex.quote(value)}" for key, value in (env or {}).items())
        if path_setup := self._path_setup():
            setup.append(path_setup)
        if setup:
            try:
                value = await shell.run(" && ".join(setup))
                if value.exit_code != 0:
                    raise self._fail("SHELL_INIT_FAILED")
            except BaseException:
                await shell.close()
                raise
        return shell

    async def read_file(self, path: str) -> bytes:
        value = await self._file_operation(self._require().files.read, path, format="bytes")
        return bytes(value)

    async def write_file(self, path: str, content: bytes | str) -> None:
        value = content.encode("utf-8") if isinstance(content, str) else content
        await self._file_operation(self._require().files.write, path, value)

    async def upload_file(self, local_file: Path | str, remote_file: str) -> None:
        await self._file_operation(self._require().files.copy_from_local, str(local_file), str(remote_file))

    async def download_file(self, remote_file: str, local_file: Path | str) -> None:
        destination = Path(local_file)
        await asyncio.to_thread(destination.parent.mkdir, parents=True, exist_ok=True)
        await self._file_operation(self._require().files.copy_to_local, str(remote_file), str(destination))

    async def _file_operation(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        transfer = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
        try:
            return await asyncio.shield(transfer)
        except asyncio.CancelledError:
            self._fail("FILE_OPERATION_CANCELLED")
            try:
                await _settle(transfer)
            except Exception:
                pass
            await self._stop_poisoned()
            raise

    async def expose_port(self, port: int) -> str:
        return self.get_port_url(port)

    def get_port_url(self, port: int) -> str:
        return self._require().get_port_url(port)

    def get_tunnel_url(self) -> str:
        return self._require().get_tunnel_url()

    def get_port_auth_headers(self) -> dict[str, str]:
        return self._require().get_port_auth_headers()


class _AgentDxShell:
    def __init__(self, sandbox: AgentDxSandbox, shell: Any) -> None:
        self._sandbox = sandbox
        self._shell = shell
        self._closed = False

    async def run(self, command: str, *, timeout: float | None = None) -> ExecResult:
        self._sandbox._require()
        if self._closed:
            raise RuntimeError("Agent DX shell is closed")
        # The persistent-shell SDK requires an integer deadline, unlike Commands.run.
        budget = _seconds(60 if timeout is None else timeout)
        try:
            value = await self._shell.run(command, timeout=budget)
            if value.exit_code is None or value.exit_code < 0:
                raise self._sandbox._fail("SHELL_STATE_LOST")
            return self._sandbox._result(value)
        except asyncio.CancelledError:
            self._closed = True
            self._sandbox._fail("SHELL_STATE_LOST")
            await self._sandbox._stop_poisoned()
            raise
        except Exception as exc:
            self._closed = True
            error = self._sandbox._fail("SHELL_STATE_LOST")
            await self._sandbox._stop_poisoned()
            raise error from exc

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self._shell.kill()
