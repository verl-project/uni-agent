"""OpenSandbox adapter for Uni-Agent sandboxes."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import _DEFAULT_STOP_TIMEOUT, ExecResult, Sandbox, _env_number, _to_str
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import ImageMount, SandboxConfig

logger = logging.getLogger(__name__)

# A command killed at its server-side deadline ends as a signal termination.
# execd reports exit code -1 for those on current servers; servers that predate
# that contract omit the code and carry the raw wait error on ``error.value``
# (e.g. "signal: killed"), so both shapes must map to a caller-budget timeout.
_SIGNAL_KILLED_MARKER = "signal: killed"


def _output(messages: Any) -> str:
    """Join streamed output messages into one string (empty when absent)."""
    return "".join(getattr(message, "text", "") or "" for message in (messages or ()))


@register_sandbox("opensandbox")
class OpenSandboxSandbox(Sandbox):
    """Create an OpenSandbox sandbox and drive it over its async SDK.

    The instance lifetime is enforced server-side (``timeout`` at creation): a
    worker that dies before ``stop()`` still gets the sandbox reclaimed at
    expiry, so a crashed rollout cannot leak it. SDK command sessions and
    snapshots are capabilities this adapter does not expose.
    """

    def __init__(
        self,
        *,
        image: str | None = None,
        snapshot_id: str | None = None,
        runtime_timeout: float = 3600.0,
        ready_timeout: float = 30.0,
        cpu: int = 1,
        memory_gb: int = 2,
        resource: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        entrypoint: list[str] | None = None,
        network_policy: Any | None = None,
        volumes: list[Any] | None = None,
        connection_config: Any | None = None,
        image_mounts: list[ImageMount] | None = None,
        executable_paths: dict[str, str] | None = None,
    ) -> None:
        if image is not None and not image.strip():
            raise ValueError("image must be non-empty when provided")
        if snapshot_id is not None and not snapshot_id.strip():
            raise ValueError("snapshot_id must be non-empty when provided")
        if (image is None) == (snapshot_id is None):
            raise ValueError("exactly one of image or snapshot_id must be provided")
        if runtime_timeout <= 0:
            raise ValueError("runtime_timeout must be positive")
        if ready_timeout <= 0:
            raise ValueError("ready_timeout must be positive")
        if isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 1:
            raise ValueError("cpu must be a positive integer")
        if isinstance(memory_gb, bool) or not isinstance(memory_gb, int) or memory_gb < 1:
            raise ValueError("memory_gb must be a positive integer")
        if image_mounts:
            raise NotImplementedError(
                "OpenSandboxSandbox does not support image mounts; bake tool images "
                "into the sandbox image or use volumes instead"
            )

        self.image = image.strip() if image is not None else None
        self.snapshot_id = snapshot_id.strip() if snapshot_id is not None else None
        self.runtime_timeout = float(runtime_timeout)
        self.ready_timeout = float(ready_timeout)
        self.cpu = cpu
        self.memory_gb = memory_gb
        self.resource = dict(resource) if resource is not None else None
        self.env = dict(env or {})
        self.metadata = dict(metadata or {})
        self.entrypoint = list(entrypoint) if entrypoint is not None else None
        self.network_policy = network_policy
        self.volumes = list(volumes or [])
        self.connection_config = connection_config
        self.image_mounts = list(image_mounts or [])
        self.executable_paths = dict(executable_paths or {})

        self._sandbox: Any | None = None
        self._destroy_task: asyncio.Task | None = None
        self.sandbox_id: str | None = None

    @classmethod
    def from_config(cls, config: SandboxConfig) -> OpenSandboxSandbox:
        return cls(
            image=config.image,
            runtime_timeout=config.runtime_timeout,
            image_mounts=config.image_mounts,
            executable_paths=config.executable_paths,
            **config.sandbox_kwargs,
        )

    # ----- control plane -----
    async def start(self) -> None:
        if self._sandbox is not None:
            if self._destroy_task is None:
                return
            # The handle belongs to a sandbox whose teardown was requested (a
            # stop() that hit SANDBOX_STOP_TIMEOUT): wait for the destroy, then
            # create a fresh instance instead of reporting the stale one ready.
            pending = self._destroy_task
            try:
                # Shielded: a cancel of this start() must not cancel the
                # retained destroy itself -- later retries would then see a
                # cancelled teardown they can never complete.
                await asyncio.shield(pending)
            except Exception:
                logger.warning(
                    "OpenSandbox destroy failed during start() retry (sandbox %s)",
                    self.sandbox_id or "?",
                    exc_info=True,
                )
            if self._destroy_task is pending:
                self._destroy_task = None
            self._sandbox = None
            self.sandbox_id = None
        import opensandbox
        from opensandbox.models.sandboxes import NetworkPolicy

        network_policy = self.network_policy
        if isinstance(network_policy, dict):
            network_policy = NetworkPolicy.model_validate(network_policy)
        create_kwargs: dict[str, Any] = {
            "timeout": timedelta(seconds=self.runtime_timeout),
            "ready_timeout": timedelta(seconds=self.ready_timeout),
            "env": self.env,
            "metadata": self.metadata,
            "resource": self.resource or {"cpu": str(self.cpu), "memory": f"{self.memory_gb}Gi"},
            "entrypoint": self.entrypoint,
            "network_policy": network_policy,
            "volumes": self.volumes,
            "connection_config": self.connection_config,
        }
        if self.image is not None:
            create_kwargs["image"] = self.image
        else:
            create_kwargs["snapshot_id"] = self.snapshot_id
        self._sandbox = await opensandbox.Sandbox.create(**create_kwargs)
        self.sandbox_id = str(self._sandbox.id)
        try:
            await self._setup_executable_paths(self.executable_paths)
        except BaseException:
            # A partially set-up sandbox must not survive a failed start(): the
            # next start() would early-return on the attached-but-unready handle.
            # __aenter__'s failed-start cleanup becomes a no-op after this.
            await self.stop()
            raise

    async def stop(self) -> None:
        """Destroy the sandbox and release local resources.

        The destroy runs in a retained task shielded from caller cancellation:
        a cancel during the await still waits for the remote kill to finish
        before propagating, so the base failed-start cleanup never finds a
        handle whose destruction it cannot retry. The wait is bounded by
        ``SANDBOX_STOP_TIMEOUT`` (``<=0`` disables); on expiry the handle stays
        attached with the destroy in flight -- a later ``stop()`` joins it and
        ``start()`` waits for it before creating a fresh instance -- while the
        server-side lifetime still reclaims the instance. Destroy failures are
        logged, not raised.
        """
        sandbox = self._sandbox
        if sandbox is None:
            return
        if self._destroy_task is not None and not self._destroy_task.done():
            task = self._destroy_task  # join the in-flight destroy, don't stack a second one
        else:
            task = asyncio.create_task(self._destroy_sandbox(sandbox))
            self._destroy_task = task
        timeout = _env_number("SANDBOX_STOP_TIMEOUT", _DEFAULT_STOP_TIMEOUT)
        deadline = None if timeout <= 0 else asyncio.get_running_loop().time() + timeout
        cancelled = False
        destroy_error: BaseException | None = None
        while not task.done():
            remaining = None if deadline is None else deadline - asyncio.get_running_loop().time()
            if remaining is not None and remaining <= 0:
                logger.warning(
                    "OpenSandbox destroy exceeded SANDBOX_STOP_TIMEOUT=%gs (sandbox %s); "
                    "a later stop() retries and the server-side lifetime still reclaims it",
                    timeout,
                    self.sandbox_id or "?",
                )
                if cancelled:
                    raise asyncio.CancelledError()
                return
            try:
                # wait_for bounds the wait; shield keeps the destroy alive when
                # the timeout or a caller cancel interrupts the outer await.
                await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                cancelled = True
            except Exception as exc:
                destroy_error = exc
        if self._sandbox is sandbox:  # a restart must not lose its newer handle
            self._sandbox = None
            self.sandbox_id = None
        if self._destroy_task is task:
            self._destroy_task = None
        if destroy_error is None:
            try:
                await task
            except Exception as exc:
                destroy_error = exc
        if destroy_error is not None:
            # The server-side lifetime still reclaims the instance if destroy fails.
            logger.warning("Failed to destroy OpenSandbox sandbox %s: %s", sandbox.id, destroy_error)
        if cancelled:
            raise asyncio.CancelledError()

    async def _destroy_sandbox(self, sandbox: Any) -> None:
        await sandbox.destroy()

    def _require_sandbox(self) -> Any:
        if self._sandbox is None:
            raise RuntimeError("OpenSandboxSandbox not started; call start() first")
        return self._sandbox

    async def is_alive(self) -> bool:
        if self._sandbox is None:
            return False
        try:
            return bool(await self._sandbox.is_healthy())
        except Exception:
            return False

    def _is_timeout_error(self, exc: BaseException) -> bool:
        # SandboxTimeoutException covers SDK transport and retry deadlines.
        return type(exc).__name__ == "SandboxTimeoutException" or super()._is_timeout_error(exc)

    # ----- data plane -----
    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        from opensandbox.models.execd import RunCommandOpts

        budgeted = timeout is not None and timeout > 0
        opts = RunCommandOpts(
            working_directory=workdir,
            timeout=timedelta(seconds=timeout) if budgeted else None,
            envs=env,
        )
        execution = await self._require_sandbox().commands.run(list(argv), opts=opts)
        return self._result(execution, budgeted=budgeted)

    def _result(self, execution: Any, *, budgeted: bool) -> ExecResult:
        """Convert an SDK execution into :class:`ExecResult`.

        Completed commands keep their exit code, including nonzero ones. A
        signal termination (exit code ``-1``) raises ``TimeoutError`` when the
        caller set a budget -- the server kills the command at the deadline --
        and ``RuntimeError`` otherwise (the signal then came from elsewhere,
        e.g. an OOM kill); the shared error policy turns the former into
        ``exit_code == -1`` and downgrades the latter to ``exit_code == 127``
        while the sandbox is alive.
        """
        logs = getattr(execution, "logs", None)
        stdout = _to_str(_output(getattr(logs, "stdout", None)))
        stderr = _to_str(_output(getattr(logs, "stderr", None)))
        error = getattr(execution, "error", None)
        detail = getattr(error, "value", None) or ""
        exit_code = getattr(execution, "exit_code", None)
        if exit_code is None:
            # No completion event and no recoverable code: the process died on
            # a server predating error.exit_code, or failed to start at all.
            if budgeted and _SIGNAL_KILLED_MARKER in detail:
                raise TimeoutError(f"OpenSandbox command timed out after the caller budget: {detail}")
            raise RuntimeError(f"OpenSandbox command result lost (no exit code): {detail or 'no completion event'}")
        if exit_code < 0:
            if budgeted:
                raise TimeoutError(f"OpenSandbox command timed out after the caller budget: {detail}")
            raise RuntimeError(f"OpenSandbox command was terminated by a signal: {detail}")
        return ExecResult(int(exit_code), stdout, stderr)

    # ----- files: native filesystem API (creates parent directories on write) -----
    async def read_file(self, path: str) -> bytes:
        return bytes(await self._require_sandbox().files.read_bytes(path))

    async def write_file(self, path: str, content: bytes | str) -> None:
        data = content.encode("utf-8") if isinstance(content, str) else content
        await self._require_sandbox().files.write_file(path, data)

    async def upload_file(self, local_file: Path | str, remote_file: str) -> None:
        with Path(local_file).open("rb") as stream:
            await self._require_sandbox().files.write_file(remote_file, stream)

    async def download_file(self, remote_file: str, local_file: Path | str) -> None:
        data = await self.read_file(remote_file)
        destination = Path(local_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    async def expose_port(self, port: int) -> str:
        return await self.get_port_url(port)

    async def get_port_url(self, port: int) -> str:
        """Return the host-reachable URL for an in-sandbox ``port``.

        Scheme-less endpoints (what the SDK returns for host:port addresses)
        take the scheme of the active connection (``http`` by default), not a
        hardcoded https; endpoints that already carry a scheme pass through.
        """
        endpoint = await self.get_port_endpoint(port)
        url = str(getattr(endpoint, "endpoint", endpoint))
        if "://" in url:
            return url
        connection = getattr(self._require_sandbox(), "connection_config", None)
        protocol = str(getattr(connection, "protocol", "") or "http")
        return f"{protocol}://{url}"

    async def get_port_endpoint(self, port: int) -> Any:
        """Return the raw SDK endpoint (URL plus any required request headers)."""
        return await self._require_sandbox().get_endpoint(port)
