from __future__ import annotations

import abc
import asyncio
import base64
import dataclasses
import logging
import os
import shlex
import tempfile
import uuid
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .utils import (
    extract_dir_from_file,
    pack_dir_to_file,
    remote_pack_command,
    remote_unpack_command,
)

logger = logging.getLogger(__name__)


def _to_str(data: str | bytes | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


@dataclasses.dataclass
class ExecResult:
    """Result of a single one-shot command."""

    exit_code: int
    stdout: str
    stderr: str


class SandboxConfig(BaseModel):
    """Which provider to run, plus its construction kwargs.

    Standard fields feed a provider's :meth:`Sandbox.from_config`; anything
    provider-specific rides along in ``sandbox_kwargs``.
    """

    provider: str = Field(
        description="Registered sandbox provider name (key in SANDBOX_REGISTRY), e.g. 'local' or 'modal'.",
    )
    runtime_timeout: float = Field(
        default=3600.0,
        description="Max sandbox runtime/lifetime (seconds) before it is killed; used by remote providers.",
    )
    image: str = Field(default="python:3.12", description="Container image for remote providers (e.g. modal).")
    sandbox_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra provider-specific kwargs forwarded to the sandbox constructor.",
    )

    model_config = ConfigDict(extra="forbid")


@runtime_checkable
class SandboxBackend(Protocol):
    """Narrow data-plane surface that tools depend on.

    The subset of :class:`Sandbox` a tool needs -- exec, file transfer and an
    optional port tunnel -- deliberately excluding lifecycle (``start`` /
    ``stop``). Any object structurally providing these methods satisfies it.
    """

    async def exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult: ...

    async def exec_shell(
        self,
        script: str,
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult: ...

    async def read_file(self, path: str) -> bytes: ...

    async def write_file(self, path: str, content: bytes | str) -> None: ...

    async def upload(self, local_path: Path | str, remote_path: str) -> None: ...

    async def download(self, remote_path: str, local_path: Path | str) -> None: ...

    async def expose_port(self, port: int) -> str: ...


_DEFAULT_STARTUP_TIMEOUT = 600.0
_DEFAULT_STARTUP_CONCURRENCY = 64
# Per-process startup semaphores, created lazily per event loop so
# each binds to the loop that uses it and is *shared* across concurrent start()s.
_startup_semaphores: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _env_number(name: str, default: float) -> float:
    """Read env var ``name`` as a number, falling back to ``default`` if unset/blank/invalid."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("invalid %s=%r; using default %s", name, raw, default)
        return default


@asynccontextmanager
async def _startup_slot() -> AsyncIterator[None]:
    """Hold one shared startup slot for the running loop (a no-op when ``<=0``).

    The semaphore must be *shared* across all concurrent ``start()`` calls to cap
    anything; a fresh ``asyncio.Semaphore`` per call always acquires immediately
    and limits nothing. It is created lazily per loop (and rebuilt if the limit
    changes) so it binds to the running loop rather than an import-time one.
    """
    limit = int(_env_number("SANDBOX_STARTUP_CONCURRENCY", _DEFAULT_STARTUP_CONCURRENCY))
    if limit <= 0:
        yield
        return
    loop = asyncio.get_running_loop()
    entry = _startup_semaphores.get(loop)
    if entry is None or entry[0] != limit:  # (re)build on first use or when the env-driven limit changes
        entry = (limit, asyncio.Semaphore(limit))
        _startup_semaphores[loop] = entry
    async with entry[1]:
        yield


class Sandbox(abc.ABC):
    """One provider = one class: owns lifecycle and is the data-plane backend.

    Providers implement :meth:`start`, :meth:`stop` and the :meth:`_exec`
    primitive; the public :meth:`exec` wraps ``_exec`` with a shared error
    policy, and the ``bash -lc`` helper and exec-based file transfer build on
    top. :meth:`expose_port` is optional (raises until a provider implements it).
    """

    #: Registry key for this provider, stamped by ``@register_sandbox``.
    provider: ClassVar[str] = ""

    @classmethod
    def from_config(cls, config: SandboxConfig) -> Sandbox:
        """Build an instance from a :class:`SandboxConfig`.

        Default: construct with no args; providers that take constructor kwargs
        override this to map them off ``config``.
        """
        return cls()

    # ----- control plane: lifecycle (owner-facing) -----
    @abc.abstractmethod
    async def start(self) -> None:
        """Create the sandbox and ready the data plane."""
        ...

    @abc.abstractmethod
    async def stop(self) -> None:
        """Terminate the sandbox and release resources."""
        ...

    async def _run_start(self) -> None:
        """Run :meth:`start`, bounding it by the ``SANDBOX_STARTUP_TIMEOUT`` env cap (``<=0`` disables)."""
        timeout = _env_number("SANDBOX_STARTUP_TIMEOUT", _DEFAULT_STARTUP_TIMEOUT)
        try:
            await asyncio.wait_for(self.start(), timeout=timeout if timeout > 0 else None)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"sandbox start() exceeded SANDBOX_STARTUP_TIMEOUT={timeout:g}s") from exc

    async def __aenter__(self, retry: int = 3) -> Sandbox:
        """Create the sandbox (retrying transient ``start()`` failures) and return it ready.

        Each attempt holds one startup slot (``SANDBOX_STARTUP_CONCURRENCY``)
        and is bounded by ``SANDBOX_STARTUP_TIMEOUT``; the slot is released before the
        retry backoff and the cleanup ``stop()``.
        """
        retry = max(1, retry)
        last_exc: BaseException | None = None
        for attempt in range(1, retry + 1):
            try:
                async with _startup_slot():
                    await self._run_start()
                return self
            except Exception as exc:
                last_exc = exc
                try:
                    await self.stop()
                except Exception:
                    logger.warning("sandbox stop() failed during start() cleanup", exc_info=True)
            logger.warning("sandbox failed to start (attempt %d/%d): %r", attempt, retry, last_exc)
            if attempt < retry:
                await asyncio.sleep(2 * attempt)
        assert last_exc is not None
        logger.error("sandbox failed to start after %d attempts: %r", retry, last_exc)
        raise last_exc

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    @asynccontextmanager
    async def entered(self, **start_kwargs: Any) -> AsyncIterator[Sandbox]:
        await self.__aenter__(**start_kwargs)
        try:
            yield self
        finally:
            await self.stop()

    # ----- data plane: providers implement the _exec primitive -----
    @abc.abstractmethod
    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run ``argv`` once via the backend and return its captured result."""
        ...

    async def exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run ``argv`` once and return its captured result (no implicit shell)."""
        try:
            return await self._exec(argv, timeout=timeout, workdir=workdir, env=env)
        except Exception as exc:
            if self._is_timeout_error(exc):
                return ExecResult(exit_code=-1, stdout="", stderr=f"exec timed out after {timeout}s: {exc}")
            if not await self.is_alive():
                raise
            return ExecResult(exit_code=127, stdout="", stderr=str(exc))

    def _is_timeout_error(self, exc: BaseException) -> bool:
        """Whether ``exc`` from :meth:`_exec` represents a command timeout.

        Base recognises the standard :class:`asyncio.TimeoutError` /
        :class:`TimeoutError`; providers whose SDK raises a bespoke timeout type
        override this and OR-in their own check.
        """
        return isinstance(exc, asyncio.TimeoutError | TimeoutError)

    async def is_alive(self) -> bool:
        """Cheap probe for whether the sandbox is still usable.

        Consulted by :meth:`exec` after a non-timeout failure to decide between
        re-raising (dead sandbox) and downgrading to an error result (live
        sandbox). The base assumes always-alive; remote providers override with a
        real probe and must never raise (return ``False`` on any error).
        """
        return True

    async def exec_shell(
        self,
        script: str,
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Convenience: run ``script`` through ``bash -lc``."""
        return await self.exec(["bash", "-lc", script], timeout=timeout, workdir=workdir, env=env)

    async def expose_port(self, port: int) -> str:
        """Return a host-reachable URL/addr for an in-sandbox ``port``.

        Optional capability; providers that cannot tunnel leave it raising.
        """
        raise NotImplementedError

    # ----- files: exec-based floor; override for a native channel -----
    async def read_file(self, path: str) -> bytes:
        """Read and return the bytes of ``path`` (floor: ``base64`` over exec)."""
        # base64 keeps binary content intact across the text-only exec channel.
        res = await self.exec(["base64", path])
        if res.exit_code != 0:
            raise RuntimeError(f"read_file {path!r} failed: {res.stderr.strip()}")
        return base64.b64decode(res.stdout)

    async def write_file(self, path: str, content: bytes | str) -> None:
        """Write ``content`` to ``path`` (floor: ``base64 -d`` over exec)."""
        data = content.encode("utf-8") if isinstance(content, str) else content
        b64 = base64.b64encode(data).decode("ascii")
        q = shlex.quote(path)
        script = f'mkdir -p "$(dirname {q})" && printf %s {shlex.quote(b64)} | base64 -d > {q}'
        res = await self.exec_shell(script)
        if res.exit_code != 0:
            raise RuntimeError(f"write_file {path!r} failed: {res.stderr.strip()}")

    async def upload(self, local_path: Path | str, remote_path: str) -> None:
        """Upload a host file or directory tree into the sandbox.

        A file goes through :meth:`upload_file`; a directory ships as one gzipped
        tar and is unpacked into ``remote_path`` (needs ``tar`` and ``gzip``).
        """
        src = Path(local_path)
        if src.is_dir():
            await self._upload_tree(src, str(remote_path))
        else:
            await self.upload_file(src, str(remote_path))

    async def download(self, remote_path: str, local_path: Path | str) -> None:
        """Download a sandbox file or directory tree to the host.

        A file goes through :meth:`download_file`; a directory is archived,
        pulled as one archive, and extracted locally (needs ``tar`` and ``gzip``).
        """
        remote = str(remote_path)
        if (await self.exec_shell(f"test -d {shlex.quote(remote)}")).exit_code == 0:
            await self._download_tree(remote, local_path)
        else:
            await self.download_file(remote, local_path)

    # ----- single-file transfer: floor over read/write; provider override seam -----
    async def upload_file(self, local_file: Path | str, remote_file: str) -> None:
        """Upload one host file into the sandbox (floor: inline via :meth:`write_file`).

        The override seam for a provider-native single-file fast path.
        """
        await self.write_file(remote_file, Path(local_file).read_bytes())

    async def download_file(self, remote_file: str, local_file: Path | str) -> None:
        """Download one sandbox file to the host (floor: via :meth:`read_file`)."""
        data = await self.read_file(remote_file)
        dst = Path(local_file)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)

    # ----- directory transfer: tar one archive over the single-file seam -----
    async def _upload_tree(self, local_dir: Path, remote_dir: str) -> None:
        """Pack a host dir into one tar, ship via :meth:`upload_file`, unpack in the sandbox."""
        remote_archive = f"/tmp/uni-agent-upload-{uuid.uuid4().hex}.tar.gz"
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "upload.tar.gz"
            pack_dir_to_file(local_dir, archive)
            await self.upload_file(archive, remote_archive)
        try:
            res = await self.exec_shell(remote_unpack_command(remote_archive, remote_dir))
            if res.exit_code != 0:
                raise RuntimeError(
                    f"upload into {remote_dir!r} failed (sandbox needs tar and gzip): {res.stderr.strip()}"
                )
        finally:
            await self.exec(["rm", "-f", remote_archive])

    async def _download_tree(self, remote_dir: str, local_dir: Path | str) -> None:
        """Archive a sandbox dir, pull via :meth:`download_file`, extract locally."""
        dst = Path(local_dir)
        dst.mkdir(parents=True, exist_ok=True)
        remote_archive = f"/tmp/uni-agent-download-{uuid.uuid4().hex}.tar.gz"
        try:
            res = await self.exec_shell(remote_pack_command(remote_dir, remote_archive))
            if res.exit_code != 0:
                raise RuntimeError(
                    f"download of {remote_dir!r} failed (sandbox needs tar and gzip): {res.stderr.strip()}"
                )
            with tempfile.TemporaryDirectory() as tmp:
                archive = Path(tmp) / "download.tar.gz"
                await self.download_file(remote_archive, archive)
                extract_dir_from_file(archive, dst)
        finally:
            await self.exec(["rm", "-f", remote_archive])
