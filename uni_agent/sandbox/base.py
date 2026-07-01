from __future__ import annotations

import abc
import base64
import dataclasses
import logging
import shlex
import tempfile
import uuid
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

    ``provider`` is a registry key (see :data:`uni_agent.sandbox.SANDBOX_REGISTRY`);
    it is a free-form string rather than an enum so new providers can register
    without touching this model. The standard fields below are what a provider's
    :meth:`Sandbox.from_config` may consume; anything provider-specific rides
    along in ``sandbox_kwargs``.
    """

    provider: str = Field(
        default="local",
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
    """Narrow data-plane surface that tools and their channels depend on.

    This is exactly the *subset* of :class:`Sandbox` a tool needs to do its
    work -- exec, file transfer and an optional port tunnel -- and pointedly
    excludes lifecycle (``start`` / ``stop``). Tools annotate their sandbox
    against this protocol, so although they are usually handed the whole
    :class:`Sandbox`, they can only reach the data plane (a shell can't
    terminate the sandbox it lives in). Any object structurally providing these
    methods satisfies it -- a real :class:`Sandbox`, or a test double.
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


class Sandbox(abc.ABC):
    """One provider = one class: owns lifecycle *and* is the data-plane backend.

    Required of a provider: :meth:`start`, :meth:`stop`, :meth:`exec`. Provided
    here: the ``bash -lc`` convenience and the exec-based file floor
    (:meth:`read_file` / :meth:`write_file` / :meth:`upload` / :meth:`download`).
    :meth:`expose_port` is an optional capability (raises until a provider
    implements it).
    """

    #: Registry key for this provider, stamped by ``@register_sandbox``.
    provider: ClassVar[str] = ""

    @classmethod
    def from_config(cls, config: SandboxConfig) -> Sandbox:
        """Build an instance from a :class:`SandboxConfig`.

        Default: construct with no args (good for host-local providers). A
        provider that takes constructor kwargs (image, timeout, ...) overrides
        this to map them off ``config`` (see :class:`ModalSandbox`).
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

    async def __aenter__(self) -> Sandbox:
        try:
            await self.start()
        except BaseException:
            try:
                await self.stop()
            except Exception:
                logger.warning("sandbox stop() failed during start() cleanup", exc_info=True)
            raise
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    # ----- data plane: exec is the one required primitive -----
    @abc.abstractmethod
    async def exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run ``argv`` once and return its captured result (no implicit shell)."""
        ...

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
        """Upload a host file *or* directory tree into the sandbox.

        A file goes through :meth:`upload_file`. A directory is packed into one
        gzipped tar locally, shipped as that single archive, and unpacked into
        ``remote_path`` -- preserving modes / symlinks / empty dirs and avoiding
        a round-trip per file (needs ``tar`` and ``gzip`` in the sandbox image).
        """
        src = Path(local_path)
        if src.is_dir():
            await self._upload_tree(src, str(remote_path))
        else:
            await self.upload_file(src, str(remote_path))

    async def download(self, remote_path: str, local_path: Path | str) -> None:
        """Download a sandbox file *or* directory tree to the host.

        The remote path's type is probed once (``test -d``): a file goes through
        :meth:`download_file`; a directory is archived in the sandbox, pulled as
        one archive, and extracted locally (tar's ``data`` filter guards against
        path traversal). Directory transfer needs ``tar`` and ``gzip``.
        """
        remote = str(remote_path)
        if (await self.exec_shell(f"test -d {shlex.quote(remote)}")).exit_code == 0:
            await self._download_tree(remote, local_path)
        else:
            await self.download_file(remote, local_path)

    # ----- single-file transfer: floor over read/write; provider override seam -----
    async def upload_file(self, local_file: Path | str, remote_file: str) -> None:
        """Upload one host file into the sandbox (floor: inline via :meth:`write_file`).

        The override point for a provider-native single-file fast path. Whole
        trees go through :meth:`upload`'s tar path, which routes the archive here.
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
        remote_archive = f"/tmp/uni-upload-{uuid.uuid4().hex}.tar.gz"
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
        remote_archive = f"/tmp/uni-download-{uuid.uuid4().hex}.tar.gz"
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
