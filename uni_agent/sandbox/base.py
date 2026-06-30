"""Sandbox layer: one class per provider = lifecycle + data-plane in one.

A :class:`Sandbox` is the single provider-specific object: it *owns* one
sandbox's lifecycle (:meth:`start` / :meth:`stop`) and *is* the data plane used
to drive it (:meth:`exec` + file transfer + optional :meth:`expose_port`).
Stateful, modality-specific channels (shell / browser / desktop) are layered on
top in the tool layer, which builds them purely on this data plane.

The only primitive a new provider *must* implement is :meth:`exec` (plus the
:meth:`start` / :meth:`stop` lifecycle). File operations ship with an exec-based
floor (``base64`` over the command channel), so a minimal provider gets working
file transfer for free; providers with a native filesystem API override
``upload`` / ``download`` (and optionally ``read_file`` / ``write_file``) for
speed and robustness. No resident HTTP server is installed into the task image;
tools talk to the container purely through :meth:`exec` (plus an optional port
tunnel).

Tools never need the control plane: they depend only on the narrow
:class:`SandboxBackend` protocol (the data-plane subset), so a channel riding
inside a sandbox cannot ``stop`` it. A single :class:`Sandbox` instance is
shared by any number of coexisting tools/channels (e.g. a shell + a browser in
the same container).
"""

from __future__ import annotations

import abc
import base64
import dataclasses
import shlex
import tempfile
import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable

from .utils import (
    extract_dir_from_file,
    pack_dir_to_file,
    remote_pack_command,
    remote_unpack_command,
)


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
        await self.start()
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
