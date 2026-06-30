from __future__ import annotations

import shlex
import tarfile
from pathlib import Path
from typing import BinaryIO


def pack_dir(source_dir: Path | str, fileobj: BinaryIO, *, compress: bool = True) -> None:
    """Pack *source_dir* into a tar stream, rooted at ``.`` (full fidelity).

    Permissions, symlinks (kept as links, not followed) and empty directories
    are preserved; the ``.`` root lets the archive extract straight into a
    target directory.
    """
    source_path = Path(source_dir)
    if not source_path.is_dir():
        raise FileNotFoundError(f"source directory {source_dir!r} does not exist")
    with tarfile.open(fileobj=fileobj, mode="w:gz" if compress else "w") as tar:
        tar.add(source_path, arcname=".")


def pack_dir_to_file(source_dir: Path | str, archive_path: Path | str, *, compress: bool = True) -> None:
    """Pack *source_dir* into a tar archive file (see :func:`pack_dir`)."""
    with Path(archive_path).open("wb") as fileobj:
        pack_dir(source_dir, fileobj, compress=compress)


def extract_dir(fileobj: BinaryIO, target_dir: Path | str) -> None:
    """Extract a tar stream into *target_dir*.

    Uses tar's ``data`` filter: rejects absolute paths, ``..`` traversal and
    links escaping the destination, while keeping exec bits and in-tree symlinks.
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=fileobj, mode="r:*") as tar:
        tar.extractall(target_path, filter="data")


def extract_dir_from_file(archive_path: Path | str, target_dir: Path | str) -> None:
    """Extract a tar archive file into *target_dir* (see :func:`extract_dir`)."""
    with Path(archive_path).open("rb") as fileobj:
        extract_dir(fileobj, target_dir)


def remote_pack_command(source_dir: str, archive_path: str) -> str:
    """POSIX-shell command packing a sandbox directory into a gzipped archive."""
    return f"tar -czf {shlex.quote(archive_path)} -C {shlex.quote(source_dir)} ."


def remote_unpack_command(archive_path: str, target_dir: str) -> str:
    """POSIX-shell command extracting a staged gzipped archive in the sandbox.

    tar exits non-zero on a truncated/corrupt archive (gzip CRC), so a partial
    transfer fails loudly instead of dropping files.
    """
    return f"mkdir -p {shlex.quote(target_dir)} && tar -xzf {shlex.quote(archive_path)} -C {shlex.quote(target_dir)}"
