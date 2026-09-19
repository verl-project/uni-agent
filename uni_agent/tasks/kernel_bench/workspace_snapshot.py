"""Bounded KernelBench snapshots, collected by one isolated Python exec.

This file is also sent verbatim with ``python3 -I -c``. Keep it stdlib-only:
never import or execute anything from the agent's workspace. Reward, AST and
implementation/metric binding decisions remain in the host Task.
"""

from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
import re
import stat
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import PurePosixPath
from typing import Any

JSON_LIMIT = 1024 * 1024
IMPLEMENTATION_LIMIT = 2 * 1024 * 1024
STATE_PATHS = (".triton_verify_stop.json", ".triton_verify_patience.json")
METRIC_PATHS = (
    "metrics.json",
    "metrics_best.json",
    "output/verify/verify_result.json",
    "output/verify/verify_result_summary.json",
    "output/verify/perf_result.json",
)


def file_limits(op_name: str, kind: str) -> dict[str, int]:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", op_name):
        raise ValueError("invalid snapshot operator name")
    implementations = {
        f"src/{op_name}_triton_ascend_impl.py": IMPLEMENTATION_LIMIT,
        f"src/{op_name}_triton_ascend_impl_best.py": IMPLEMENTATION_LIMIT,
    }
    if kind == "baseline":
        return implementations
    if kind != "evaluate":
        raise ValueError(f"invalid snapshot kind: {kind!r}")
    return {
        **implementations,
        f"output/verify/{op_name}_triton_ascend_impl.py": IMPLEMENTATION_LIMIT,
        **dict.fromkeys(METRIC_PATHS, JSON_LIMIT),
        # Early-stop and verifier progress travel with the evaluation snapshot.
        **dict.fromkeys(STATE_PATHS, 16384),
    }


@contextmanager
def _directory(path: str, *, parent_fd: int | None = None):
    """Walk components using directory fds; reject symlinks at every level."""
    parts = PurePosixPath(path).parts
    if ".." in parts:
        raise ValueError("parent traversal is not allowed")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY) if parent_fd is None else os.dup(parent_fd)
    try:
        for part in parts:
            if part in {"/", "."}:
                continue
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        yield fd
    finally:
        os.close(fd)


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read_file(root_fd: int, relative: str, limit: int) -> dict[str, Any]:
    path = PurePosixPath(relative)
    try:
        with _directory(str(path.parent), parent_fd=root_fd) as parent:
            # O_NONBLOCK prevents FIFOs/devices from hanging before fstat.
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                return {"status": "not_regular"}
            if before.st_size > limit:
                return {"status": "too_large", "size": before.st_size}
            raw = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
            if _identity(before) != _identity(after) or len(raw) != before.st_size:
                raise RuntimeError(f"file changed during snapshot: {relative}")
            return {
                "status": "ok",
                "size": len(raw),
                "mtime_ns": after.st_mtime_ns,
                "ctime_ns": after.st_ctime_ns,
                "device": after.st_dev,
                "inode": after.st_ino,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "data": base64.b64encode(raw).decode("ascii"),
            }
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return {"status": "missing"}
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            return {"status": "unsafe_path"}
        # Transport / permission / I/O failures are NOT 'missing metrics'.
        raise


def _fingerprints(files: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        path: {key: value for key, value in item.items() if key != "data"}
        for path, item in files.items() if path not in STATE_PATHS
    }


def collect(workspace: str, op_name: str, kind: str) -> dict[str, Any]:
    if not PurePosixPath(workspace).is_absolute() or workspace == "/" or ".." in PurePosixPath(workspace).parts:
        raise ValueError("snapshot workspace must be a dedicated absolute directory")
    limits = file_limits(op_name, kind)
    started = time.monotonic()
    with _directory(workspace) as root:
        directories = {".": True}
        for relative in ("src", "output", "output/verify"):
            try:
                with _directory(relative, parent_fd=root):
                    directories[relative] = True
            except OSError as exc:
                if exc.errno not in {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}:
                    raise
                directories[relative] = False
        files = {}
        for path, limit in limits.items():
            try:
                files[path] = _read_file(root, path, limit)
            except (OSError, RuntimeError):
                if path not in STATE_PATHS:
                    raise
                # A missing/unreadable/changing control-state file must not change reward.
                files[path] = {"status": "unreadable"}
    return {
        "version": 1, "workspace": workspace, "op_name": op_name, "kind": kind,
        "directories": directories, "files": files,
        "collected_at": time.time(), "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
    }


class WorkspaceSnapshot:
    """Host-side view; validates framing, byte limits and all payload digests."""

    def __init__(
        self,
        payload: dict[str, Any],
        workspace: str,
        op_name: str,
        kind: str,
    ) -> None:
        limits = file_limits(op_name, kind)
        if (payload.get("version"), payload.get("workspace"), payload.get("op_name"), payload.get("kind")) != (
            1, workspace, op_name, kind,
        ):
            raise ValueError("snapshot identity/version mismatch")
        self.files = payload.get("files")
        self.directories = payload.get("directories")
        if not isinstance(self.files, dict) or set(self.files) != set(limits):
            raise ValueError("snapshot file manifest mismatch")
        if not isinstance(self.directories, dict) or set(self.directories) != {".", "src", "output", "output/verify"}:
            raise ValueError("snapshot directory manifest mismatch")
        if any(type(value) is not bool for value in self.directories.values()):
            raise ValueError("invalid snapshot directory status")
        self._data = {}
        for path, limit in limits.items():
            item = self.files[path]
            if not isinstance(item, dict):
                raise ValueError(f"invalid snapshot entry: {path}")
            status = item.get("status")
            if status in {"missing", "unsafe_path", "not_regular", "too_large", "unreadable"}:
                if "data" in item:
                    raise ValueError(f"unexpected data for unavailable file: {path}")
                continue
            if status != "ok" or type(item.get("size")) is not int or not 0 <= item["size"] <= limit:
                raise ValueError(f"invalid snapshot file size/status: {path}")
            encoded = item.get("data")
            if not isinstance(encoded, str) or len(encoded) > 4 * ((limit + 2) // 3):
                raise ValueError(f"invalid snapshot encoding/size: {path}")
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) != item["size"] or hashlib.sha256(raw).hexdigest() != item.get("sha256"):
                raise ValueError(f"snapshot byte/digest mismatch: {path}")
            if any(type(item.get(key)) is not int for key in ("mtime_ns", "ctime_ns", "device", "inode")):
                raise ValueError(f"invalid snapshot file metadata: {path}")
            self._data[path] = raw

    def read_bytes(self, path: str) -> bytes | None:
        return self._data.get(path)

    def read_json(self, path: str) -> dict[str, Any] | None:
        raw = self.read_bytes(path)
        if raw is None:
            return None
        try:
            value = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
        except (UnicodeDecodeError, ValueError, RecursionError):
            return None
        return value if isinstance(value, dict) else None

    def digest(self, path: str) -> str | None:
        raw = self.read_bytes(path)
        return hashlib.sha256(raw).hexdigest() if raw else None

    def mtime(self, path: str) -> int | None:
        item = self.files.get(path, {})
        return item["mtime_ns"] // 1_000_000_000 if item.get("status") == "ok" else None

    def fingerprints(self) -> dict[str, dict[str, Any]]:
        return _fingerprints(self.files)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _write_atomic(root_fd: int, relative: str, data: bytes) -> None:
    path = PurePosixPath(relative)
    with _directory(str(path.parent), parent_fd=root_fd) as parent:
        temporary = f".triton-recover-{uuid.uuid4().hex}"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # replace does not follow a pre-existing target symlink.
            os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass


def recover(request: dict[str, Any]) -> dict[str, Any]:
    workspace, op_name = request["workspace"], request["op_name"]
    before = collect(workspace, op_name, "evaluate")
    if _fingerprints(before["files"]) != request["expected_files"]:
        raise RuntimeError("workspace changed between evaluation and best recovery")
    snapshot = WorkspaceSnapshot(before, workspace, op_name, "evaluate")
    staged_path = f"output/verify/{op_name}_triton_ascend_impl.py"
    staged = snapshot.read_bytes(staged_path)
    if not staged or snapshot.digest(staged_path) != request["staged_digest"]:
        raise RuntimeError("staged implementation changed before recovery")
    best_path = f"src/{op_name}_triton_ascend_impl_best.py"
    metrics = {
        **request["metrics"], "implementation_path": best_path,
        "implementation_sha256": request["staged_digest"],
    }
    raw_metrics = (json.dumps(metrics, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
    if len(raw_metrics) > JSON_LIMIT:
        raise ValueError("recovered metrics too large")
    with _directory(workspace) as root:
        _write_atomic(root, best_path, staged)
        _write_atomic(root, "metrics_best.json", raw_metrics)
    # Return fresh bytes for host verification; no extra remote round-trip.
    return collect(workspace, op_name, "evaluate")


def main() -> None:
    request = json.loads(sys.argv[1], parse_constant=_reject_constant)
    if request.get("kind") == "recover":
        result = recover(request)
    else:
        result = collect(
            request["workspace"], request["op_name"], request["kind"],
        )
    print(json.dumps(result, separators=(",", ":"), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
