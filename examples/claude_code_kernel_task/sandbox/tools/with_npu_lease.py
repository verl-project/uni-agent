#!/usr/bin/env python3
"""Run one verifier while holding a host-shared exclusive Ascend device lock.

This file is an image-build asset. Install it root-owned at
``/opt/triton-agent-tools/with_npu_lease.py`` and pre-create one root-owned,
mode-0666 lock file per exposed evaluator device in a shared sticky directory.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import math
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

_DEVICES_ENV = "TRITON_EVAL_DEVICE_IDS"
_LOCK_DIR_ENV = "TRITON_EVAL_LOCK_DIR"
_TIMEOUT_ENV = "TRITON_EVAL_LOCK_TIMEOUT"
_STATE_DIR = ".triton_verify_processes"


def _devices() -> tuple[str, ...]:
    devices = tuple(part.strip() for part in os.environ.get(_DEVICES_ENV, "").split(","))
    if not devices or any(not part or not part.replace("-", "").replace("_", "").isalnum() for part in devices):
        raise ValueError(f"{_DEVICES_ENV} must contain one or more safe comma-separated device IDs")
    if len(set(devices)) != len(devices):
        raise ValueError(f"{_DEVICES_ENV} contains duplicate device IDs")
    return devices


def _lock_files() -> tuple[tuple[str, Path], ...]:
    lock_dir = Path(os.environ.get(_LOCK_DIR_ENV, "/var/lock/triton-agent-npu"))
    if not lock_dir.is_absolute():
        raise ValueError(f"{_LOCK_DIR_ENV} must name a pre-created absolute directory")
    directory_info = lock_dir.lstat()
    if (
        not stat.S_ISDIR(directory_info.st_mode)
        or directory_info.st_uid != 0
        or directory_info.st_mode & stat.S_ISVTX == 0
    ):
        raise PermissionError(f"lock directory must be a root-owned, non-symlink sticky directory: {lock_dir}")
    files = []
    for device in _devices():
        path = lock_dir / f"device-{device}.lock"
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o002 == 0:
            raise PermissionError(f"lock must be a root-owned writable regular file: {path}")
        files.append((device, path))
    return tuple(files)


def _timeout() -> float:
    value = float(os.environ.get(_TIMEOUT_ENV, "1200"))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{_TIMEOUT_ENV} must be finite and positive")
    return value


def _acquire(files: tuple[tuple[str, Path], ...], timeout: float) -> tuple[str, object]:
    deadline = time.monotonic() + timeout
    while True:
        for device, path in files:
            handle = path.open("r+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return device, handle
            except BlockingIOError:
                handle.close()
        if time.monotonic() >= deadline:
            raise TimeoutError(f"no evaluator NPU lease became available within {timeout:g}s")
        time.sleep(0.2)


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_process_group(pgid: int, grace: float = 5.0) -> None:
    """Quiesce leaked descendants even after their original leader exited."""

    if not _process_group_alive(pgid):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            raise
        return
    deadline = time.monotonic() + grace
    while _process_group_alive(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _process_group_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError as exc:
            if exc.errno != errno.ESRCH:
                raise


def _write_process_state(child_pid: int | None) -> Path:
    directory = Path.cwd() / _STATE_DIR
    directory.mkdir(exist_ok=True)
    path = directory / f"{os.getpid()}.state"
    temporary = path.with_suffix(".tmp")
    kind, child = ("none", 0) if child_pid is None else ("pgid", child_pid)
    temporary.write_text(f"{os.getpid()} {kind} {child}\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate the image/mount contract without locking")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    files = _lock_files()
    if args.check:
        if args.command:
            parser.error("--check does not accept a command")
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("expected -- COMMAND [ARG ...]")
    state_path = _write_process_state(None)
    try:
        device, handle = _acquire(files, _timeout())
        try:
            env = os.environ.copy()
            env.update(
                {
                    "ASCEND_RT_VISIBLE_DEVICES": device,
                    "ASCEND_VISIBLE_DEVICES": device,
                    "NPU_VISIBLE_DEVICES": device,
                    "TRITON_EVAL_LEASED_DEVICE": device,
                }
            )
            # Keep the lock FD inherited by the verifier. If the wrapper is
            # uncatchably killed, the lease remains held until the verifier and
            # its descendants exit instead of being released into live NPU work.
            os.set_inheritable(handle.fileno(), True)
            child = subprocess.Popen(
                command,
                env=env,
                start_new_session=True,
                pass_fds=(handle.fileno(),),
            )
            _write_process_state(child.pid)

            def forward(signum, _frame):
                if child.poll() is None:
                    os.killpg(child.pid, signum)

            previous = {
                signum: signal.signal(signum, forward) for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
            }
            try:
                return child.wait()
            finally:
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
                _terminate_process_group(child.pid)
                if child.poll() is None:
                    child.wait()
        finally:
            # Do not issue LOCK_UN: descendants inherit this open-file
            # description and keep the lease until their final FD is closed.
            handle.close()
    finally:
        state_path.unlink(missing_ok=True)
        try:
            state_path.parent.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TimeoutError, ValueError) as exc:
        print(f"with_npu_lease: {exc}", file=sys.stderr)
        raise SystemExit(75) from exc
