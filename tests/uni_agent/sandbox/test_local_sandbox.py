import asyncio
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

from uni_agent.sandbox.local import LocalSandbox

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.level0,
    pytest.mark.asyncio,
    pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups and ps"),
]


def _running(pid: int) -> bool:
    # /proc is absent on macOS; a missing procfs entry there is not proof of exit.
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False, timeout=2
    )
    assert result.returncode in {0, 1} and not result.stderr.strip(), f"cannot observe PID {pid}: {result.stderr}"
    status = result.stdout.strip()
    if not status:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        raise AssertionError(f"PID {pid} exists but ps did not return its state")
    return not status.startswith("Z")


async def _read_pid(path: Path) -> int:
    for _ in range(100):
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        await asyncio.sleep(0.02)
    raise AssertionError(f"{path} was not written")


async def _assert_gone(pid: int) -> None:
    for _ in range(100):
        if not _running(pid):
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"process {pid} is still running")


async def test_exec_timeout_kills_background_children(tmp_path):
    pidfile = tmp_path / "pid"
    res = await LocalSandbox().exec(["bash", "-c", f"sleep 30 & echo $! > {pidfile}; wait"], timeout=0.5)
    assert res.exit_code == -1
    await _assert_gone(await _read_pid(pidfile))


async def test_exec_timeout_returns_when_a_detached_process_holds_the_pipes(tmp_path):
    pidfile = tmp_path / "pid"
    # Python exposes setsid semantics on both Linux and macOS without requiring
    # the Linux util-linux command. The detached child inherits the exec pipes.
    child_script = (
        f"import os, pathlib, time; pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(30)"
    )
    script = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-I', '-c', {child_script!r}], start_new_session=True); time.sleep(30)"
    )
    start = time.monotonic()
    try:
        res = await asyncio.wait_for(LocalSandbox().exec([sys.executable, "-I", "-c", script], timeout=0.5), 5)
    finally:
        # Descendant-group cleanup may have already terminated and reaped it.
        with suppress(ProcessLookupError):
            os.kill(await _read_pid(pidfile), signal.SIGKILL)
    assert res.exit_code == -1 and time.monotonic() - start < 2


async def test_exec_cancellation_kills_process(tmp_path):
    pidfile = tmp_path / "pid"
    task = asyncio.create_task(LocalSandbox().exec(["bash", "-c", f"echo $$ > {pidfile}; exec sleep 30"]))
    pid = await _read_pid(pidfile)
    assert _running(pid), "the observer must detect the live fixture before cancellation"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _assert_gone(pid)


async def test_exec_does_not_inherit_stdin():
    saved, (read_end, write_end) = os.dup(0), os.pipe()
    os.dup2(read_end, 0)
    try:
        # Check the device itself instead of a Linux-only procfs symlink.
        script = (
            "import os, stat; actual = os.fstat(0); expected = os.stat(os.devnull); "
            "print(stat.S_ISCHR(actual.st_mode) and actual.st_rdev == expected.st_rdev)"
        )
        res = await LocalSandbox().exec([sys.executable, "-I", "-c", script], timeout=5)
    finally:
        os.dup2(saved, 0)
        for fd in (saved, read_end, write_end):
            os.close(fd)
    assert res.exit_code == 0
    assert res.stdout.strip() == "True"
