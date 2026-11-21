import asyncio
import os
import signal
import time
from pathlib import Path

import pytest

from uni_agent.sandbox.local import LocalSandbox

pytestmark = [pytest.mark.cpu, pytest.mark.level0, pytest.mark.asyncio]


def _running(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


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
    script = f"setsid bash -c 'echo $$ > {pidfile}; exec sleep 30' & sleep 30"
    start = time.monotonic()
    try:
        res = await asyncio.wait_for(LocalSandbox().exec(["bash", "-c", script], timeout=0.5), 5)
    finally:
        os.kill(await _read_pid(pidfile), signal.SIGKILL)
    assert res.exit_code == -1 and time.monotonic() - start < 2


async def test_exec_cancellation_kills_process(tmp_path):
    pidfile = tmp_path / "pid"
    task = asyncio.create_task(LocalSandbox().exec(["bash", "-c", f"echo $$ > {pidfile}; exec sleep 30"]))
    pid = await _read_pid(pidfile)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _assert_gone(pid)


async def test_exec_does_not_inherit_stdin():
    saved, (read_end, write_end) = os.dup(0), os.pipe()
    os.dup2(read_end, 0)
    try:
        res = await LocalSandbox().exec(["readlink", "/proc/self/fd/0"], timeout=5)
    finally:
        os.dup2(saved, 0)
        for fd in (saved, read_end, write_end):
            os.close(fd)
    assert res.stdout.strip() == "/dev/null"
