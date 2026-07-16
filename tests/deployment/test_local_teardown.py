from __future__ import annotations

import asyncio
import subprocess
import threading
import time

import pytest

from tests.deployment.support import (
    FakeHandle,
    FakeProcess,
    cid,
    completed,
    fail_start,
    fast_backoff,
    fast_rm,
    hang_start,
    make_loop,
    make_plain,
    reset_limiter,
)
from uni_agent.deployment.local import deployment as mod


@pytest.mark.parametrize(("call", "bound"), [("exec", "_CLIWAIT"), ("logs", "_LOGWAIT"), ("rm", "_RMWAIT")])
def test_cli_bounds(monkeypatch, call, bound):
    dep = make_plain()
    seen = []
    monkeypatch.setattr(
        mod.subprocess, "run", lambda args, **kw: seen.append(kw.get("timeout")) or completed(0, stdout="ok")
    )
    if call == "exec":
        dep._runtime_exec(["docker", "inspect", cid(1)])
    elif call == "logs":
        dep._get_container_logs(cid(1))
    else:
        dep._force_remove(cid(1))
    assert seen == [getattr(mod, bound)]


def test_log_bound(monkeypatch):
    """A log fetch that outlives its bound keeps the slot, so stop() waits for it before any rm."""
    monkeypatch.setattr(mod, "_LOGWAIT", 0.1)
    dep = make_plain()
    dep._container_id = cid(1)
    dep._owned = (cid(1), "n-x")
    release = threading.Event()
    order = []

    def _exec(args, check=True, timeout=None):
        order.append(args[1])
        if args[1] == "logs":
            release.wait(30)
        return completed(0)

    dep._runtime_exec = _exec

    async def _go():
        assert await dep._collect_logs("x") == "<log collection failed or timed out>"
        assert dep._has_cli()  # the fetch runs on; the slot is still taken

        stopping = asyncio.create_task(dep.stop())
        await asyncio.sleep(0.05)
        assert order == ["logs"]  # no rm has overtaken the log fetch
        assert not stopping.done()
        release.set()
        await asyncio.wait_for(stopping, timeout=5)

    asyncio.run(asyncio.wait_for(_go(), timeout=10))
    assert order == ["logs", "rm"]
    assert dep._future is None
    assert dep._owned is None


def test_close_bound(monkeypatch):
    """Bound a hung runtime.close() so container teardown can continue."""
    monkeypatch.setattr(mod, "_CLOSEWAIT", 0.1)
    dep = make_plain()
    dep._runtime = type("R", (), {"close": staticmethod(lambda: asyncio.sleep(3600))})()
    dep._owned = (cid(1), "a")
    dep._container_id = cid(1)
    dep._secrets = {"tok"}
    dep._runtime_exec = lambda args, check=True, timeout=None: completed(0)

    asyncio.run(asyncio.wait_for(dep.stop(), timeout=5))
    assert dep._runtime is None
    assert dep._owned is None
    assert not dep._has_cleanup()
    assert dep._secrets == set()  # nothing is left that could still log it
    assert any("runtime.close() failed" in m for m in dep.logger.messages("warning"))


def test_clean_wait(monkeypatch):
    """A hung teardown between retries is bounded by _CLEANWAIT and reported as a possible leak."""
    reset_limiter(monkeypatch)
    fast_backoff(monkeypatch)
    monkeypatch.setattr(mod, "_CLEANWAIT", 0.2)
    dep = make_loop(starter=fail_start, stopper=hang_start)

    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match=r"after 2 attempt"):
        asyncio.run(asyncio.wait_for(dep.start(max_retries=2), timeout=10))
    assert time.monotonic() - t0 < 5.0
    assert any("may leak" in m for m in dep.logger.messages("warning"))


# A removal that the runtime confirms -- it went, or it was never there -- is not retried; one it
# could not confirm is, exactly once.
_RM_CASES = {
    "ok": {"gone": True, "calls": 1},
    "no-such-container": {"gone": True, "calls": 1},
    "daemon-busy": {"gone": False, "calls": 2},
    "timeout": {"gone": False, "calls": 2},
}


@pytest.mark.parametrize("case", sorted(_RM_CASES))
def test_remove(monkeypatch, case):
    fast_rm(monkeypatch)
    dep = make_plain()
    calls = []

    def _exec(args, check=True, timeout=None):
        calls.append(args)
        if case == "timeout":
            raise mod.CliTimeout(f"rm: timed out after {timeout:.0f}s")
        if case == "no-such-container":
            return completed(1, stderr=f"Error: No such container: {cid(1)}")
        return completed(1, stderr="daemon busy") if case == "daemon-busy" else completed(0)

    dep._runtime_exec = _exec
    gone = _RM_CASES[case]["gone"]
    assert dep._force_remove(cid(1)) is gone
    assert len(calls) == _RM_CASES[case]["calls"]
    if not gone:
        assert any("retrying once" in m for m in dep.logger.messages("warning"))
    else:
        assert dep.logger.messages("warning") == []


def test_sweep_again(monkeypatch):
    fast_rm(monkeypatch)
    dep = make_plain()
    dep._owned = (cid(1), "a")
    dep._container_id = cid(1)
    state = {"fail": True}
    dep._runtime_exec = lambda args, check=True, timeout=None: completed(
        1 if state["fail"] else 0, stderr="daemon busy" if state["fail"] else ""
    )

    asyncio.run(dep.stop())
    assert dep._owned == (cid(1), "a")
    assert any("may leak" in m for m in dep.logger.messages("error"))

    state["fail"] = False
    asyncio.run(dep.stop())
    assert dep._owned is None


def test_stop_fails():
    """A process that survives SIGKILL is retained, and cleanup still finishes."""
    dep = make_plain(container_runtime="apptainer")
    state = {"kills": 0}

    def _kill():
        state["kills"] += 1

    def _wait(timeout=None):
        raise subprocess.TimeoutExpired(cmd="server", timeout=timeout)

    proc = FakeProcess(kill=_kill, wait=_wait)
    proc.terminate = lambda: None
    dep._server_process = proc
    handle = FakeHandle()
    dep._server_log_handle = handle

    asyncio.run(dep.stop())
    assert dep._server_process is proc
    assert handle.closed and dep._server_log_handle is None
    assert state["kills"] == 1
    assert any("did not exit after SIGKILL" in m for m in dep.logger.messages("error"))


@pytest.mark.parametrize("resource", ["handle", "path"])
def test_log_cleanup(resource):
    """Retain failed log cleanup so a later stop() can retry."""
    dep = make_plain(container_runtime="apptainer")
    state = {"n": 0}

    def _fail_once():
        state["n"] += 1
        if state["n"] == 1:
            raise OSError("busy")

    if resource == "handle":
        target = FakeHandle(close=_fail_once)
        dep._server_log_handle = target
        attr = "_server_log_handle"
    else:
        target = type("P", (), {"unlink": staticmethod(lambda missing_ok=False: _fail_once())})()
        dep._server_log_path = target
        attr = "_server_log_path"

    asyncio.run(dep.stop())
    assert getattr(dep, attr) is target
    assert dep._has_cleanup()

    asyncio.run(dep.stop())
    assert getattr(dep, attr) is None
    assert state["n"] == 2
