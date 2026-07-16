from __future__ import annotations

import asyncio
import subprocess

import pytest

from tests.deployment.support import (
    FakeHandle,
    FakeProcess,
    cancel_start,
    cid,
    completed,
    fail_start,
    fake_docker,
    fast_backoff,
    hang_start,
    make_apptainer,
    make_oci,
    make_plain,
    reset_limiter,
)
from uni_agent.deployment.local import deployment as mod


def test_cancel_ids(monkeypatch):
    reset_limiter(monkeypatch)
    docker = fake_docker(monkeypatch)

    dep = make_oci(sessioner=hang_start)

    async def _go():
        await cancel_start(dep, until=lambda: dep._owned)
        assert dep._owned == (cid(1), "uni-agent-test-run")
        await dep.stop()

    asyncio.run(_go())
    assert docker["rm"] == [cid(1)]
    assert dep._owned is None


@pytest.mark.parametrize("reap_ok", [True, False])
def test_cancel_keep(monkeypatch, reap_ok):
    reset_limiter(monkeypatch)
    unlinked = []
    if reap_ok:
        proc = FakeProcess()
        handle = FakeHandle()
        path = type("P", (), {"unlink": staticmethod(lambda missing_ok=False: unlinked.append(True))})()
    else:
        proc = FakeProcess(kill=lambda: (_ for _ in ()).throw(OSError("kill failed")))
        handle = FakeHandle(close=lambda: (_ for _ in ()).throw(OSError("close failed")))
        path = type(
            "P", (), {"unlink": staticmethod(lambda missing_ok=False: (_ for _ in ()).throw(OSError("busy")))}
        )()

    async def plant_and_hang(dep):
        dep._server_process = proc
        dep._server_log_handle = handle
        dep._server_log_path = path
        await asyncio.sleep(3600)

    dep = make_apptainer(starter=plant_and_hang)

    asyncio.run(cancel_start(dep, until=lambda: dep._server_process is proc))
    assert dep._abandoned is True
    if reap_ok:
        assert proc.killed and proc.wait_timeout == mod._REAPWAIT
        assert dep._server_process is None
        assert handle.closed and dep._server_log_handle is None
        assert unlinked == [True] and dep._server_log_path is None
    else:
        assert dep._server_process is proc
        assert dep._server_log_handle is handle and dep._server_log_path is path
        assert dep._has_cleanup()
        warnings = dep.logger.messages("warning")
        assert any("could not reap" in m for m in warnings)
        assert any("failed to close" in m for m in warnings)
        assert any("failed to unlink" in m for m in warnings)


def test_stop_retry():
    dep = make_plain(container_runtime="apptainer")
    state = {"terminate": 0, "kill": 0, "kill_fails": True}

    def _terminate():
        state["terminate"] += 1
        raise OSError("terminate failed")

    def _kill():
        state["kill"] += 1
        if state["kill_fails"]:
            raise OSError("kill failed")

    proc = FakeProcess(terminate=_terminate, kill=_kill)
    dep._server_process = proc

    asyncio.run(dep.stop())
    assert dep._server_process is proc
    assert state["terminate"] == 1 and state["kill"] == 1

    state["kill_fails"] = False
    asyncio.run(dep.stop())
    assert dep._server_process is None
    assert state["kill"] == 2


@pytest.mark.parametrize("resource", ["proc", "logs"])
def test_stuck_res(monkeypatch, resource):
    """A retry never starts over a resource the previous attempt's cleanup could not reclaim."""
    reset_limiter(monkeypatch)
    fast_backoff(monkeypatch)
    if resource == "proc":
        stale = FakeProcess(
            terminate=lambda: (_ for _ in ()).throw(OSError("terminate failed")),
            kill=lambda: (_ for _ in ()).throw(OSError("kill failed")),
        )
        attr = "_server_process"
    else:
        stale = type(
            "P", (), {"unlink": staticmethod(lambda missing_ok=False: (_ for _ in ()).throw(OSError("file busy")))}
        )()
        attr = "_server_log_path"

    async def plant_and_fail(dep):
        setattr(dep, attr, stale)
        raise RuntimeError("boom")

    dep = make_apptainer(starter=plant_and_fail)

    with pytest.raises(RuntimeError, match="not retrying"):
        asyncio.run(dep.start(max_retries=2))
    assert getattr(dep, attr) is stale
    assert dep._starts == 1


_SWEEP = {
    "ok": {"rc": 0, "gone": True},
    "no-such": {"rc": 1, "stderr": "Error: No such container: xid", "gone": True},
    "no-container-with": {"rc": 1, "stderr": "Error: no container with name or ID xid", "gone": True},
    "busy": {"rc": 1, "stderr": "daemon busy", "gone": False},
    "timeout": {"rc": None, "gone": False},
}


@pytest.mark.parametrize("case", sorted(_SWEEP))
def test_del_sweep(monkeypatch, case):
    """An id leaves the ledger only on a confirmed removal; any other outcome keeps it and the barrier."""
    spec = _SWEEP[case]
    dep = make_plain()
    dep._owned = (cid(1), "x")
    dep._container_id = cid(1)
    calls = []

    def fake_run(args, **kw):
        calls.append((args, kw))
        if spec["rc"] is None:
            raise subprocess.TimeoutExpired(cmd=args, timeout=kw.get("timeout"))
        return completed(spec["rc"], stderr=spec.get("stderr", ""))

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    dep.__del__()
    args, kwargs = calls[0]
    assert len(calls) == 1
    assert args[:4] == ["docker", "rm", "-f", cid(1)]
    assert kwargs["timeout"] == mod._DELWAIT

    if spec["gone"]:
        assert dep._owned is None and dep._container_id is None
        assert not dep._has_cleanup()
    else:
        assert dep._owned == (cid(1), "x") and dep._container_id == cid(1)
        assert dep._has_cleanup()
        assert any("may leak" in m for m in dep.logger.messages("error"))


def _mute(dep):
    """Give the deployment a logger that raises on every level (a torn-down sink)."""

    def dead(*args, **kwargs):
        raise RuntimeError("the sink is gone")

    dep.logger = type("L", (), {"debug": dead, "info": dead, "warning": dead, "error": dead})()


@pytest.mark.parametrize("case", ["removed", "barrier"])
def test_del_mute(monkeypatch, tmp_path, case):
    """A dead sink never aborts the sweep, in either lifecycle outcome: the orphan resolves to an id
    and is removed (`removed`), or the query cannot answer and the barrier holds (`barrier`)."""
    dep = make_plain()
    cidfile = tmp_path / "cid"
    cidfile.write_text("")
    dep._orphan = (cidfile, "owner-uuid", True)
    _mute(dep)
    calls = []

    def fake_run(args, **kw):
        calls.append(args[1])
        if args[1] != "ps":
            return completed(0)
        if case == "barrier":
            raise subprocess.TimeoutExpired(cmd=args, timeout=kw.get("timeout"))
        return completed(0, stdout=f"{cid(1)}\n")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    dep.__del__()
    if case == "removed":
        assert calls == ["ps", "rm"]
        assert dep._orphan is None and dep._owned is None and not dep._has_cleanup()
    else:
        assert calls == ["ps"]
        assert dep._orphan is not None and not cidfile.exists()


def test_del_logs():
    """A dead sink must not cost the finalizer the apptainer log file it was about to unlink."""
    dep = make_plain(container_runtime="apptainer")
    unlinked = []
    dep._server_log_handle = FakeHandle(close=lambda: (_ for _ in ()).throw(OSError("busy")))
    dep._server_log_path = type("P", (), {"unlink": staticmethod(lambda missing_ok=False: unlinked.append(True))})()
    _mute(dep)

    dep.__del__()
    assert unlinked == [True]  # the failed close did not take the unlink with it
    assert dep._server_log_handle is not None and dep._server_log_path is None


def test_fail_mute(monkeypatch):
    """A sink that raises on the failure record must not take the cleanup that record introduces."""
    reset_limiter(monkeypatch)
    fast_backoff(monkeypatch)
    docker = fake_docker(monkeypatch)
    dep = make_oci(sessioner=fail_start)
    dep.logger.error = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("the sink is gone"))

    with pytest.raises(RuntimeError):
        asyncio.run(asyncio.wait_for(dep.start(max_retries=1), timeout=20))
    assert docker["rm"] == [cid(1)] and dep._owned is None


def test_del_kept(monkeypatch, tmp_path):
    """The finalizer names the container by tag but cannot remove it: the id it learned is kept."""
    dep = make_plain()
    dep._orphan = (tmp_path / "cid", "owner-uuid", True)

    def fake_run(args, **kw):
        if args[1] == "ps":
            return completed(0, stdout=f"{cid(1)}\n")
        return completed(1, stderr="daemon busy")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    dep.__del__()
    assert dep._owned == (cid(1), cid(1)[: mod._IDLEN])
    assert dep._orphan is None and dep._has_cleanup()  # named, so no longer nameless -- but not gone


def test_del_orphan(monkeypatch, tmp_path):
    """The finalizer probes for the container it could not name: by the owner tag, and bounded."""
    dep = make_plain()
    cidfile = tmp_path / "cid"
    dep._orphan = (cidfile, "owner-uuid", True)
    calls = []

    def fake_run(args, **kw):
        calls.append((args, kw))
        if args[1] == "ps":
            return completed(0, stdout=f"{cid(1)}\n")
        return completed(0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    dep.__del__()
    ps = [(a, kw) for a, kw in calls if a[1] == "ps"]
    assert [a[-1] for a, _ in ps] == [f"label={mod._OWNERTAG}=owner-uuid"]
    assert [kw["timeout"] for _, kw in ps] == [mod._DELWAIT]
    assert [a[3] for a, _ in calls if a[1] == "rm"] == [cid(1)]
    assert dep._orphan is None


def test_del_stuck(monkeypatch, tmp_path):
    """A finalizer that still cannot name the container drops the cidfile, logs the owner tag as the
    handle left, and keeps the barrier (a hand-called __del__ must not let a start follow)."""
    dep = make_plain()
    cidfile = tmp_path / "cid"
    cidfile.write_text("")
    dep._orphan = (cidfile, "owner-uuid", False)
    monkeypatch.setattr(mod.subprocess, "run", lambda args, **kw: completed(0, stdout=""))

    dep.__del__()
    assert not cidfile.exists()
    assert dep._orphan is not None and dep._has_cleanup()
    assert any("owner-uuid" in m for m in dep.logger.messages("error"))


def test_del_proc(monkeypatch):
    """__del__ reaps a pending Apptainer server without CLI calls."""
    dep = make_plain(container_runtime="apptainer")
    proc = FakeProcess()
    dep._server_process = proc
    dep._owned = (cid(1), "uni-agent-test-run")
    calls = []
    monkeypatch.setattr(mod.subprocess, "run", lambda args, **kw: calls.append(args) or completed(0))

    dep.__del__()
    assert proc.killed
    assert dep._server_process is None
    assert calls == []
