from __future__ import annotations

import asyncio
import subprocess
import types
import weakref
from pathlib import Path

from uni_agent.deployment.local import deployment as mod
from uni_agent.deployment.local.deployment import LocalDeployment


def reset_limiter(monkeypatch, *, per_worker=16, wall_budget=600.0):
    monkeypatch.setenv("LOCAL_MAX_STARTING_PER_WORKER", str(per_worker))
    monkeypatch.setenv("LOCAL_INIT_WALL_BUDGET", str(wall_budget))
    monkeypatch.setattr(mod, "_LIMITERS", weakref.WeakKeyDictionary(), raising=True)


def fast_backoff(monkeypatch):
    monkeypatch.setattr(mod, "_retrydelay", lambda s: asyncio.sleep(0))


def fast_rm(monkeypatch):
    """Skip the real pause between the two rm attempts."""
    monkeypatch.setattr(mod, "_RMPAUSE", 0)


class RecordingLogger:
    def __init__(self):
        self.records = []

    def _log(self, level, template, *args):
        self.records.append((level, template, args))

    def debug(self, template, *args):
        self._log("debug", template, *args)

    def info(self, template, *args):
        self._log("info", template, *args)

    def warning(self, template, *args):
        self._log("warning", template, *args)

    def error(self, template, *args):
        self._log("error", template, *args)

    def messages(self, level=None):
        return [
            template.format(*args) if args else template
            for lvl, template, args in self.records
            if not level or lvl == level
        ]


class FakeRuntime:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class FakeHandle:
    def __init__(self, *, close=None):
        self.closed = False
        self._close = close

    def close(self):
        if self._close:
            return self._close()
        self.closed = True


class FakeProcess:
    def __init__(self, *, terminate=None, kill=None, wait=None):
        self.terminated = False
        self.killed = False
        self.wait_timeout = None
        self._terminate = terminate
        self._kill = kill
        self._wait = wait

    def poll(self):
        return None

    def terminate(self):
        if self._terminate:
            return self._terminate()
        self.terminated = True

    def kill(self):
        if self._kill:
            return self._kill()
        self.killed = True

    def wait(self, timeout=None):
        self.wait_timeout = timeout
        if self._wait:
            return self._wait(timeout)
        return 0


async def _ok(dep=None):
    return None


async def fail_start(dep):
    raise RuntimeError("boom")


async def hang_start(dep):
    await asyncio.sleep(3600)


class LoopDeployment(LocalDeployment):
    """Drives the production retry loop with a faked OCI start."""

    async def _enter(self, container_name, published_port):
        self._starts += 1
        self._names.append(container_name)
        self._ports.append(published_port)
        await self._starter(self)

    async def _start_oci_container(self, token, container_name, published_port):  # type: ignore[override]
        await self._enter(container_name, published_port)
        self._runtime = types.SimpleNamespace(create_session=self._session, close=_ok)

    async def _session(self, request):
        await self._sessioner(self)

    async def _wait_until_alive(self, timeout):  # type: ignore[override]
        return None

    def _get_container_logs(self, ref):  # type: ignore[override]
        self._events.append("logs")
        return ""

    async def _cleanup(self):  # type: ignore[override]
        await self._stopper(self)


class OciDeployment(LoopDeployment):
    """Drives the production OCI start too, so ownership is exercised; use with fake_docker()."""

    async def _start_oci_container(self, token, container_name, published_port):  # type: ignore[override]
        await self._enter(container_name, published_port)
        await LocalDeployment._start_oci_container(self, token, container_name, published_port)
        self._runtime = types.SimpleNamespace(create_session=self._session, close=_ok)

    _cleanup = LocalDeployment._cleanup


class ApptDeployment(LoopDeployment):
    """Fakes Apptainer startup while exercising production cleanup."""

    async def _start_apptainer(self, token, published_port):  # type: ignore[override]
        await self._enter("apptainer", published_port)
        self._runtime = types.SimpleNamespace(create_session=self._session, close=_ok)

    _cleanup = LocalDeployment._cleanup


def _new(cls, config):
    config.setdefault("container_runtime", "docker")
    dep = cls("test-run", type="local", **config)
    dep.logger = RecordingLogger()
    return dep


def _wire(dep, *, starter, stopper, sessioner):
    dep._starts = 0
    dep._names = []
    dep._ports = []
    dep._events = []
    dep._starter = starter or _ok
    dep._stopper = stopper or _ok
    dep._sessioner = sessioner or _ok
    return dep


def make_plain(**config):
    return _new(LocalDeployment, config)


def make_loop(*, starter=None, stopper=None, **config):
    return _wire(_new(LoopDeployment, config), starter=starter, stopper=stopper, sessioner=None)


def make_oci(*, starter=None, sessioner=None, **config):
    return _wire(_new(OciDeployment, config), starter=starter, stopper=None, sessioner=sessioner)


def make_apptainer(*, starter=None, sessioner=None, **config):
    config["container_runtime"] = "apptainer"
    return _wire(_new(ApptDeployment, config), starter=starter, stopper=None, sessioner=sessioner)


def cid(n: int = 1) -> str:
    """A container id shaped like the runtime's own: 64 lowercase hex. Anything shorter is a partial
    write, and the deployment refuses to claim one."""
    return f"{n:064x}"


def completed(returncode=0, stderr="", stdout=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stderr=stderr, stdout=stdout)


def fake_docker(monkeypatch, *, started=None, release=None, result=None):
    """A docker CLI double. `started`/`release` let a test hold `docker run` inside its thread;
    `result` overrides the run's CompletedProcess (e.g. a non-zero exit)."""
    state = {"rm": [], "runs": 0}

    def fake_run(args, **kwargs):
        if args[1] == "run":
            state["runs"] += 1
            container_id = cid(state["runs"])
            Path(args[args.index("--cidfile") + 1]).write_text(f"{container_id}\n")
            if started is not None:
                started.set()
            if release is not None:
                release.wait(timeout=10)
            if result is not None:
                return result
            return completed(0, stdout=f"{container_id}\n")
        if args[1] == "rm":
            state["rm"].append(args[3])
        return completed(0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    return state


async def cancel_start(dep, *, until):
    import pytest

    task = asyncio.create_task(dep.start())
    for _ in range(400):
        if until():
            break
        await asyncio.sleep(0.005)
    else:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
        raise AssertionError("cancel_start: until() never became true")
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=10)
