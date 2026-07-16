from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import shutil
import subprocess
import threading
from pathlib import Path

import pytest
from loguru import logger as loguru_logger

from tests.deployment.support import (
    FakeProcess,
    FakeRuntime,
    cid,
    completed,
    fail_start,
    fake_docker,
    fast_backoff,
    fast_rm,
    make_loop,
    make_oci,
    make_plain,
    reset_limiter,
)
from uni_agent.async_logging import add_file_handler, cleanup_handlers
from uni_agent.deployment.local import deployment as mod


def test_logs_gated(monkeypatch):
    reset_limiter(monkeypatch)
    dep = make_loop(starter=fail_start)

    with pytest.raises(RuntimeError):
        asyncio.run(dep.start(max_retries=1))
    assert "logs" not in dep._events
    assert any("creation unconfirmed" in m for m in dep.logger.messages("error"))


def test_cancel_reap(monkeypatch):
    """Cancel with docker run in flight: nothing may start over it, stop() drains it, and the
    container is removed exactly once -- by the thread that made it."""
    reset_limiter(monkeypatch)
    started, release = threading.Event(), threading.Event()
    docker = fake_docker(monkeypatch, started=started, release=release)
    dep = make_oci()

    async def _go():
        task = asyncio.create_task(dep.start(max_retries=1))
        while not started.is_set():
            await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert docker["rm"] == []  # the daemon still holds the container
        with pytest.raises(RuntimeError, match="stop\\(\\) must reclaim them first"):
            await dep.start()
        stopping = asyncio.create_task(dep.stop())
        await asyncio.sleep(0.05)
        assert not stopping.done()  # stop() waits for the call it cannot cancel
        release.set()
        await asyncio.wait_for(stopping, timeout=10)

    asyncio.run(asyncio.wait_for(_go(), timeout=20))
    assert docker["rm"] == [cid(1)]
    assert dep._owned is None
    assert dep._future is None
    assert any("abandoned" in m for m in dep.logger.messages("warning"))


def test_retry_slot(monkeypatch):
    """The CLI timeout outlives the cleanup timeout, so cleanup may not get the call back; the loop
    must then stop rather than submit over it."""
    reset_limiter(monkeypatch)
    fast_backoff(monkeypatch)
    monkeypatch.setattr(mod, "_STARTPAD", 0.1)
    monkeypatch.setattr(mod, "_CLEANWAIT", 0.2)
    release = threading.Event()
    docker = fake_docker(monkeypatch, release=release)
    dep = make_oci(startup_timeout=0.1)

    async def _go():
        with pytest.raises(RuntimeError, match="not retrying"):
            await dep.start(max_retries=2)
        assert docker["runs"] == 1  # the second attempt never launched a container
        held = dep._future
        assert held is not None and not held.done()

        stopping = asyncio.create_task(dep.stop())
        await asyncio.sleep(0.05)
        assert not stopping.done()  # stop() waits for the call the retry loop refused to abandon
        assert dep._future is held  # and it is still the same call
        release.set()
        await asyncio.wait_for(stopping, timeout=10)

    asyncio.run(asyncio.wait_for(_go(), timeout=20))
    assert docker["rm"] == [cid(1)]
    assert dep._future is None
    assert dep._owned is None


def test_rm_stops(monkeypatch):
    """A container cleanup could not remove stays owned, and no retry may start a second one over it."""
    reset_limiter(monkeypatch)
    fast_backoff(monkeypatch)
    runs = {"n": 0, "rm": 0}

    def fake_run(args, **kwargs):
        if args[1] == "run":
            runs["n"] += 1
            container_id = cid(runs["n"])
            Path(args[args.index("--cidfile") + 1]).write_text(f"{container_id}\n")
            return completed(0, stdout=f"{container_id}\n")
        if args[1] == "rm":
            runs["rm"] += 1
            return completed(1, stderr="daemon busy")  # fails fast, twice
        return completed(0)

    fast_rm(monkeypatch)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    dep = make_oci(sessioner=fail_start)

    with pytest.raises(RuntimeError, match="not retrying"):
        asyncio.run(dep.start(max_retries=2))

    assert runs["n"] == 1  # the second attempt never launched a container
    assert dep._owned == (cid(1), "uni-agent-test-run")
    assert dep._container_id == cid(1)  # the diagnostics pointer survives a failed rm
    dep._owned = None
    dep._container_id = None


def test_clean_drain(monkeypatch):
    """A cleanup cancelled mid-rm leaves that rm in the slot; stop() drains the same call and the
    container is never removed twice."""
    reset_limiter(monkeypatch)
    fast_backoff(monkeypatch)
    monkeypatch.setattr(mod, "_CLEANWAIT", 0.2)
    release = threading.Event()
    runs = {"n": 0, "rm": 0}

    def fake_run(args, **kwargs):
        if args[1] == "run":
            runs["n"] += 1
            container_id = cid(runs["n"])
            Path(args[args.index("--cidfile") + 1]).write_text(f"{container_id}\n")
            return completed(0, stdout=f"{container_id}\n")
        if args[1] == "rm":
            runs["rm"] += 1
            release.wait(timeout=10)  # the rm outlives the cleanup that started it
            return completed(0)
        return completed(0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    dep = make_oci(sessioner=fail_start)

    async def _go():
        with pytest.raises(RuntimeError, match="not retrying"):
            await dep.start(max_retries=2)
        assert runs["n"] == 1
        assert runs["rm"] == 1
        held = dep._future
        assert held is not None and not held.done()

        stopping = asyncio.create_task(dep.stop())
        await asyncio.sleep(0.05)
        assert dep._future is held  # stop() drains the very call the cleanup left behind
        release.set()
        await asyncio.wait_for(stopping, timeout=10)

    asyncio.run(asyncio.wait_for(_go(), timeout=20))
    assert runs["rm"] == 1  # never removed twice
    assert dep._owned is None
    assert dep._container_id is None
    assert dep._future is None


def test_drain_keep(monkeypatch):
    """A cancelled drain leaves the call in flight; the next stop() consumes it, error and all."""
    reset_limiter(monkeypatch)
    started, release = threading.Event(), threading.Event()
    unhandled = []
    fake_docker(monkeypatch, started=started, release=release, result=completed(1, stderr="daemon died"))
    dep = make_oci()

    async def _go():
        asyncio.get_running_loop().set_exception_handler(lambda loop, ctx: unhandled.append(ctx))
        task = asyncio.create_task(dep.start(max_retries=1))
        while not started.is_set():
            await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        stopping = asyncio.create_task(dep.stop())
        await asyncio.sleep(0.05)
        stopping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stopping
        assert dep._has_cli()  # the call runs on; a later stop() drains it
        assert dep._has_cleanup()

        release.set()
        await asyncio.wait_for(dep.stop(), timeout=10)
        gc.collect()  # force Future.__del__ while the loop can still report it
        await asyncio.sleep(0)

    asyncio.run(asyncio.wait_for(_go(), timeout=20))
    assert dep._future is None
    assert any("failed after its attempt ended" in m for m in dep.logger.messages("warning"))
    assert unhandled == []


def test_permit_bind(monkeypatch):
    """A cancelled attempt's permit is held until its CLI thread returns."""
    reset_limiter(monkeypatch, per_worker=1)
    started, release = threading.Event(), threading.Event()
    docker = fake_docker(monkeypatch, started=started, release=release)
    dep_a = make_oci()
    dep_b = make_oci()

    async def _go():
        task = asyncio.create_task(dep_a.start(max_retries=1))
        while not started.is_set():
            await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        second = asyncio.create_task(dep_b.start(max_retries=1))
        await asyncio.sleep(0.1)
        assert docker["runs"] == 1
        release.set()
        await asyncio.wait_for(second, timeout=10)
        assert docker["runs"] == 2
        await dep_a.stop()
        await dep_b.stop()

    asyncio.run(asyncio.wait_for(_go(), timeout=20))
    assert cid(1) in docker["rm"]
    assert dep_a._future is None and dep_b._future is None


def test_reap_retry(monkeypatch):
    """A failed late reap returns the id to the ledger for future sweeps."""
    fast_rm(monkeypatch)
    dep = make_plain()
    dep._abandoned = True

    def fake_run(args, **kwargs):
        if args[1] == "run":
            Path(args[args.index("--cidfile") + 1]).write_text(f"{cid(9)}\n")
            return completed(0, stdout=f"{cid(9)}\n")
        return completed(1, stderr="daemon busy")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    dep._run_container(["docker", "run", "-d", "img", "-lc", "cmd"], "n-x")
    assert any("stays owned and may leak" in m for m in dep.logger.messages("error"))
    assert dep._owned == (cid(9), "n-x")
    dep._owned = None
    dep._container_id = None


def test_cid_queued(monkeypatch):
    """A queued call the pool never dispatches must allocate nothing: no one is left to free it."""
    reset_limiter(monkeypatch)
    made = []
    real_mkdtemp = mod.tempfile.mkdtemp

    def spy(**kwargs):
        made.append(Path(real_mkdtemp(**kwargs)))
        return str(made[-1])

    class Stall:
        def __init__(self):
            self.n = 0

        def submit(self, call, *args):
            self.n += 1
            fut = concurrent.futures.Future()
            if self.n == 1:
                fut.set_result(call(*args))
            return fut

    pool = Stall()
    monkeypatch.setattr(mod.tempfile, "mkdtemp", spy)
    monkeypatch.setattr(mod, "_clipool", lambda: pool)
    dep = make_plain(image="img")

    async def _go():
        task = asyncio.create_task(dep._start_oci_container("tok", "n-x", 8123))
        while pool.n < 2:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(asyncio.wait_for(_go(), timeout=5))
    assert made == []


def test_cid_unread(monkeypatch):
    """An unreadable cidfile is surfaced and kept; the run's stdout still confirms the id."""
    reset_limiter(monkeypatch)
    kept = {}

    def fake_run(args, **kw):
        if args[1] == "run":
            # A directory where the id should be: read_text raises an OSError that is not "absent".
            kept["path"] = Path(args[args.index("--cidfile") + 1])
            kept["path"].mkdir()
            return completed(0, stdout=f"{cid(2)}\n")
        return completed(0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    dep = make_plain(image="img")

    try:
        asyncio.run(dep._start_oci_container("tok", "n-x", 8123))
        assert dep._owned == (cid(2), "n-x")  # stdout named it, so no barrier is needed
        assert dep._orphan is None
        assert kept["path"].exists()  # the file could not be removed either, and that is reported
        assert any("failed to remove cidfile" in m for m in dep.logger.messages("warning"))
    finally:
        shutil.rmtree(kept["path"].parent, ignore_errors=True)
    dep._owned = None
    dep._container_id = None
    dep._runtime = None


def test_inspect_gap(monkeypatch):
    """Cancellation during the inspect phase must not submit docker run."""
    reset_limiter(monkeypatch)
    started = threading.Event()
    release = threading.Event()
    calls = []

    def fake_run(args, **kw):
        calls.append(args[1])
        if args[1] == "inspect":
            started.set()
            release.wait(timeout=10)
            return completed(0, stdout="bridge\n")
        return completed(0, stdout="cid\n")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "_is_running_in_container", lambda: True)
    monkeypatch.setenv("HOSTNAME", "host0")
    dep = make_plain(image="img")

    async def _go():
        task = asyncio.create_task(dep._start_oci_container("tok", "n-x", 8123))
        while not started.is_set():
            await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await dep.stop()  # drains the build call the cancellation left in the slot

    asyncio.run(asyncio.wait_for(_go(), timeout=10))
    assert calls == ["inspect"]  # docker run was never submitted
    assert dep._future is None
    assert dep._owned is None


def test_cid_timeout(monkeypatch):
    """A CLI timeout after the daemon created the container must not orphan it."""
    reset_limiter(monkeypatch)
    seen = {}

    def fake_run(args, **kw):
        if args[1] == "run":
            seen["cid"] = Path(args[args.index("--cidfile") + 1])
            seen["cid"].write_text(f"{cid(1)}\n")
            raise subprocess.TimeoutExpired(cmd=args, timeout=600)
        return completed(0, stdout="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    dep = make_plain(image="img")

    with pytest.raises(mod.CliError, match="timed out"):
        asyncio.run(dep._start_oci_container("tok", "n-x", 8123))
    assert dep._owned == (cid(1), "n-x")  # recoverable: stop()/__del__ can still remove it
    assert not seen["cid"].parent.exists()
    dep._owned = None
    dep._container_id = None


def test_oci_inspect(monkeypatch):
    """Both CLI phases of a real OCI start run on the tracked pool."""
    reset_limiter(monkeypatch)
    calls = []
    seen = {}

    def fake_run(args, **kw):
        calls.append(args[1])
        if args[1] == "run":
            # Real docker confirms the id twice: in the cidfile and on stdout.
            seen["cid"] = Path(args[args.index("--cidfile") + 1])
            seen["cid"].write_text(f"{cid(3)}\n")
            return completed(0, stdout=f"{cid(3)}\n")
        return completed(0, stdout="172.17.0.2\n")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    dep = make_plain(network="bridge", image="img")

    asyncio.run(dep._start_oci_container("tok", "name-x", 8123))
    assert calls == ["run", "inspect"]
    assert dep._future is None  # every phase handed its result back
    assert dep._container_id == cid(3)
    assert dep._owned == (cid(3), "name-x")
    assert not seen["cid"].parent.exists()
    dep._owned = None
    dep._container_id = None
    dep._runtime = None


def test_start_owner(monkeypatch):
    reset_limiter(monkeypatch)
    fast_backoff(monkeypatch)
    docker = fake_docker(monkeypatch)
    tries = {"n": 0}

    async def fail_once(dep):
        tries["n"] += 1
        if tries["n"] == 1:
            raise RuntimeError("docker run failed")

    dep = make_oci(starter=fail_once)

    asyncio.run(dep.start())
    base = "uni-agent-test-run"
    assert dep._starts == 2
    assert docker["runs"] == 1
    assert dep._owned == (cid(1), f"{base}-r1")
    infos = dep.logger.messages("info")
    assert any("local sandbox alive in" in m and "(attempt 2)" in m and f"id={cid(1)[:12]}" in m for m in infos)

    asyncio.run(dep.stop())
    assert docker["rm"] == [cid(1)]
    assert dep._owned is None
    assert dep._container_id is None


def test_session_err(monkeypatch):
    """The training path calls start() with no argument: every attempt's container is swept."""
    reset_limiter(monkeypatch)
    fast_backoff(monkeypatch)
    docker = fake_docker(monkeypatch)
    dep = make_oci(sessioner=fail_start)

    with pytest.raises(RuntimeError, match=r"after 5 attempt"):
        asyncio.run(dep.start())
    assert any("bash session creation failed" in m for m in dep.logger.messages("error"))
    assert docker["rm"] == [cid(i) for i in range(1, 6)]
    assert dep._owned is None


_TOKEN = "s3kr1t-auth-token-value"


def _token_docker(mode, running, release):
    """A daemon that hands the token back where no `--auth-token <value>` pattern would find it."""

    def fake_run(args, **kwargs):
        if args[1] == "run":
            if mode == "timeout":
                assert _TOKEN in " ".join(args)  # else this branch would pass on an empty input
                Path(args[args.index("--cidfile") + 1]).write_text(f"{cid(1)}\n")
                raise subprocess.TimeoutExpired(cmd=args, timeout=600)
            if mode == "exit":
                # The real path runs with check=False: a failing daemon returns non-zero and echoes
                # back the command it could not run.
                return completed(125, stderr=f"OCI runtime failed: TOKEN={_TOKEN} start.sh")
            if mode == "cancel":
                running.set()
                release.wait(timeout=10)
                return completed(125, stderr=f"daemon died: https://host/start?token={_TOKEN}")
            Path(args[args.index("--cidfile") + 1]).write_text(f"{cid(1)}\n")
            return completed(0, stdout=f"{cid(1)}\n")
        if args[1] == "logs":
            return completed(0, stdout=f"swerex.server booting, TOKEN={_TOKEN}")
        return completed(0)

    return fake_run


def _token_dep(monkeypatch, mode, *, running, release):
    """A real LocalDeployment -- no double may short-circuit the container-log path under test."""
    reset_limiter(monkeypatch)
    monkeypatch.setattr(mod.LocalDeployment, "_get_token", lambda self: _TOKEN)
    monkeypatch.setattr(mod.subprocess, "run", _token_docker(mode, running, release))
    monkeypatch.setattr(mod.LocalRuntime, "from_config", lambda cfg, run_id=None: FakeRuntime())
    dep = make_plain(image="img")
    dep.logger = mod.get_logger("deployment", f"token-{mode}")

    async def unhealthy(timeout):
        # A deep cause carries the token; the wrapper that reaches the retry loop does not.
        try:
            raise ConnectionError(f"connect failed: TOKEN={_TOKEN}")
        except ConnectionError as exc:
            raise TimeoutError(f"runtime not alive within {timeout:.0f}s") from exc

    dep._wait_until_alive = unhealthy
    return dep


async def _drive(dep, mode, *, running, release):
    task = asyncio.create_task(dep.start(max_retries=1))
    if mode != "cancel":
        with pytest.raises(Exception) as excinfo:
            await task
        return excinfo.value
    while not running.is_set():  # the blocked docker run, not the build call before it
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await asyncio.wait_for(dep.stop(), timeout=10)
    return None


@pytest.mark.parametrize(
    ("mode", "wanted"),
    [
        ("exit", "TOKEN=***"),
        ("timeout", "timed out after"),
        ("unhealthy", "TOKEN=***"),
        ("cancel", "token=***"),
    ],
)
def test_no_token(monkeypatch, mode, wanted):
    """The token must not reach a record by any route: an exception's message, a deep cause in its
    chain, the container's own output, or a CLI error whose argv the exception carried.

    The sandbox command is a user template, so the token can sit anywhere in it -- these fakes put it
    where no `--auth-token <value>` pattern would find it.
    """
    running, release = threading.Event(), threading.Event()
    dep = _token_dep(monkeypatch, mode, running=running, release=release)

    captured = []
    sink = loguru_logger.add(captured.append, level="DEBUG")
    try:
        asyncio.run(asyncio.wait_for(_drive(dep, mode, running=running, release=release), timeout=20))
    finally:
        release.set()
        loguru_logger.remove(sink)

    text = "\n".join(str(m) for m in captured)
    record = "failed after its attempt ended" if mode == "cancel" else "failed to start local sandbox"
    assert record in text, f"the {mode} scenario never logged the record under test"
    assert wanted in text, f"the {mode} scenario dropped the output instead of redacting it"
    assert _TOKEN not in text
    # Cleanup reclaimed everything, so the tokens went with it.
    assert dep._orphan is None and dep._owned is None and dep._secrets == set()


def test_app_token():
    """The apptainer server's own argv holds the token, so any exception from it carries the command."""
    dep = make_plain(container_runtime="apptainer")
    dep.logger = mod.get_logger("deployment", "token-apptainer")
    dep._secrets.add(_TOKEN)
    argv = ["apptainer", "exec", "img", "sh", "-lc", f"server --auth-token {_TOKEN}"]

    def _wait(timeout=None):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout)

    dep._server_process = FakeProcess(wait=_wait)

    captured = []
    sink = loguru_logger.add(captured.append, level="DEBUG")
    try:
        dep._kill_server()
    finally:
        loguru_logger.remove(sink)

    text = "\n".join(str(m) for m in captured)
    assert "could not reap the apptainer server process" in text
    assert "--auth-token ***" in text  # redacted, not dropped
    assert _TOKEN not in text
    assert dep._server_process is not None  # a survivor is kept for the next stop()


def test_final_token(monkeypatch):
    """The error raised to the caller is redacted while the token is alive -- cleanup drops it."""
    dep = _token_dep(monkeypatch, "unhealthy", running=threading.Event(), release=threading.Event())

    async def echoing(timeout):
        raise RuntimeError(f"server echoed TOKEN={_TOKEN}")

    dep._wait_until_alive = echoing

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(dep.start(max_retries=1))

    assert dep._secrets == set()  # cleanup dropped them before this message was ever read
    assert "TOKEN=***" in str(excinfo.value)
    assert _TOKEN not in str(excinfo.value)


# How the run ended, what the tag query then said, and -- once the container is left unnamed -- what a
# later query says when the runtime can finally answer. "post-create" is the case that drives all of
# this: docker exits non-zero and deletes the cidfile when the *post-create* id write fails, so a
# failure exit proves nothing.
_CASES = {
    "post-create": {"code": 125, "ps": "one", "end": "claimed"},
    "name-clash": {"code": 125, "ps": "none", "end": "cleared"},
    "partial-cid": {"code": None, "ps": "one", "cid": "partial", "end": "claimed"},
    "failed-multi": {"code": 125, "ps": "two", "end": "barrier", "recover": "one"},
    "failed-junk": {"code": 125, "ps": "junk", "end": "barrier", "recover": "none"},
    "killed-empty": {"code": None, "ps": "none", "end": "barrier", "recover": "one"},
    "unknown-empty": {"code": "raise", "ps": "none", "end": "barrier", "recover": "one"},
    "query-error": {"code": None, "ps": "error", "end": "barrier", "recover": "one"},
}


def _fake_cli(spec, calls):
    """A container runtime that behaves as `spec` says, and holds the causality the recovery rests on:
    a tag query must ask for the owner of the run it follows, or its answer is about someone else."""

    def fake_run(args, **kwargs):
        if args[1] == "run":
            calls["run"] += 1
            calls["owners"].append(next(a for a in args if a.startswith(f"{mod._OWNERTAG}=")).split("=", 1)[1])
            if spec.get("cid") == "partial":
                Path(args[args.index("--cidfile") + 1]).write_text(cid(1)[:20])
            if spec["code"] == "raise":
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")  # post-spawn decode
            if spec["code"] is None:
                raise subprocess.TimeoutExpired(cmd=args, timeout=600)
            return completed(spec["code"], stderr="post-create failure" if spec["code"] else "")
        if args[1] == "ps":
            assert args[-1] == f"label={mod._OWNERTAG}={calls['owners'][-1]}"
            calls["asked"] += 1
            if spec["ps"] == "error":
                raise subprocess.TimeoutExpired(cmd=args, timeout=60)
            listing = {
                "one": f"{cid(calls['run'])}\n",
                "two": f"{cid(1)}\n{cid(2)}\n",
                "junk": "deadbeef\n",
                "none": "",
            }[spec["ps"]]
            return completed(0, stdout=listing)
        if args[1] == "rm":
            calls["rm"].append(args[3])
        return completed(0)

    return fake_run


def _tagged_dep(monkeypatch):
    """A real deployment whose start always fails after the run, so only the claim path is under test."""
    reset_limiter(monkeypatch)
    fast_backoff(monkeypatch)
    monkeypatch.setattr(mod.LocalRuntime, "from_config", lambda cfg, run_id=None: FakeRuntime())
    dep = make_plain(image="img")

    async def never_alive(timeout):
        raise TimeoutError("runtime not alive")

    dep._wait_until_alive = never_alive
    return dep


@pytest.mark.parametrize("case", sorted(_CASES))
def test_cid_owner(monkeypatch, case):
    """No exit code proves a container was not created. With no full id, the owner tag decides -- but
    only a runtime that ran, returned and reported failure makes an empty answer authoritative."""
    spec = dict(_CASES[case])
    calls = {"run": 0, "asked": 0, "rm": [], "owners": []}
    monkeypatch.setattr(mod.subprocess, "run", _fake_cli(spec, calls))
    dep = _tagged_dep(monkeypatch)

    with pytest.raises(RuntimeError):
        asyncio.run(dep.start(max_retries=2))

    if spec["end"] == "claimed":  # the tag found what no id could name: reclaimed by id, then retried
        assert calls["run"] == 2 and calls["asked"] == 2 and calls["rm"] == [cid(1), cid(2)]
    elif spec["end"] == "cleared":  # nothing of ours exists, so there is nothing to reclaim
        assert calls["run"] == 2 and calls["asked"] == 2 and calls["rm"] == []
    else:
        _barrier(dep, spec, calls)
    # A retry that reused the owner could be handed the container the attempt before it left behind.
    assert len(set(calls["owners"])) == calls["run"]
    assert dep._orphan is None and dep._owned is None
    assert not dep._has_cleanup() and dep._secrets == set()


def _barrier(dep, spec, calls):
    """No answer, so we know nothing: nothing may start over it, and the token lives until it lifts."""
    assert calls["run"] == 1 and calls["rm"] == []  # no second container over the one that may exist
    assert calls["asked"] == 2  # the claim asked, and so did the cleanup that could not lift it
    assert dep._orphan is not None and dep._has_cleanup()
    assert dep._secrets, "the sandbox may exist and may still be logged about; its token must live"
    with pytest.raises(RuntimeError, match="stop\\(\\) must reclaim them first"):
        asyncio.run(dep.start())

    spec["ps"] = spec["recover"]  # the runtime can answer now
    asyncio.run(dep.stop())
    assert calls["rm"] == ([cid(1)] if spec["recover"] == "one" else [])


def test_info_token(monkeypatch):
    """A start whose per-attempt announce raises before registration strands no token."""
    reset_limiter(monkeypatch)
    dep = make_plain(image="img")
    dep.logger.info = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("the sink is gone"))

    with pytest.raises(RuntimeError, match="the sink is gone"):
        asyncio.run(dep.start(max_retries=1))
    assert dep._secrets == set()
    assert not dep._has_cleanup()


def test_queue_token(monkeypatch):
    """A start cancelled while it queues for a permit made nothing at all, so the token it registered
    dies with it: no resource is left that a later stop() could log an exception carrying."""
    monkeypatch.setattr(mod, "_get_limiter", lambda: sem)
    sem = asyncio.Semaphore(1)
    dep = make_plain(image="img")

    async def _go():
        await sem.acquire()  # the only permit is taken, so the attempt never gets past the queue
        task = asyncio.create_task(dep.start(max_retries=1))
        while not dep._secrets:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(asyncio.wait_for(_go(), timeout=5))
    assert dep._secrets == set()
    assert not dep._has_cleanup()


def test_clean_token(monkeypatch):
    """A cancel landing on the last await of the attempt's own cleanup: what it made is reclaimed, so
    the cleanup's own release never runs -- and the token would outlive every resource it authorised."""
    reset_limiter(monkeypatch)
    reached = asyncio.Event()

    async def hang_cleanup(dep):
        reached.set()
        await asyncio.sleep(3600)

    dep = make_loop(starter=fail_start, stopper=hang_cleanup)

    async def _go():
        task = asyncio.create_task(dep.start(max_retries=2))
        await asyncio.wait_for(reached.wait(), timeout=5)
        assert dep._secrets  # the attempt registered one
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(asyncio.wait_for(_go(), timeout=10))
    assert not dep._has_cleanup()
    assert dep._secrets == set()


def test_cid_handles():
    """A --cidfile or same-name label in extra_run_args cannot override ours; ours go last and win."""
    extra = ["--cidfile", "/tmp/evil", "--label", f"{mod._OWNERTAG}=evil"]
    dep = make_plain(image="img", extra_run_args=extra)
    dep._get_current_container_network = lambda: None

    args = dep._with_cid(dep._build_run_command("n-x", 8000, "cmd"), Path("/tmp/ours"), "ours")

    last_cid = len(args) - 1 - args[::-1].index("--cidfile")
    assert args[last_cid + 1] == "/tmp/ours"
    labels = [args[i + 1] for i, arg in enumerate(args) if arg == "--label"]
    assert labels[-1] == f"{mod._OWNERTAG}=ours"
    assert args[-3:] == ["img", "-lc", "cmd"]


def test_dashdash():
    """Reject an ambiguous bare --."""
    dep = make_plain(image="img", extra_run_args=["--"])
    dep._get_current_container_network = lambda: None

    with pytest.raises(RuntimeError, match="bare '--'"):
        dep._with_cid(dep._build_run_command("n-x", 8000, "cmd"), Path("/tmp/ours"), "ours")


def test_log_token(monkeypatch, tmp_path):
    """A caller logging the error this class raises must print neither the token nor a raw cause."""
    running, release = threading.Event(), threading.Event()
    dep = _token_dep(monkeypatch, "unhealthy", running=running, release=release)

    log_file = tmp_path / "run.log"
    add_file_handler(log_file, "token-unhealthy")
    try:
        escaped = asyncio.run(asyncio.wait_for(_drive(dep, "unhealthy", running=running, release=release), timeout=20))
        assert escaped.__cause__ is None and escaped.__context__ is None
        mod.get_logger("deployment", "token-unhealthy").opt(
            exception=(type(escaped), escaped, escaped.__traceback__)
        ).error("caller sees: {}", escaped)
        loguru_logger.complete()  # the dispatch sink enqueues; wait for its writer thread
    finally:
        cleanup_handlers("token-unhealthy")

    written = log_file.read_text()
    assert "caller sees:" in written
    assert "Traceback" in written  # the sink really rendered it -- just without the frame locals
    assert "TOKEN=***" in written  # the deep cause reached the internal record, redacted
    assert _TOKEN not in written
