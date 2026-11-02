"""Local provider regressions: observe exit before any test-owned PID rescue."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from uni_agent.sandbox import local
from uni_agent.sandbox.local import LocalSandbox

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.level0,
    pytest.mark.asyncio,
    pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups and ps"),
]


_WORKER = """
import json, os, pathlib, subprocess, sys, time
directory = pathlib.Path(sys.argv[1])
role, mode = sys.argv[2:4]
record = dict(pid=os.getpid(), ppid=os.getppid(), pgid=os.getpgrp(), sid=os.getsid(0))
temporary = directory / (role + '.tmp')
temporary.write_text(json.dumps(record))
temporary.replace(directory / (role + '.json'))
if role == 'parent':
    child_role = 'detached' if mode == 'detached' else 'leaf'
    subprocess.Popen([sys.executable, __file__, str(directory), child_role, mode],
                     start_new_session=mode == 'detached')
    if mode == 'leader-exits':
        while not (directory / 'leaf.json').exists():
            time.sleep(0.01)
        sys.exit(0)
elif role == 'detached':
    subprocess.Popen([sys.executable, __file__, str(directory), 'leaf', mode], start_new_session=True)
time.sleep(60)
"""


def _worker(tmp_path: Path) -> Path:
    script = tmp_path / "worker.py"
    script.write_text(_WORKER)
    return script


def _running(pid: int) -> bool:
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False, timeout=2
    )
    assert not result.stderr.strip(), f"ps failed to observe PID {pid}: {result.stderr}"
    assert result.returncode in {0, 1}, f"ps failed with exit code {result.returncode}"
    status = result.stdout.strip()
    if not status:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        raise AssertionError(f"PID {pid} exists but ps did not return its state")
    return not status.startswith("Z")


async def _records(tmp_path: Path, roles: tuple[str, ...]) -> dict[str, dict[str, int]]:
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        if all((tmp_path / f"{role}.json").exists() for role in roles):
            return {role: json.loads((tmp_path / f"{role}.json").read_text()) for role in roles}
        await asyncio.sleep(0.02)
    raise AssertionError(f"fixture process records missing: {roles}")


async def _assert_stopped(pids: list[int]) -> None:
    deadline = asyncio.get_running_loop().time() + 3
    while True:
        alive = [pid for pid in pids if _running(pid)]
        if not alive:
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"still running before test rescue: {alive}")
        await asyncio.sleep(0.03)


async def _rescue(tmp_path: Path, task: asyncio.Task, sibling: asyncio.subprocess.Process | None = None) -> None:
    # This runs only after the pass/fail assertions. It cannot turn a leak into
    # a passing test; exact fixture PIDs are rescued to keep negative controls safe.
    if not task.done():
        task.cancel()
    for path in tmp_path.glob("*.json"):
        pid = json.loads(path.read_text())["pid"]
        if _running(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)
    if sibling is not None:
        if sibling.returncode is None:
            try:
                sibling.kill()
            except ProcessLookupError:
                pass
        await asyncio.wait_for(sibling.wait(), timeout=5)


@pytest.mark.parametrize("stop", ["timeout", "cancel"])
async def test_exec_stops_detached_tool_group_but_not_unrelated_sibling(tmp_path, stop):
    script = _worker(tmp_path)
    sibling = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        str(tmp_path),
        "sibling",
        "detached",
        start_new_session=True,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    task = asyncio.create_task(
        LocalSandbox().exec(
            [sys.executable, str(script), str(tmp_path), "parent", "detached"],
            timeout=2 if stop == "timeout" else 30,
        )
    )
    try:
        tree = await _records(tmp_path, ("parent", "detached", "leaf", "sibling"))
        parent, detached, leaf = (tree[role] for role in ("parent", "detached", "leaf"))
        assert parent["pid"] == parent["pgid"] == parent["sid"]
        assert detached["ppid"] == parent["pid"]
        assert detached["pid"] == detached["pgid"] == detached["sid"]
        assert detached["pgid"] != parent["pgid"]
        assert leaf["ppid"] == detached["pid"]
        assert leaf["pid"] == leaf["pgid"] == leaf["sid"]
        assert leaf["pgid"] != detached["pgid"]
        assert all(_running(record["pid"]) for record in tree.values())
        if stop == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
        else:
            result = await asyncio.wait_for(task, timeout=5)
            assert result.exit_code == -1
        await _assert_stopped([parent["pid"], detached["pid"], leaf["pid"]])
        assert _running(sibling.pid), "cleanup killed an unrelated sibling"
    finally:
        await _rescue(tmp_path, task, sibling)


@pytest.mark.parametrize("failure", ["timeout", "missing", "nonzero", "malformed"])
async def test_exec_snapshot_failure_still_kills_original_group(tmp_path, monkeypatch, caplog, failure):
    script = _worker(tmp_path)
    original_run = subprocess.run

    def fail_snapshot(argv, **kwargs):
        if argv[:3] == ["ps", "-A", "-o"]:
            if failure == "timeout":
                raise subprocess.TimeoutExpired(argv, timeout=1)
            if failure == "missing":
                raise FileNotFoundError("injected missing ps")
            if failure == "nonzero":
                raise subprocess.CalledProcessError(1, argv)
            return subprocess.CompletedProcess(argv, 0, "malformed process table")
        return original_run(argv, **kwargs)

    monkeypatch.setattr(local.subprocess, "run", fail_snapshot)
    task = asyncio.create_task(
        LocalSandbox().exec([sys.executable, str(script), str(tmp_path), "parent", "same-group"], timeout=30)
    )
    try:
        tree = await _records(tmp_path, ("parent", "leaf"))
        assert tree["parent"]["pgid"] == tree["leaf"]["pgid"]
        assert all(_running(record["pid"]) for record in tree.values())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        await _assert_stopped([record["pid"] for record in tree.values()])
        assert "descendant snapshot failed" in caplog.text
    finally:
        await _rescue(tmp_path, task)


async def test_exec_cancelled_during_spawn_stops_detached_groups_when_spawn_finishes(tmp_path, monkeypatch):
    script = _worker(tmp_path)
    release_spawn = asyncio.Event()
    original_create = asyncio.create_subprocess_exec
    processes = []

    async def delayed_create(*argv, **kwargs):
        process = await original_create(*argv, **kwargs)
        processes.append(process)
        await release_spawn.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_create)
    task = asyncio.create_task(
        LocalSandbox().exec([sys.executable, str(script), str(tmp_path), "parent", "detached"], timeout=30)
    )
    try:
        tree = await _records(tmp_path, ("parent", "detached", "leaf"))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        assert all(_running(record["pid"]) for record in tree.values())
        release_spawn.set()
        await _assert_stopped([record["pid"] for record in tree.values()])
        await asyncio.wait_for(processes[0].wait(), timeout=5)
    finally:
        release_spawn.set()
        await _rescue(tmp_path, task)
        for process in processes:
            await asyncio.wait_for(process.wait(), timeout=5)


async def test_exec_kills_group_after_its_leader_has_already_exited(tmp_path):
    script = _worker(tmp_path)
    task = asyncio.create_task(
        LocalSandbox().exec([sys.executable, str(script), str(tmp_path), "parent", "leader-exits"], timeout=2)
    )
    try:
        tree = await _records(tmp_path, ("parent", "leaf"))
        await _assert_stopped([tree["parent"]["pid"]])
        assert _running(tree["leaf"]["pid"]), "fixture child must hold pipes after its parent exits"
        assert tree["parent"]["pgid"] == tree["leaf"]["pgid"]
        result = await asyncio.wait_for(task, timeout=5)
        assert result.exit_code == -1
        await _assert_stopped([tree["leaf"]["pid"]])
    finally:
        await _rescue(tmp_path, task)


async def test_denied_group_does_not_hide_cancellation_or_skip_later_groups(tmp_path, monkeypatch, caplog):
    script = _worker(tmp_path)
    task = asyncio.create_task(
        LocalSandbox().exec([sys.executable, str(script), str(tmp_path), "parent", "detached"], timeout=30)
    )
    try:
        tree = await _records(tmp_path, ("parent", "detached", "leaf"))
        groups = sorted({tree[role]["pgid"] for role in ("detached", "leaf")})
        denied, later = groups
        original_killpg = os.killpg
        attempted = []

        def deny_one_group(group, sig):
            attempted.append(group)
            if group == denied:
                raise PermissionError("synthetic denied group")
            original_killpg(group, sig)

        monkeypatch.setattr(local.os, "killpg", deny_one_group)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        await _assert_stopped([record["pid"] for record in tree.values() if record["pgid"] != denied])
        assert all(_running(record["pid"]) for record in tree.values() if record["pgid"] == denied)
        assert attempted.index(denied) < attempted.index(later)
        assert f"could not terminate process group {denied}" in caplog.text
    finally:
        await _rescue(tmp_path, task)


@pytest.mark.parametrize("member_state", ["gone", "different-group", "denied", "surviving-member"])
async def test_extra_group_is_revalidated_using_all_observed_members(monkeypatch, member_state):
    # No real signals: simulate an extra group whose leader exited between the
    # snapshot and cleanup, with a surviving member only in the final case.
    parent = os.getpgrp() + 1000000
    leader, member = parent + 1, parent + 2
    snapshot = f"{parent} 1 {parent}\n{leader} {parent} {leader}\n{member} {leader} {leader}\n"
    monkeypatch.setattr(
        local.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, snapshot)
    )

    def current_group(pid):
        if pid == leader or member_state == "gone":
            raise ProcessLookupError
        if member_state == "denied":
            raise PermissionError
        return leader if member_state == "surviving-member" else leader + 100

    monkeypatch.setattr(local.os, "getpgid", current_group)
    signalled = []
    monkeypatch.setattr(local.os, "killpg", lambda group, sig: signalled.append((group, sig)))
    local._kill_command(SimpleNamespace(pid=parent, returncode=None))
    expected = [parent, leader] if member_state == "surviving-member" else [parent]
    assert signalled == [(group, signal.SIGKILL) for group in expected]


@pytest.mark.parametrize("count", [8, 32])
@pytest.mark.parametrize("stop", ["cancel", "spawn-cancel", "timeout"])
async def test_exec_concurrent_detached_trees_exit_before_rescue(tmp_path, monkeypatch, count, stop):
    """Start all descendants before triggering a burst; never count rescue as success."""
    script = _worker(tmp_path)
    gate = asyncio.Event()
    original_create = asyncio.create_subprocess_exec
    spawned = []

    async def delayed_create(*args, **kwargs):
        proc = await original_create(*args, **kwargs)
        spawned.append(proc)
        if stop == "spawn-cancel":
            await gate.wait()
        return proc

    sibling = await original_create(
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        start_new_session=True,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_create)
    directories = [tmp_path / str(i) for i in range(count)]
    tasks = []
    # wait_for's timer starts only after this gate: every timeout tests a live tree.
    communicate_gate = asyncio.Event()
    if stop == "timeout":
        # A controlled wait_for wrapper arms each timeout only after fixture readiness.
        original_wait_for = local.asyncio.wait_for

        async def ready_wait_for(awaitable, timeout):
            if timeout == 0.15:
                await communicate_gate.wait()
            return await original_wait_for(awaitable, timeout)

        monkeypatch.setattr(local.asyncio, "wait_for", ready_wait_for)
    try:
        for directory in directories:
            directory.mkdir()
            tasks.append(
                asyncio.create_task(
                    LocalSandbox().exec(
                        [sys.executable, str(script), str(directory), "parent", "detached"],
                        timeout=0.15 if stop == "timeout" else 30,
                    )
                )
            )
        trees = await asyncio.gather(*(_records(d, ("parent", "detached", "leaf")) for d in directories))
        pids = [record["pid"] for tree in trees for record in tree.values()]
        assert all(_running(pid) for pid in pids)
        assert _running(sibling.pid)
        if stop == "timeout":
            communicate_gate.set()
        else:
            for task in tasks:
                task.cancel()
                task.cancel()  # repeated cancellation must still trigger cleanup
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 15)
        if stop == "timeout":
            assert all(getattr(result, "exit_code", None) == -1 for result in results)
        else:
            assert all(isinstance(result, asyncio.CancelledError) for result in results)
        gate.set()
        await _assert_stopped(pids)
        assert _running(sibling.pid), "burst cleanup killed an unrelated sibling"
    finally:
        gate.set()
        communicate_gate.set()
        for directory, task in zip(directories, tasks, strict=False):
            await _rescue(directory, task)
        if sibling.returncode is None:
            sibling.kill()
        await asyncio.wait_for(sibling.wait(), 5)
        for proc in spawned:
            await asyncio.wait_for(proc.wait(), 5)


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
@pytest.mark.parametrize("repeat_cancel", [False, True])
async def test_slow_snapshot_keeps_loop_responsive_and_cleanup_owned(tmp_path, monkeypatch, stop, repeat_cancel):
    """The caller must retain cleanup ownership while unrelated loop work proceeds."""
    import threading

    entered, release = threading.Event(), threading.Event()
    original_run = subprocess.run

    def gated_snapshot(argv, **kwargs):
        if argv == ["ps", "-A", "-o", "pid=,ppid=,pgid="]:
            entered.set()
            release.wait(2)  # finite negative control on the old synchronous implementation
        return original_run(argv, **kwargs)

    monkeypatch.setattr(local.subprocess, "run", gated_snapshot)
    script = _worker(tmp_path)
    task = asyncio.create_task(
        LocalSandbox().exec(
            [sys.executable, str(script), str(tmp_path), "parent", "detached"],
            timeout=0.5 if stop == "timeout" else 30,
        )
    )
    try:
        tree = await _records(tmp_path, ("parent", "detached", "leaf"))
        assert all(_running(record["pid"]) for record in tree.values())
        if stop == "cancel":
            task.cancel()
        deadline = asyncio.get_running_loop().time() + 5
        while not entered.is_set():
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)
        assert not task.done(), "cleanup blocked the loop until the snapshot gate expired"
        if repeat_cancel:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
        other = await asyncio.wait_for(LocalSandbox().exec([sys.executable, "-c", "print('responsive')"]), 1)
        assert other.stdout.strip() == "responsive"
        assert not task.done(), "repeated cancellation abandoned unfinished cleanup"
        release.set()
        if stop == "cancel" or repeat_cancel:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
        else:
            assert (await asyncio.wait_for(task, 5)).exit_code == -1
        await _assert_stopped([record["pid"] for record in tree.values()])
    finally:
        release.set()
        await _rescue(tmp_path, task)


async def test_deferred_spawn_cleanup_is_submitted_before_asyncio_shutdown(tmp_path):
    # A separate asyncio.run exercises shutdown; a pytest loop cannot test its own
    # shutdown. No actual signals: the worker records completion after a delay.
    marker = tmp_path / "completed"
    script = """
import asyncio, pathlib, time
from types import SimpleNamespace
from uni_agent.sandbox import local

def cleanup(proc):
    time.sleep(0.05)
    pathlib.Path(proc.marker).write_text('completed')

local._kill_command = cleanup
async def main():
    spawn = asyncio.get_running_loop().create_future()
    spawn.set_result(SimpleNamespace(marker=MARKER))
    local.LocalSandbox()._kill_spawned(spawn)
    # Return immediately, without yielding to any newly created cleanup Task.
asyncio.run(main())
""".replace("MARKER", repr(str(marker)))
    result = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await asyncio.wait_for(result.communicate(), 5)
    assert result.returncode == 0, (stdout, stderr)
    assert marker.read_text() == "completed"


async def test_stop_owns_delayed_spawn_and_survives_repeated_cancellation(tmp_path, monkeypatch):
    import threading

    sandbox, unrelated = LocalSandbox(), LocalSandbox()
    spawn_ready = asyncio.Event()
    snapshot_entered, release_snapshot = threading.Event(), threading.Event()
    original_create, original_kill = asyncio.create_subprocess_exec, local._kill_command

    async def delayed_create(*args, **kwargs):
        proc = await original_create(*args, **kwargs)
        await spawn_ready.wait()
        return proc

    def delayed_kill(proc):
        snapshot_entered.set()
        assert release_snapshot.wait(5)
        original_kill(proc)

    monkeypatch.setattr(local.asyncio, "create_subprocess_exec", delayed_create)
    monkeypatch.setattr(local, "_kill_command", delayed_kill)
    task = asyncio.create_task(
        sandbox.exec([sys.executable, str(_worker(tmp_path)), str(tmp_path), "parent", "detached"])
    )
    stop = None
    try:
        tree = await _records(tmp_path, ("parent", "detached", "leaf"))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        stop = asyncio.create_task(sandbox.stop())
        await asyncio.sleep(0)
        await asyncio.wait_for(unrelated.stop(), 0.5)
        assert not stop.done(), "stop discarded this instance's unresolved spawn"
        spawn_ready.set()
        deadline = asyncio.get_running_loop().time() + 3
        while not snapshot_entered.is_set():
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)
        for _ in range(2):
            stop.cancel()
            await asyncio.sleep(0)
        assert not stop.done(), "cancellation abandoned the owned drain"
        release_snapshot.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(stop, 5)
        await _assert_stopped([r["pid"] for r in tree.values()])
        assert not sandbox._pending_spawns and not sandbox._pending_drains
        await sandbox.stop()  # teardown is idempotent
    finally:
        spawn_ready.set()
        release_snapshot.set()
        await _rescue(tmp_path, task)
        if stop is not None:
            await asyncio.gather(stop, return_exceptions=True)
        await sandbox.stop()


async def test_stop_bounds_pipe_drain_when_reparented_child_escapes(tmp_path, caplog):
    # The fixture intentionally leaves the PPID tree, outside best-effort cleanup.
    marker = tmp_path / "escaped.pid"
    orphan = (
        "import os,pathlib,sys,time; parent=os.getpid(); child=os.fork(); "
        "os._exit(0) if child else None; "
        "exec('while os.getppid() == parent: time.sleep(0.01)'); "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
    )
    parent = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]], start_new_session=True).wait(); "
        "time.sleep(60)"
    )
    sandbox = LocalSandbox()
    task = asyncio.create_task(sandbox.exec([sys.executable, "-c", parent, orphan, str(marker)]))
    pid = None
    try:
        deadline = asyncio.get_running_loop().time() + 5
        while not marker.exists() or not marker.read_text():
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.02)
        pid = int(marker.read_text())
        assert _running(pid)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(sandbox.stop(), 4)
        assert _running(pid), "fixture must actually escape ancestry-based cleanup"
        assert "pipe drain timed out" in caplog.text
        assert not sandbox._pending_drains
    finally:
        if pid is not None and _running(pid):
            os.kill(pid, signal.SIGKILL)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await sandbox.stop()


async def test_drain_yields_to_callbacks_when_gather_is_already_complete():
    # A separate process bounds the negative control: a tight loop cannot be
    # interrupted by an asyncio timeout running on that same blocked loop.
    script = """
import asyncio
from uni_agent.sandbox.local import LocalSandbox
async def main():
    sandbox = LocalSandbox()
    completed = asyncio.get_running_loop().create_future()
    completed.set_result(None)
    sandbox._pending_drains.add(completed)
    completed.add_done_callback(sandbox._pending_drains.discard)
    original = asyncio.gather
    def eager(*args, **kwargs):
        if all(f.done() for f in args):
            result = asyncio.get_running_loop().create_future()
            result.set_result([f.result() for f in args])
            return result
        return original(*args, **kwargs)
    asyncio.gather = eager
    await sandbox._drain_pending()
    assert not sandbox._pending_drains
asyncio.run(main())
"""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), 3)
        assert proc.returncode == 0 and not stderr, (stdout, stderr)
    finally:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
