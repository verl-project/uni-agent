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


async def test_exec_snapshot_failure_still_kills_original_group(tmp_path, monkeypatch, caplog):
    script = _worker(tmp_path)
    original_run = subprocess.run

    def fail_snapshot(argv, **kwargs):
        if argv[:3] == ["ps", "-A", "-o"]:
            raise subprocess.TimeoutExpired(argv, timeout=1)
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
