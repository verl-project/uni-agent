from __future__ import annotations

import asyncio
import sys

import pytest

from uni_agent.sandbox.base import ExecResult
from uni_agent.sandbox.docker import DockerSandbox

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def ok(stdout=""):
    return ExecResult(0, stdout, "")


@pytest.mark.parametrize("interrupt", ["timeout", "cancel"])
def test_docker_cli_is_killed_and_reaped_on_interrupt(monkeypatch, interrupt):
    """Exercise a real child process, without requiring a Docker daemon."""
    sandbox = DockerSandbox(docker_binary=sys.executable)
    spawn = asyncio.create_subprocess_exec
    processes = []

    async def run():
        spawned = asyncio.Event()

        async def capture_spawn(*args, **kwargs):
            process = await spawn(*args, **kwargs)
            processes.append(process)
            spawned.set()
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", capture_spawn)
        task = asyncio.create_task(
            sandbox._run_docker("-c", "import time; time.sleep(60)", timeout=0.1 if interrupt == "timeout" else None)
        )
        await spawned.wait()
        if interrupt == "cancel":
            task.cancel()
        try:
            with pytest.raises(asyncio.TimeoutError if interrupt == "timeout" else asyncio.CancelledError):
                await task
            assert processes[0].returncode is not None
        finally:
            # Also clean up when reproducing the regression on an unfixed checkout.
            if processes[0].returncode is None:
                processes[0].kill()
                await processes[0].communicate()

    asyncio.run(run())


@pytest.mark.parametrize("cleanup_recovers", [True, False])
def test_restart_retries_pending_cleanup_before_allocating_container(monkeypatch, cleanup_recovers):
    sandbox = DockerSandbox()
    calls = []
    removal_fails = True

    async def fake_run(*args, timeout=None):
        calls.append(args)
        if args[0] == "container":
            return ok("owned-id\n")
        if args[0] == "rm" and removal_fails:
            return ExecResult(1, "", "daemon unavailable")
        return ok()

    monkeypatch.setattr(sandbox, "_run_docker", fake_run)

    async def run():
        nonlocal removal_fails
        await sandbox.start()
        previous_owner = sandbox._cleanup_label
        with pytest.raises(RuntimeError, match="daemon unavailable"):
            await sandbox.stop()
        assert sandbox._container_name is None
        assert sandbox._cleanup_label == previous_owner

        calls.clear()
        removal_fails = not cleanup_recovers
        if cleanup_recovers:
            await sandbox.start()
            assert [call[0] for call in calls] == ["container", "rm", "run"]
            assert sandbox._container_name is not None
            assert sandbox._cleanup_label is not None
            assert sandbox._cleanup_label != previous_owner
        else:
            with pytest.raises(RuntimeError, match="daemon unavailable"):
                await sandbox.start()
            assert [call[0] for call in calls] == ["container", "rm"]
            assert sandbox._container_name is None
            assert sandbox._cleanup_label == previous_owner
        assert calls[0][-1] == f"label={previous_owner}"
        assert calls[1] == ("rm", "-f", "owned-id")

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["cancel", "outer_timeout", "nonzero"])
def test_interrupted_start_removes_only_owned_container(monkeypatch, failure):
    sandbox = DockerSandbox(container_name="possibly-shared-name")
    calls = []

    async def run():
        starting = asyncio.Event()

        async def fake_run(*args, timeout=None):
            calls.append(args)
            if args[0] == "run":
                starting.set()
                if failure == "nonzero":
                    return ExecResult(125, "", "start failed after create")
                await asyncio.Future()
            if args[0] == "container":
                assert args[-1] == f"label={sandbox._cleanup_label}"
                return ok("owned-container-id\n")
            return ok()

        monkeypatch.setattr(sandbox, "_run_docker", fake_run)
        if failure == "outer_timeout":
            monkeypatch.setenv("SANDBOX_STARTUP_TIMEOUT", "0.01")
            with pytest.raises(TimeoutError, match="SANDBOX_STARTUP_TIMEOUT"):
                await sandbox.__aenter__(retry=1)
        elif failure == "nonzero":
            with pytest.raises(RuntimeError, match="start failed after create"):
                await sandbox.start()
        else:
            task = asyncio.create_task(sandbox.start())
            await starting.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert ("rm", "-f", "owned-container-id") in calls
        assert all(call != ("rm", "-f", "possibly-shared-name") for call in calls)
        assert sandbox._container_name is None
        assert sandbox._cleanup_label is None

    asyncio.run(run())


def test_failed_start_does_not_remove_an_existing_same_name_container(monkeypatch):
    sandbox = DockerSandbox(container_name="other-users-container")
    calls = []

    async def fake_run(*args, timeout=None):
        calls.append(args)
        if args[0] == "run":
            return ExecResult(125, "", "container name is already in use")
        return ok()  # The existing container does not have this attempt's label.

    monkeypatch.setattr(sandbox, "_run_docker", fake_run)
    with pytest.raises(RuntimeError, match="already in use"):
        asyncio.run(sandbox.start())
    assert not any(call[0] == "rm" for call in calls)


@pytest.mark.parametrize("stage", ["container", "rm"])
def test_failed_cleanup_is_reported_and_can_be_retried(monkeypatch, stage):
    sandbox = DockerSandbox()
    sandbox._container_name = "agent-test"
    sandbox._cleanup_label = "uni-agent.sandbox=test"
    fail = True
    calls = []

    async def fake_run(*args, timeout=None):
        calls.append(args)
        assert timeout == 30.0
        if fail and args[0] == stage:
            return ExecResult(1, "", "daemon unavailable")
        return ok("owned-id\n" if args[0] == "container" else "")

    monkeypatch.setattr(sandbox, "_run_docker", fake_run)

    async def run():
        nonlocal fail
        with pytest.raises(RuntimeError, match="daemon unavailable"):
            await sandbox.stop()
        assert sandbox._cleanup_label == "uni-agent.sandbox=test"
        with pytest.raises(RuntimeError, match="not started"):
            sandbox._require_container()
        fail = False
        await sandbox.stop()
        assert sandbox._cleanup_label is None
        assert calls[-1] == ("rm", "-f", "owned-id")
        count = len(calls)
        await sandbox.stop()
        assert len(calls) == count

    asyncio.run(run())


@pytest.mark.parametrize("interrupt", ["timeout", "cancel"])
def test_exec_interrupt_retires_container_before_returning(monkeypatch, interrupt):
    sandbox = DockerSandbox()
    sandbox._container_name = "agent-test"
    sandbox._cleanup_label = "uni-agent.sandbox=test"
    calls = []

    async def fake_run(*args, timeout=None):
        calls.append(args)
        if args[0] == "exec":
            raise asyncio.TimeoutError if interrupt == "timeout" else asyncio.CancelledError
        return ok("owned-id\n" if args[0] == "container" else "")

    monkeypatch.setattr(sandbox, "_run_docker", fake_run)

    async def run():
        if interrupt == "timeout":
            result = await sandbox.exec(["sleep", "60"], timeout=0.1)
            assert result.exit_code == -1
            assert "timed out" in result.stderr
        else:
            with pytest.raises(asyncio.CancelledError):
                await sandbox.exec(["sleep", "60"])
        assert calls[-1] == ("rm", "-f", "owned-id")
        assert not await sandbox.is_alive()
        with pytest.raises(RuntimeError, match="not started"):
            await sandbox.exec(["echo", "must not run"])

    asyncio.run(run())


def test_nonzero_command_does_not_retire_container(monkeypatch):
    sandbox = DockerSandbox()
    sandbox._container_name = "agent-test"

    async def fake_run(*args, **kwargs):
        assert args[0] == "exec"
        return ExecResult(1, "", "test failure")

    monkeypatch.setattr(sandbox, "_run_docker", fake_run)
    result = asyncio.run(sandbox.exec(["false"]))
    assert result.exit_code == 1
    assert sandbox._container_name == "agent-test"


def test_cleanup_timeout_is_not_hidden_as_a_command_timeout(monkeypatch):
    sandbox = DockerSandbox()
    sandbox._container_name = "agent-test"
    sandbox._cleanup_label = "uni-agent.sandbox=test"

    async def fake_run(*args, **kwargs):
        raise asyncio.TimeoutError

    monkeypatch.setattr(sandbox, "_run_docker", fake_run)
    with pytest.raises(RuntimeError, match="cleanup timed out"):
        asyncio.run(sandbox.exec(["sleep", "60"], timeout=0.1))
    assert sandbox._cleanup_label == "uni-agent.sandbox=test"


def test_repeated_cancellation_waits_for_removal(monkeypatch):
    sandbox = DockerSandbox()
    sandbox._container_name = "agent-test"
    sandbox._cleanup_label = "uni-agent.sandbox=test"

    async def run():
        removing = asyncio.Event()
        release = asyncio.Event()

        async def fake_run(*args, **kwargs):
            if args[0] == "container":
                return ok("owned-id\n")
            removing.set()
            await release.wait()
            return ok()

        monkeypatch.setattr(sandbox, "_run_docker", fake_run)
        task = asyncio.create_task(sandbox.stop())
        await removing.wait()
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sandbox._cleanup_label is None

    asyncio.run(run())
