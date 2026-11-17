from __future__ import annotations

import asyncio
import sys

import pytest

from uni_agent.sandbox.base import ExecResult, _stop_after_failed_start
from uni_agent.sandbox.docker import DockerSandbox


def ok(stdout=""):
    return ExecResult(0, stdout, "")


@pytest.mark.parametrize("interrupt", ["timeout", "cancel"])
@pytest.mark.cpu
@pytest.mark.level0
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
@pytest.mark.cpu
@pytest.mark.level0
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
            assert [call[0] for call in calls] == ["container", "rm", "image", "run"]
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
@pytest.mark.cpu
@pytest.mark.level0
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


@pytest.mark.cpu
@pytest.mark.level0
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
@pytest.mark.cpu
@pytest.mark.level0
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
@pytest.mark.cpu
@pytest.mark.level0
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


@pytest.mark.cpu
@pytest.mark.level0
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


@pytest.mark.cpu
@pytest.mark.level0
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


@pytest.mark.cpu
@pytest.mark.level0
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


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.TimeoutError, asyncio.CancelledError])
@pytest.mark.cpu
@pytest.mark.level0
def test_failed_executable_path_setup_cleans_started_container(monkeypatch, failure):
    sandbox = DockerSandbox()
    calls = []

    async def docker(*args, **kwargs):
        calls.append(args)
        return ok("owned-id\n" if args[0] == "container" else "")

    async def setup(paths):
        assert sandbox._container_name is not None
        raise failure("setup interrupted")

    monkeypatch.setattr(sandbox, "_run_docker", docker)
    monkeypatch.setattr(sandbox, "_setup_executable_paths", setup)
    # Python 3.10's asyncio.run may discard the cancellation message.
    message = None if failure is asyncio.CancelledError else "setup interrupted"
    with pytest.raises(failure, match=message):
        asyncio.run(sandbox.start())
    assert calls[-1] == ("rm", "-f", "owned-id")
    assert sandbox._container_name is None
    assert sandbox._cleanup_label is None


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.TimeoutError, asyncio.CancelledError])
@pytest.mark.parametrize("cleanup_failure", ["nonzero", "timeout"])
@pytest.mark.cpu
@pytest.mark.level0
def test_start_preserves_original_error_when_cleanup_fails(monkeypatch, caplog, failure, cleanup_failure):
    sandbox = DockerSandbox()
    original = failure("original startup failure")

    async def docker(*args, **kwargs):
        if args[0] == "container":
            return ok("owned-id\n")
        if args[0] == "rm":
            if cleanup_failure == "timeout":
                raise asyncio.TimeoutError
            return ExecResult(1, "", "daemon unavailable")
        return ok()

    async def setup(paths):
        raise original

    monkeypatch.setattr(sandbox, "_run_docker", docker)
    monkeypatch.setattr(sandbox, "_setup_executable_paths", setup)

    async def run():
        with pytest.raises(failure) as exc:
            await sandbox.start()
        assert exc.value is original
        assert sandbox._container_name is None
        assert sandbox._cleanup_label is not None

    asyncio.run(run())
    assert "cleanup failed after startup failure" in caplog.text


@pytest.mark.cpu
@pytest.mark.level0
def test_cancelled_start_is_not_retried_when_cleanup_fails(monkeypatch):
    sandbox = DockerSandbox()
    attempts = 0

    async def docker(*args, **kwargs):
        nonlocal attempts
        if args[0] == "run":
            attempts += 1
            raise asyncio.CancelledError
        if args[0] == "container":
            return ok("owned-id\n")
        if args[0] == "rm":
            return ExecResult(1, "", "daemon unavailable")
        return ok()

    monkeypatch.setattr(sandbox, "_run_docker", docker)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(sandbox.__aenter__(retry=2))
    assert attempts == 1
    assert sandbox._cleanup_label is not None


@pytest.mark.parametrize("caller", ["direct", "cancelled", "failed_start_cleanup"])
@pytest.mark.cpu
@pytest.mark.level0
def test_stop_enforces_configured_deadline_inside_cancellation_shield(monkeypatch, caller):
    monkeypatch.setenv("SANDBOX_STOP_TIMEOUT", "0.05")
    sandbox = DockerSandbox()
    sandbox._container_name = "agent-test"
    sandbox._cleanup_label = "uni-agent.sandbox=test"

    async def run():
        removing = asyncio.Event()
        release = asyncio.Event()
        removal_cancelled = asyncio.Event()

        async def docker(*args, **kwargs):
            if args[0] == "container":
                return ok("owned-id\n")
            removing.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                removal_cancelled.set()
                raise
            return ok()

        monkeypatch.setattr(sandbox, "_run_docker", docker)
        task = asyncio.create_task(
            _stop_after_failed_start(sandbox) if caller == "failed_start_cleanup" else sandbox.stop()
        )
        try:
            await removing.wait()
            if caller == "cancelled":
                for _ in range(2):
                    task.cancel()
                    await asyncio.sleep(0)
            # A watchdog makes the regression fail without leaving a shielded
            # cleanup task hung when tested against the unfixed implementation.
            done, _ = await asyncio.wait({task}, timeout=1)
            assert task in done, "cleanup ignored SANDBOX_STOP_TIMEOUT"
            if caller == "failed_start_cleanup":
                await task
            else:
                with pytest.raises(RuntimeError, match="cleanup timed out"):
                    await task
            assert removal_cancelled.is_set()
            assert sandbox._container_name is None
            assert sandbox._cleanup_label == "uni-agent.sandbox=test"
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
        await sandbox.stop()
        assert sandbox._cleanup_label is None

    asyncio.run(run())


@pytest.mark.parametrize("timeout", ["0", "-1"])
@pytest.mark.cpu
@pytest.mark.level0
def test_nonpositive_stop_timeout_disables_overall_deadline(monkeypatch, timeout):
    monkeypatch.setenv("SANDBOX_STOP_TIMEOUT", timeout)
    sandbox = DockerSandbox()
    sandbox._cleanup_label = "uni-agent.sandbox=test"

    async def run():
        removing = asyncio.Event()
        release = asyncio.Event()

        async def docker(*args, **kwargs):
            if args[0] == "container":
                return ok("owned-id\n")
            removing.set()
            await release.wait()
            return ok()

        monkeypatch.setattr(sandbox, "_run_docker", docker)
        task = asyncio.create_task(sandbox.stop())
        try:
            await removing.wait()
            done, _ = await asyncio.wait({task}, timeout=0.02)
            assert not done
        finally:
            release.set()
            await task
        assert sandbox._cleanup_label is None

    asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.level0
def test_old_stop_does_not_clear_restarted_container_ownership(monkeypatch):
    sandbox = DockerSandbox()
    sandbox._container_name = "old-container"
    old_owner = "uni-agent.sandbox=old"
    sandbox._cleanup_label = old_owner

    async def run():
        removing = asyncio.Event()
        release = asyncio.Event()

        async def docker(*args, **kwargs):
            if args[0] == "container":
                assert args[-1] == f"label={old_owner}"
                return ok("old-id\n")
            if args[0] == "rm" and not removing.is_set():
                removing.set()
                await release.wait()
            return ok()

        monkeypatch.setattr(sandbox, "_run_docker", docker)
        old_stop = asyncio.create_task(sandbox.stop())
        try:
            await removing.wait()
            await sandbox.stop()
            await sandbox.start()
            new_owner = sandbox._cleanup_label
            new_name = sandbox._container_name
            assert new_owner is not None and new_owner != old_owner
        finally:
            release.set()
            await old_stop
        assert sandbox._cleanup_label == new_owner
        assert sandbox._container_name == new_name

    asyncio.run(run())
