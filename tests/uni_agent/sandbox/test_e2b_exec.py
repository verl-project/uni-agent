"""E2B command execution: session wrapper and process-group cleanup on timeout or cancellation."""

import asyncio
import subprocess
import sys
from types import SimpleNamespace

import pytest


class _CommandExitException(Exception):
    pass


class TimeoutException(Exception):
    """Same class name as the E2B SDK timeout, which ``_is_timeout_error`` matches by name."""


class _Handle:
    def __init__(self, pid, wait):
        self.pid = pid
        self._wait = wait
        self.killed = False

    async def wait(self):
        return await self._wait()

    async def kill(self):
        self.killed = True
        return True


class _Commands:
    def __init__(self, wait, kill_run=None):
        self.calls = []
        self.handle = _Handle(4242, wait)
        self._kill_run = kill_run

    async def run(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if kwargs.get("background"):
            return self.handle
        if self._kill_run is not None:
            return await self._kill_run()
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    def kill_calls(self):
        return [(cmd, kw) for cmd, kw in self.calls if not kw.get("background")]


def _sandbox(monkeypatch, commands):
    from uni_agent.sandbox.e2b import E2BSandbox

    monkeypatch.setitem(sys.modules, "e2b", SimpleNamespace(CommandExitException=_CommandExitException))
    sandbox = E2BSandbox(template="ready-template", user="user")
    sandbox._sandbox = SimpleNamespace(commands=commands)
    sandbox.sandbox_id = "sb-1"
    return sandbox


@pytest.mark.cpu
@pytest.mark.level0
def test_session_command_quotes_argv_and_falls_back_without_setsid():
    from uni_agent.sandbox.e2b import _session_command

    argv = ["printf", "%s|", "a b", "c'd", "$HOME", ""]
    command = _session_command(argv)

    assert command == (
        "if command -v setsid >/dev/null 2>&1; "
        "then exec setsid bash -c 'exec \"$@\"' bash printf '%s|' 'a b' 'c'\"'\"'d' '$HOME' ''; "
        "else exec printf '%s|' 'a b' 'c'\"'\"'d' '$HOME' ''; fi"
    )
    # The wrapper must not change what runs, whichever branch this host takes.
    out = subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=True).stdout
    assert out == "a b|c'd|$HOME||"


@pytest.mark.cpu
@pytest.mark.level0
def test_session_command_runs_scripts_without_a_shebang_under_bash(tmp_path):
    from uni_agent.sandbox.e2b import _session_command

    script = tmp_path / "noshebang"
    script.write_text('arr=(alpha beta)\necho "${arr[1]}"\n')
    script.chmod(0o755)

    # With setsid present, execvp alone would hand this script to /bin/sh (dash on
    # Debian/Ubuntu), which rejects the Bash array.
    result = subprocess.run(
        ["bash", "-c", _session_command(["./noshebang"])], cwd=tmp_path, capture_output=True, text=True
    )

    assert (result.returncode, result.stdout) == (0, "beta\n"), result.stderr


@pytest.mark.cpu
@pytest.mark.level0
def test_timeout_kills_the_process_group_as_root_then_the_handle(monkeypatch):
    async def wait():
        raise TimeoutException("deadline exceeded")

    commands = _Commands(wait)
    sandbox = _sandbox(monkeypatch, commands)

    result = asyncio.run(sandbox.exec(["bash", "-c", "sleep 1000 | cat"], timeout=3))

    assert result.exit_code == -1
    kills = [(cmd, kw["user"]) for cmd, kw in commands.kill_calls()]
    assert kills == [("kill -KILL -- -4242 2>/dev/null || true", "root")]
    assert commands.handle.killed


@pytest.mark.cpu
@pytest.mark.level0
def test_cancellation_kills_the_process_group(monkeypatch):
    started = asyncio.Event()

    async def wait():
        started.set()
        await asyncio.Event().wait()

    commands = _Commands(wait)
    sandbox = _sandbox(monkeypatch, commands)

    async def main():
        task = asyncio.create_task(sandbox.exec(["sleep", "1000"]))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())

    assert [cmd for cmd, _ in commands.kill_calls()] == ["kill -KILL -- -4242 2>/dev/null || true"]
    assert commands.handle.killed


@pytest.mark.cpu
@pytest.mark.level0
def test_normal_exit_issues_no_kill(monkeypatch):
    async def wait():
        return SimpleNamespace(exit_code=0, stdout="ok", stderr="")

    commands = _Commands(wait)
    sandbox = _sandbox(monkeypatch, commands)

    result = asyncio.run(sandbox.exec(["echo", "ok"]))

    assert (result.exit_code, result.stdout) == (0, "ok")
    assert commands.kill_calls() == []
    assert not commands.handle.killed


@pytest.mark.cpu
@pytest.mark.level0
def test_failed_group_kill_still_kills_the_handle_and_keeps_the_timeout(monkeypatch):
    async def wait():
        raise TimeoutException("deadline exceeded")

    async def kill_run():
        raise RuntimeError("envd unreachable")

    commands = _Commands(wait, kill_run)
    sandbox = _sandbox(monkeypatch, commands)

    result = asyncio.run(sandbox.exec(["sleep", "1000"], timeout=3))

    assert result.exit_code == -1
    assert commands.handle.killed


@pytest.mark.cpu
@pytest.mark.level0
def test_hanging_group_kill_is_bounded(monkeypatch):
    import uni_agent.sandbox.e2b as e2b_module

    monkeypatch.setattr(e2b_module, "_KILL_TIMEOUT", 0.05)

    async def wait():
        raise TimeoutException("deadline exceeded")

    async def kill_run():
        await asyncio.Event().wait()

    commands = _Commands(wait, kill_run)
    sandbox = _sandbox(monkeypatch, commands)

    async def main():
        return await asyncio.wait_for(sandbox.exec(["sleep", "1000"], timeout=3), timeout=2)

    result = asyncio.run(main())

    assert result.exit_code == -1
    assert commands.handle.killed  # the leader is still killed when the group kill stalls
