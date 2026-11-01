"""Real helper processes must terminate even when an SDK command wait blocks."""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from uni_agent.sandbox import SandboxExecutionError, SandboxTerminatedError
from uni_agent.sandbox.openyuanrong import OpenyuanrongSandbox


@pytest.fixture
def provider(monkeypatch, tmp_path):
    # Only the remote service is simulated. The command worker, pipes, event
    # loop, lifecycle monitor, cancellation, and OS process reaping are real.
    distribution = tmp_path / "openyuanrong_sandbox-0.10.2rc9.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text("Metadata-Version: 2.1\nName: openyuanrong-sandbox\nVersion: 0.10.2rc9\n")
    (tmp_path / "yr_sandbox.py").write_text("""
from pathlib import Path
from types import SimpleNamespace
import time

class ConnectionConfig:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)

class Handle:
    id = "one-owned-command"
    pid = 123
    def __init__(self, command, env, cwd):
        self.command, self.env, self.cwd = command, env, cwd
        Path(env["MARKER"]).write_text("running")
    def wait(self, timeout):
        if self.command == "broken": raise OSError("command service unavailable")
        if self.command == "blocked": time.sleep(120)
        if self.command == "slow": time.sleep(.15)
        return SimpleNamespace(exit_code=7, stdout="x" * 100000 + self.env["VALUE"] + self.cwd,
                               stderr="detail", status=SimpleNamespace(value="FAILED"))
    def kill(self):
        Path(self.env["MARKER"]).write_text("killed")
        return True

class Commands:
    def run(self, command, *, background, envs, cwd, timeout):
        assert background is True
        return Handle(command, envs, cwd)

class Sandbox:
    commands = Commands()
    @classmethod
    def from_id(cls, sid, connection): return cls()
    def close(self): pass
""")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    instance = OpenyuanrongSandbox(image="test:1", command_wait_interval=0.01)
    instance._connection = SimpleNamespace(
        server_address="https://example.invalid",
        use_tls=True,
        gateway_address=None,
        gateway_use_tls=True,
        verify_tls=False,
        resolved_token=lambda: "test-token",
        token_provider=None,
    )
    instance._sandbox = SimpleNamespace(
        sandbox_id="owned-test", get_info=lambda: SimpleNamespace(state="running"), is_running=lambda: True
    )
    children = []
    spawn = asyncio.create_subprocess_exec

    async def record(*args, **kwargs):
        child = await spawn(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", record)
    yield instance, {"MARKER": str(tmp_path / "command-state"), "VALUE": "one two"}, children
    assert children
    assert all(child.returncode is not None and not Path(f"/proc/{child.pid}").exists() for child in children)


async def until_running(env):
    async def wait():
        while not Path(env["MARKER"]).exists():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), timeout=5)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_background_command_preserves_large_output_exit_code_and_context(provider):
    instance, env, _ = provider
    result = await instance.exec(["normal"], timeout=10, env=env, workdir="/testbed")
    assert (result.exit_code, result.stdout, result.stderr) == (7, "x" * 100000 + "one two/testbed", "detail")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_confirmed_death_reaps_worker_even_during_blocking_sdk_wait(provider):
    instance, env, _ = provider
    task = asyncio.create_task(instance.exec(["blocked"], timeout=120, env=env))
    await until_running(env)
    instance._sandbox.get_info = lambda: SimpleNamespace(state="stopped")
    with pytest.raises(SandboxTerminatedError, match="stopped"):
        await asyncio.wait_for(task, 2)
    # Confirmed dead sandbox does not require a second blocking SDK kill.
    assert Path(env["MARKER"]).read_text() == "running"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_health_network_error_is_not_a_death_notification(provider):
    instance, env, _ = provider

    def unavailable():
        raise OSError("temporary network failure")

    instance._sandbox.get_info = unavailable
    result = await instance.exec(["slow"], timeout=10, env=env, workdir="/testbed")
    assert result.exit_code == 7


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_cancellation_kills_command_and_reaps_blocked_worker(provider):
    instance, env, _ = provider
    task = asyncio.create_task(instance.exec(["blocked"], timeout=120, env=env))
    await until_running(env)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert Path(env["MARKER"]).read_text() == "killed"
    assert await instance.is_alive()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_deadline_kills_command_and_preserves_timeout_result(provider):
    instance, env, _ = provider
    result = await instance.exec(["blocked"], timeout=1, env=env)
    assert result.exit_code == -1 and "timed out" in result.stderr
    assert Path(env["MARKER"]).read_text() == "killed"
    assert await instance.is_alive()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_infrastructure_failure_is_not_downgraded_to_exit_127(provider):
    instance, env, _ = provider
    with pytest.raises(SandboxExecutionError, match="command service unavailable"):
        await instance.exec(["broken"], timeout=10, env=env)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("partial_packet", [False, True])
async def test_owner_pipe_closure_exits_worker_even_during_sdk_wait(provider, partial_packet):
    instance, env, children = provider
    task = asyncio.create_task(instance.exec(["blocked"], timeout=120, env=env))
    await until_running(env)
    try:
        if partial_packet:
            children[0].stdin.write(b"\x00\x00")
            await children[0].stdin.drain()
        children[0].stdin.close()
        await asyncio.wait_for(children[0].wait(), 2)
        with pytest.raises(SandboxExecutionError, match="without a valid response"):
            await asyncio.wait_for(task, 2)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
