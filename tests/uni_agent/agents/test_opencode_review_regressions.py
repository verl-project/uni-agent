"""Review regressions shared by the 1.x and pinned 2.x adapters."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_opencode_agent import SUCCESS, FakeSandbox

from uni_agent.agents.base import ModelConfig
from uni_agent.agents.opencode import OpenCodeAgent, OpenCodeConfig
from uni_agent.agents.opencode2 import OpenCode2Agent, OpenCode2Config
from uni_agent.sandbox import local

pytestmark = [pytest.mark.cpu, pytest.mark.level0]
ADAPTERS = [(OpenCodeAgent, OpenCodeConfig, "1.19.3"), (OpenCode2Agent, OpenCode2Config, "opencode v2.2.5")]


@pytest.mark.parametrize("agent_cls,config_cls,version", ADAPTERS)
@pytest.mark.parametrize("port", ["abc", "65536", "-1"])
def test_bad_port_fails_before_any_command(agent_cls, config_cls, version, port):
    sandbox = FakeSandbox(version=version)
    cfg = config_cls(model=ModelConfig(base_url=f"http://gateway:{port}/v1", model_name="test", api_key="dummy"))
    with pytest.raises(ValueError):
        asyncio.run(agent_cls(cfg).run(sandbox=sandbox, messages=[{"role": "user", "content": "test"}]))
    assert not sandbox.commands


@pytest.mark.parametrize("agent_cls,config_cls,version", ADAPTERS)
@pytest.mark.parametrize("cancel_during_run", [False, True])
def test_repeated_cancel_waits_for_private_directory_removal(
    tmp_path, agent_cls, config_cls, version, cancel_during_run
):
    async def scenario():
        running, removing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        sandbox = local.LocalSandbox()
        original = sandbox.exec
        state = tmp_path / "private"

        async def execute(argv, **kwargs):
            if argv[-1] == "--version":
                return SimpleNamespace(exit_code=0, stdout=version, stderr="")
            if argv[1:2] == ["run"]:
                running.set()
                if cancel_during_run:
                    await asyncio.Event().wait()
                return SimpleNamespace(exit_code=0, stdout=SUCCESS, stderr="")
            if argv[0] == "rm":
                removing.set()
                await release.wait()
            return await original(argv, **kwargs)

        sandbox.exec = execute
        cfg = config_cls(
            state_root=str(state), model=ModelConfig(base_url="http://gateway/v1", model_name="test", api_key="dummy")
        )
        task = asyncio.create_task(agent_cls(cfg).run(sandbox=sandbox, messages=[{"role": "user", "content": "x"}]))
        try:
            await asyncio.wait_for(running.wait(), 5)
            if cancel_during_run:
                task.cancel()
            await asyncio.wait_for(removing.wait(), 5)
            assert list(state.iterdir())
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
            assert not task.done(), "caller must retain ownership until removal finishes"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            assert not list(state.iterdir())
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await sandbox.stop()

    asyncio.run(scenario())


def test_non_posix_fails_before_spawn(monkeypatch):
    monkeypatch.setattr(local, "os", SimpleNamespace(name="nt"))
    with pytest.raises(NotImplementedError, match="POSIX"):
        asyncio.run(local.LocalSandbox()._exec(["never-spawn-this"]))
    result = asyncio.run(local.LocalSandbox().exec(["never-spawn-this"]))
    assert result.exit_code != 0 and "POSIX" in result.stderr


def test_v2_registry_in_fresh_interpreter():
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from uni_agent.agents import get_agent_cls; print(get_agent_cls('opencode2').__name__)",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    assert result.stdout.strip() == "OpenCode2Agent"
