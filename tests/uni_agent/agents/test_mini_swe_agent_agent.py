"""Tests for the mini-swe-agent agent's host-side glue.

mini-swe-agent owns its own loop and runs entirely *inside* the sandbox (like
claude_code), so there's nothing to drive here except the sandbox calls: the
venv install step, the config/driver files it writes, the launch argv, and how
it turns the sandbox's result file back into an :class:`AgentResult`. No real
sandbox / mini-swe-agent install -- :class:`_FakeSandbox` is a tiny in-memory
fake, so this runs fast under ``pytest`` (or ``python`` on this file).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from uni_agent.agents.base import AgentResult, ModelConfig
from uni_agent.agents.mini_swe_agent.agent import MiniSweAgentAgent, MiniSweAgentConfig
from uni_agent.sandbox.base import ExecResult


class _FakeSandbox:
    """Records every call and answers just enough to drive the agent's flow."""

    def __init__(self, *, install_exit_code: int = 0, result: dict | None = None):
        self.install_exit_code = install_exit_code
        self._result = result if result is not None else {"exit_status": "Submitted", "submission": "diff --git..."}
        self.exec_shell_calls: list[str] = []
        self.exec_calls: list[dict] = []
        self.written_files: dict[str, str] = {}

    async def exec_shell(self, script, *, timeout=None, workdir=None, env=None):
        self.exec_shell_calls.append(script)
        stderr = "" if self.install_exit_code == 0 else "boom"
        return ExecResult(exit_code=self.install_exit_code, stdout="", stderr=stderr)

    async def write_file(self, path, content):
        self.written_files[path] = content

    async def exec(self, argv, *, timeout=None, workdir=None, env=None):
        self.exec_calls.append({"argv": argv, "timeout": timeout})
        return ExecResult(exit_code=0, stdout="mini-swe-agent noisy log line\n", stderr="")

    async def read_file(self, path):
        assert path in self.written_files or path.endswith("result.json")
        return json.dumps(self._result).encode("utf-8")


def _agent(**config_kwargs) -> MiniSweAgentAgent:
    model = ModelConfig(base_url="http://gateway:8000/v1", model_name="policy")
    return MiniSweAgentAgent(MiniSweAgentConfig(model=model, **config_kwargs))


# --------------------------- validation ---------------------------


def test_missing_base_url_raises():
    agent = MiniSweAgentAgent(MiniSweAgentConfig())
    with pytest.raises(ValueError, match="base_url"):
        asyncio.run(agent.run(sandbox=_FakeSandbox(), messages=[{"role": "user", "content": "fix the bug"}]))


def test_missing_user_message_raises():
    agent = _agent()
    with pytest.raises(ValueError, match="requires a 'user' message"):
        asyncio.run(agent.run(sandbox=_FakeSandbox(), messages=[{"role": "system", "content": "sys"}]))


def test_too_many_messages_raises():
    agent = _agent()
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "user", "content": "u2"}]
    with pytest.raises(ValueError, match="at most 2 messages"):
        asyncio.run(agent.run(sandbox=_FakeSandbox(), messages=messages))


# --------------------------- happy path ---------------------------


def test_run_installs_writes_config_and_launches_the_venv_python():
    sandbox = _FakeSandbox()
    agent = _agent(step_limit=25)
    messages = [{"role": "system", "content": "be careful"}, {"role": "user", "content": "fix the off-by-one bug"}]

    result = asyncio.run(agent.run(sandbox=sandbox, messages=messages))

    # install step ran once, before anything else touched the sandbox.
    assert len(sandbox.exec_shell_calls) == 1
    assert "/opt/mini-swe-agent-venv" in sandbox.exec_shell_calls[0]
    assert "pip install" in sandbox.exec_shell_calls[0]

    # driver script + task config were written before the launch.
    driver_paths = [p for p in sandbox.written_files if p.endswith("run_agent.py")]
    config_paths = [p for p in sandbox.written_files if p.endswith("task.json")]
    assert len(driver_paths) == 1 and len(config_paths) == 1
    assert "DefaultAgent" in sandbox.written_files[driver_paths[0]]

    task_config = json.loads(sandbox.written_files[config_paths[0]])
    assert task_config["task"] == "fix the off-by-one bug"
    assert task_config["agent"]["step_limit"] == 25
    assert task_config["agent"]["system_template"] == "be careful"
    assert task_config["environment"]["cwd"] == "/testbed"
    assert task_config["model"]["model_name"] == "openai/policy"
    assert task_config["model"]["model_kwargs"]["api_base"] == "http://gateway:8000/v1"
    assert task_config["model"]["cost_tracking"] == "ignore_errors"

    # launched with the venv's own interpreter against the driver + config + result paths.
    assert len(sandbox.exec_calls) == 1
    argv = sandbox.exec_calls[0]["argv"]
    assert argv[0] == "/opt/mini-swe-agent-venv/bin/python"
    assert argv[1] == driver_paths[0]
    assert argv[2] == config_paths[0]
    assert argv[3].endswith("result.json")

    # result file (not the noisy stdout) is what fills the AgentResult.
    assert isinstance(result, AgentResult)
    assert result.output["exit_status"] == "Submitted"
    assert result.output["submission"] == "diff --git..."
    assert result.output["agent_stdout"] == "mini-swe-agent noisy log line\n"


def test_default_model_name_used_when_unset():
    sandbox = _FakeSandbox()
    config = MiniSweAgentConfig(model=ModelConfig(base_url="http://gateway:8000/v1"))
    agent = MiniSweAgentAgent(config)

    asyncio.run(agent.run(sandbox=sandbox, messages=[{"role": "user", "content": "task"}]))

    config_path = next(p for p in sandbox.written_files if p.endswith("task.json"))
    assert json.loads(sandbox.written_files[config_path])["model"]["model_name"] == "openai/default"


def test_install_failure_raises_runtime_error():
    sandbox = _FakeSandbox(install_exit_code=1)
    agent = _agent()
    with pytest.raises(RuntimeError, match="install step failed"):
        asyncio.run(agent.run(sandbox=sandbox, messages=[{"role": "user", "content": "task"}]))


def test_custom_install_command_is_honored():
    sandbox = _FakeSandbox()
    agent = _agent(install_command="echo custom-install")
    asyncio.run(agent.run(sandbox=sandbox, messages=[{"role": "user", "content": "task"}]))
    assert sandbox.exec_shell_calls == ["echo custom-install"]


def test_unreadable_result_file_is_reported_instead_of_raising():
    class _BrokenResultSandbox(_FakeSandbox):
        async def read_file(self, path):
            return b"not json"

    agent = _agent()
    result = asyncio.run(agent.run(sandbox=_BrokenResultSandbox(), messages=[{"role": "user", "content": "task"}]))
    assert result.output["exit_status"] == "runner_error"
    assert "error" in result.info


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
