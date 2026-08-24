"""Tests for the mini-swe-agent agent's host-side glue.

mini-swe-agent runs entirely *inside* the sandbox from a prebuilt tool image
mounted at ``/opt/mini-swe-agent``. The agent's host-side glue is thin:
base64-encode the task config (the gateway URL is passed through as-is; the
reverse-tunnel rewrite happens in ``run_task``), pipe it into the tool-image
python via stdin, and parse the result JSON out of stdout (litellm noise
tolerated). These tests cover that glue with a tiny in-memory fake sandbox, so
they run fast under ``pytest`` (or ``python``).
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from uni_agent.agents.base import AgentResult, ModelConfig
from uni_agent.agents.mini_swe_agent.agent import (
    MiniSweAgentAgent,
    MiniSweAgentConfig,
    build_agent_command,
    parse_agent_result,
)
from uni_agent.sandbox.base import ExecResult


class _FakeSandbox:
    """Records the one ``exec_shell`` call and returns canned stdout."""

    def __init__(self, *, stdout: str = "", exit_code: int = 0):
        self._stdout = stdout
        self._exit_code = exit_code
        self.exec_shell_calls: list[str] = []

    async def exec_shell(self, script, *, timeout=None, workdir=None, env=None):
        self.exec_shell_calls.append(script)
        return ExecResult(exit_code=self._exit_code, stdout=self._stdout, stderr="")


_TOOL_PYTHON = "/opt/mini-swe-agent/bin/python"
_RUN_AGENT_SCRIPT = "/opt/mini-swe-agent/bin/run_agent.py"


def _agent(base_url: str = "http://gateway:8000/v1", **config_kwargs) -> MiniSweAgentAgent:
    model = ModelConfig(base_url=base_url, model_name="policy")
    config_kwargs.setdefault("tool_python", _TOOL_PYTHON)
    config_kwargs.setdefault("run_agent_script", _RUN_AGENT_SCRIPT)
    return MiniSweAgentAgent(MiniSweAgentConfig(model=model, **config_kwargs))


def _decode_task_config(cmd: str) -> dict:
    """Pull the base64 task config out of a built agent command and decode it."""
    _, _, after = cmd.partition("printf %s ")
    config_b64, _, _ = after.partition(" | base64 -d")
    return json.loads(base64.b64decode(config_b64))


# --------------------------- helpers ---------------------------


def test_build_agent_command_pipes_config_and_invokes_tool_python():
    cmd = build_agent_command(
        config_b64="Zm9v",
        conda_env="testbed",
        tool_python=_TOOL_PYTHON,
        run_agent_script=_RUN_AGENT_SCRIPT,
    )
    assert cmd.startswith("unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy;")
    assert "printf %s Zm9v | base64 -d |" in cmd
    assert "/opt/mini-swe-agent/bin/python /opt/mini-swe-agent/bin/run_agent.py" in cmd
    # The task conda env is activated around the launch.
    assert "CONDA_DEFAULT_ENV=testbed" in cmd
    assert "/opt/miniconda3/envs/testbed/bin" in cmd


def test_build_agent_command_honors_overrides():
    cmd = build_agent_command(
        config_b64="",
        conda_env="myenv",
        tool_python="/x/python",
        run_agent_script="/y/run.py",
    )
    assert "/x/python /y/run.py" in cmd
    assert "CONDA_DEFAULT_ENV=myenv" in cmd


def test_parse_agent_result_empty_is_error():
    assert parse_agent_result("") == {"exit_status": "error", "submission": ""}


def test_parse_agent_result_picks_last_json_line_ignoring_litellm_noise():
    stdout = "litellm warning: something\nrandom noise\n" + json.dumps(
        {"exit_status": "Submitted", "submission": "diff"}
    )
    assert parse_agent_result(stdout) == {"exit_status": "Submitted", "submission": "diff"}


def test_parse_agent_result_single_json_object():
    result = parse_agent_result(json.dumps({"exit_status": "error", "submission": ""}))
    assert result["exit_status"] == "error"


def test_parse_agent_result_unparseable_is_error():
    assert parse_agent_result("totally not json at all") == {"exit_status": "error", "submission": ""}


# --------------------------- validation ---------------------------


def test_missing_base_url_raises():
    agent = MiniSweAgentAgent(MiniSweAgentConfig(tool_python=_TOOL_PYTHON, run_agent_script=_RUN_AGENT_SCRIPT))
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


def test_run_pipes_config_and_parses_stdout_into_result():
    stdout = "mini-swe-agent noisy log line\n" + json.dumps(
        {"exit_status": "Submitted", "submission": "diff --git a/...", "model_stats": {"api_calls": 3}}
    )
    sandbox = _FakeSandbox(stdout=stdout)
    agent = _agent(step_limit=25)
    messages = [{"role": "system", "content": "be careful"}, {"role": "user", "content": "fix the off-by-one bug"}]

    result = asyncio.run(agent.run(sandbox=sandbox, messages=messages))

    # Exactly one exec_shell, piping the base64 config into the tool-image python.
    assert len(sandbox.exec_shell_calls) == 1
    cmd = sandbox.exec_shell_calls[0]
    assert "base64 -d" in cmd
    assert "/opt/mini-swe-agent/bin/python /opt/mini-swe-agent/bin/run_agent.py" in cmd

    # The decoded task config carries the user task, the gateway URL (base_url
    # passed through unchanged -- run_task does the tunnel rewrite), and step_limit.
    task_config = _decode_task_config(cmd)
    assert task_config["task"] == "fix the off-by-one bug"
    assert task_config["gateway_url"] == "http://gateway:8000/v1"
    assert task_config["agent"]["step_limit"] == 25

    # The result is parsed out of stdout (litellm noise tolerated) into an AgentResult.
    assert isinstance(result, AgentResult)
    assert result.output["exit_status"] == "Submitted"
    assert result.output["submission"] == "diff --git a/..."
    assert result.output["model_stats"]["api_calls"] == 3
    assert result.info == {"step_limit": 25, "exit_status": "Submitted"}
    assert result.finished is True
    assert result.transcript == messages


def test_run_marks_unfinished_when_not_submitted():
    sandbox = _FakeSandbox(stdout=json.dumps({"exit_status": "error", "submission": ""}))
    agent = _agent()
    result = asyncio.run(agent.run(sandbox=sandbox, messages=[{"role": "user", "content": "task"}]))
    assert result.finished is False
    assert result.info["exit_status"] == "error"


def test_run_passes_base_url_through_unchanged():
    sandbox = _FakeSandbox(stdout=json.dumps({"exit_status": "Submitted", "submission": "diff"}))
    # A tunnel-rewritten base_url (what run_task injects) is forwarded verbatim.
    agent = _agent(base_url="http://127.0.0.1:38197/sessions/abc/v1")
    asyncio.run(agent.run(sandbox=sandbox, messages=[{"role": "user", "content": "task"}]))
    task_config = _decode_task_config(sandbox.exec_shell_calls[0])
    assert task_config["gateway_url"] == "http://127.0.0.1:38197/sessions/abc/v1"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
