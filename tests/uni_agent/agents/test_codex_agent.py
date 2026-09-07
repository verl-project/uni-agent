from __future__ import annotations

import asyncio
import base64
import json

import pytest

from uni_agent.agents.base import AgentResult, ModelConfig
from uni_agent.agents.codex.agent import CodexAgent, CodexConfig, build_agent_command, parse_agent_result
from uni_agent.sandbox.base import ExecResult

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


class FakeSandbox:
    def __init__(self, stdout: str = "", exit_code: int = 0):
        self.stdout = stdout
        self.exit_code = exit_code
        self.calls: list[dict] = []

    async def exec_shell(self, script, *, timeout=None, workdir=None, env=None):
        self.calls.append({"script": script, "timeout": timeout, "workdir": workdir, "env": env})
        return ExecResult(exit_code=self.exit_code, stdout=self.stdout, stderr="")


def make_agent(**kwargs):
    kwargs.setdefault("tool_script", "/opt/codex/bin/run_agent.sh")
    kwargs.setdefault("model", ModelConfig(base_url="http://gateway/v1", model_name="policy"))
    return CodexAgent(CodexConfig(**kwargs))


def test_build_agent_command_uses_stdin_and_isolates_env():
    task = base64.b64encode(b"fix 'this'").decode()
    command = build_agent_command(
        task_b64=task,
        tool_script="/opt/codex/bin/run_agent.sh",
        gateway_url="http://127.0.0.1:38197/sessions/s1/v1",
        model_name="policy",
        api_key="key",
        project_dir="/testbed",
    )
    assert "| base64 -d |" in command
    assert "CODEX_API_BASE=http://127.0.0.1:38197/sessions/s1/v1" in command
    assert "CODEX_MODEL=policy" in command
    assert "CODEX_HOME=" not in command
    assert "fix 'this'" not in command


def test_parse_agent_result_jsonl():
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "t"}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "fixed"}}),
            json.dumps({"type": "turn.completed"}),
        ]
    )
    assert parse_agent_result(stdout, 0) == {
        "exit_status": "ok",
        "ok": True,
        "content": "fixed",
        "event_count": 3,
    }


def test_parse_agent_result_timeout_and_failure():
    timeout = parse_agent_result("", -1)
    assert timeout["exit_status"] == "timeout"
    failed = parse_agent_result(json.dumps({"type": "turn.failed", "error": {"message": "bad"}}), 1)
    assert failed["exit_status"] == "error"
    assert failed["error"] == "bad"


def test_parse_agent_result_honors_wrapper_process_exit_event():
    stdout = "\n".join(
        [
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "partial"}}),
            json.dumps({"type": "process.completed", "exit_code": 17}),
        ]
    )
    failed = parse_agent_result(stdout, 0)
    assert failed["exit_status"] == "error"
    assert failed["ok"] is False
    assert failed["error"] == "codex exited with code 17"


def test_codex_agent_runs_and_returns_agent_result():
    sandbox = FakeSandbox(
        stdout=json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "done"}})
    )
    agent = make_agent()
    result = asyncio.run(agent.run(sandbox=sandbox, messages=[{"role": "user", "content": "fix bug"}]))
    assert isinstance(result, AgentResult)
    assert result.finished is True
    assert result.output["content"] == "done"
    assert len(sandbox.calls) == 1
    assert sandbox.calls[0]["workdir"] == "/testbed"


@pytest.mark.parametrize(
    "field,value", [("extra_args", []), ("extra_env", {}), ("codex_home", "/tmp/custom"), ("step_limit", 1)]
)
def test_codex_config_rejects_removed_or_unsupported_options(field, value):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=field):
        make_agent(**{field: value})
