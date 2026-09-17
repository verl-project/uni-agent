"""Host-side tests for the Hermes sidecar adapter."""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from uni_agent.agents.base import ModelConfig
from uni_agent.agents.hermes.agent import (
    HermesAgent,
    HermesConfig,
    build_runner_command,
    parse_result_stdout,
    split_messages,
    validate_result,
)
from uni_agent.sandbox.base import ExecResult


class FakeSandbox:
    def __init__(self, *, exit_code: int = 0, stderr: str = ""):
        self.exit_code = exit_code
        self.stderr = stderr
        self.files: dict[str, bytes] = {}
        self.commands: list[str] = []

    async def write_file(self, path: str, content: bytes | str) -> None:
        self.files[path] = content.encode() if isinstance(content, str) else content

    async def read_file(self, path: str) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def exec_shell(self, script: str, *, timeout=None, workdir=None, env=None) -> ExecResult:
        self.commands.append(script)
        result_path = re.search(r"--result-path (\S+)", script).group(1).strip("'")
        run_id = result_path.removeprefix("/tmp/hermes-").removesuffix("/result.json")
        if self.exit_code == 0:
            envelope = {
                "schema_version": 1,
                "run_id": run_id,
                "status": "completed",
                "finished": True,
                "final_response": "done",
                "error": None,
                "stop_reason": "text_response",
                "stats": {"api_calls": 3},
            }
            self.files[result_path] = json.dumps(envelope).encode()
        return ExecResult(exit_code=self.exit_code, stdout="noise", stderr=self.stderr)


def _agent(**kwargs) -> HermesAgent:
    kwargs.setdefault("model", ModelConfig(base_url="http://gateway/sessions/s/v1", model_name="policy"))
    return HermesAgent(HermesConfig(**kwargs))


@pytest.mark.cpu
@pytest.mark.level0
def test_split_messages_preserves_all_text_and_rejects_unrepresentable_roles():
    assert split_messages(
        [
            {"role": "system", "content": "rules"},
            {"role": "system", "content": "more rules"},
            {"role": "user", "content": "issue"},
            {"role": "user", "content": "details"},
        ]
    ) == ("rules\n\nmore rules", "issue\n\ndetails")
    with pytest.raises(ValueError, match="only supports"):
        split_messages([{"role": "assistant", "content": "hidden context"}, {"role": "user", "content": "issue"}])


@pytest.mark.cpu
@pytest.mark.level0
def test_command_does_not_contain_prompt_or_key():
    command = build_runner_command(
        input_path="/tmp/hermes/run/input.json",
        result_path="/tmp/hermes/run/result.json",
        messages_path="/tmp/hermes/run/messages.json",
        log_path="/tmp/hermes/run/runner.log",
        hermes_home="/tmp/hermes/run/home",
        environment_prefix="/custom env/testbed",
        tool_python="/opt/hermes/bin/python",
        runner_script="/opt/hermes/bin/run_hermes.py",
        terminal_timeout=600,
        approval_mode="off",
    )
    assert "/tmp/hermes/run/input.json" in command
    assert "--result-path" in command and "--messages-path" in command and "<" in command
    assert "prompt" not in command and "api_key" not in command
    assert "HERMES_INTERACTIVE=0" in command
    assert "HERMES_YOLO_MODE=1" in command
    assert "TERMINAL_TIMEOUT=600" in command
    assert "TERMINAL_TIMEOUT=600.0" not in command
    assert "'/custom env/testbed/bin'" in command
    assert "/opt/miniconda3" not in command


@pytest.mark.cpu
@pytest.mark.level0
def test_parse_and_validate_marked_result():
    result = {"schema_version": 1, "run_id": "run-1", "status": "completed", "finished": True}
    parsed = parse_result_stdout("untrusted json: {}\nHERMES_RECIPE_RESULT " + json.dumps(result))
    assert validate_result(parsed, run_id="run-1") == result
    assert parse_result_stdout(json.dumps(result)) is None
    with pytest.raises(ValueError, match="mismatch"):
        validate_result(result, run_id="other")
    with pytest.raises(ValueError, match="agree"):
        validate_result({**result, "status": "timeout"}, run_id="run-1")


@pytest.mark.cpu
@pytest.mark.level0
def test_run_reads_result_and_preserves_endpoint_and_messages():
    sandbox = FakeSandbox()
    agent = _agent()
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "fix it"}]
    result = asyncio.run(agent.run(sandbox=sandbox, messages=messages, workdir="/testbed"))
    assert result.finished is True
    assert result.output["status"] == "completed"
    assert result.transcript == messages
    assert len(sandbox.commands) == 1
    payload = json.loads(next(iter(sandbox.files.values())))
    assert payload["model"]["endpoint"] == "http://gateway/sessions/s/v1"
    assert payload["messages"] == messages


@pytest.mark.cpu
@pytest.mark.level0
def test_timeout_is_not_treated_as_success():
    result = asyncio.run(
        _agent().run(
            sandbox=FakeSandbox(exit_code=-1, stderr="exec timed out"),
            messages=[{"role": "user", "content": "fix it"}],
        )
    )
    assert result.finished is False
    assert result.output["status"] == "timeout"
