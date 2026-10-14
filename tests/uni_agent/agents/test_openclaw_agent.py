"""Host adapter tests without GPU or provider dependencies."""

from __future__ import annotations

import asyncio
import json
import shlex
from types import SimpleNamespace

import pytest

from uni_agent.agents.base import ModelConfig
from uni_agent.agents.openclaw.agent import OpenClawAgent, OpenClawConfig, _diagnostic, _redact, parse_openclaw_result


class FakeSandbox:
    def __init__(self, mode="success"):
        self.mode = mode
        self.shells, self.commands, self.files = [], [], {}

    async def exec_shell(self, command, **kwargs):
        self.shells.append(command)
        if command.startswith("rm -f") and self.mode in {"cleanup_failure", "timeout_cleanup"}:
            raise OSError("cleanup unavailable")
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    async def write_file(self, path, content):
        self.files[path] = content

    async def exec(self, argv, **kwargs):
        self.commands.append((argv, kwargs))
        if self.mode in {"timeout", "timeout_cleanup"}:
            raise TimeoutError("deadline")
        if self.mode == "exit_timeout":
            return SimpleNamespace(exit_code=-1, stdout="", stderr="deadline")
        if self.mode == "startup":
            raise FileNotFoundError("openclaw")
        payload = {
            "payloads": [{"text": "DONE"}],
            "meta": {"aborted": False, "executionTrace": {"fallbackUsed": self.mode == "fallback"}},
        }
        return SimpleNamespace(
            exit_code=1 if self.mode == "exit_failure" else 0,
            stdout="not JSON" if self.mode == "protocol" else json.dumps(payload),
            stderr="",
        )


def run_agent(mode, messages=None, **config_overrides):
    sandbox = FakeSandbox(mode)
    cfg = OpenClawConfig(model=ModelConfig(base_url="http://endpoint/v1", model_name="policy"), **config_overrides)
    if messages is None:
        messages = [{"role": "user", "content": "solve 中文 ' $(false)"}]
    result = asyncio.run(
        OpenClawAgent(cfg).run(
            sandbox=sandbox,
            messages=messages,
            workdir="/workspace",
        )
    )
    assert any(command.startswith("rm -f ") for command in sandbox.shells)
    return result, sandbox


@pytest.mark.cpu
@pytest.mark.level0
def test_success():
    result, sandbox = run_agent("success")
    assert result.finished
    assert len(sandbox.commands) == 1
    launch_command = " ".join(sandbox.commands[0][0])
    assert "--message-file" in launch_command
    message_path = sandbox.commands[0][0][-1]
    message_path = message_path.split("--message-file", 1)[1].split()[0].strip("'\"")
    assert sandbox.files[message_path] == "solve 中文 ' $(false)"
    assert "solve" not in launch_command


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "system", "content": "instructions"}],
        [{"role": "assistant", "content": "answer"}],
        [{"role": "system", "content": "instructions"}, {"role": "user", "content": "task"}],
        [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}],
        [{"role": "user", "content": ""}],
        [{"role": "user", "content": " \n\t"}],
        [{"role": "user", "content": None}],
        [{"role": "user", "content": [{"type": "text", "text": "task"}]}],
    ],
)
def test_invalid_messages_fail_before_sandbox_operations(messages):
    sandbox = FakeSandbox()
    cfg = OpenClawConfig(model=ModelConfig(base_url="http://endpoint/v1", model_name="policy"))
    with pytest.raises(ValueError, match="openclaw requires"):
        asyncio.run(OpenClawAgent(cfg).run(sandbox=sandbox, messages=messages))
    assert not sandbox.shells and not sandbox.commands and not sandbox.files


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("config_overrides, expected_timeout", [({}, 1700), ({"cli_timeout_seconds": 23}, 23)])
def test_cli_budget_and_outer_grace(config_overrides, expected_timeout):
    result, sandbox = run_agent("success", **config_overrides)
    assert result.finished
    argv, options = sandbox.commands[0]
    command = shlex.split(argv[-1])
    assert command[:2] == ["exec", "openclaw"]
    assert command[command.index("--timeout") + 1] == str(expected_timeout)
    assert options["timeout"] == expected_timeout + 100
    config = json.loads(next(value for path, value in sandbox.files.items() if path.endswith("openclaw.json")))
    assert config["agents"]["defaults"]["timeoutSeconds"] == expected_timeout


@pytest.mark.cpu
@pytest.mark.level0
def test_large_user_text_is_forwarded_without_a_byte_cap():
    prompt = "任务" * 700_000
    result, sandbox = run_agent("success", messages=[{"role": "user", "content": prompt}])
    assert result.finished
    assert next(value for path, value in sandbox.files.items() if path.endswith("task.txt")) == prompt


@pytest.mark.cpu
@pytest.mark.level0
def test_fallback():
    assert not run_agent("fallback")[0].finished


@pytest.mark.cpu
@pytest.mark.level0
def test_protocol():
    result, _ = run_agent("protocol")
    assert not result.finished
    assert result.info["error_kind"] == "protocol_error"


@pytest.mark.cpu
@pytest.mark.level0
def test_exit_failure():
    result, _ = run_agent("exit_failure")
    assert not result.finished
    assert result.info["error_kind"] == "agent_failure"


@pytest.mark.cpu
@pytest.mark.level0
def test_timeout():
    result, _ = run_agent("timeout")
    assert not result.finished
    assert result.info["error_kind"] == "timeout"


@pytest.mark.cpu
@pytest.mark.level0
def test_exit_code_timeout():
    result, _ = run_agent("exit_timeout")
    assert not result.finished
    assert result.info["error_kind"] == "timeout"


@pytest.mark.cpu
@pytest.mark.level0
def test_startup():
    result, _ = run_agent("startup")
    assert not result.finished
    assert result.info["error_kind"] == "startup_failure"


@pytest.mark.cpu
@pytest.mark.level0
def test_cleanup_failure():
    result, _ = run_agent("cleanup_failure")
    assert not result.finished
    assert result.info["error_kind"] == "cleanup_failure"


@pytest.mark.cpu
@pytest.mark.level0
def test_timeout_cleanup_keeps_original_error():
    result, _ = run_agent("timeout_cleanup")
    assert result.info["error_kind"] == "timeout"
    assert result.info["cleanup_error"] == "OSError"


@pytest.mark.cpu
@pytest.mark.level0
def test_redacts_payload_and_bounded_diagnostic():
    value = {"apiKey": "hidden", "data": ["prefix fake-secret suffix", "Bearer abc123"]}
    rendered = json.dumps(_redact(value, "fake-secret"))
    for secret in ("hidden", "fake-secret", "abc123"):
        assert secret not in rendered
    assert len(_diagnostic("x" * 9000)) <= 4000


@pytest.mark.cpu
@pytest.mark.level0
def test_nested_parser():
    assert parse_openclaw_result('noise\n{"meta":{"a":1}}') == {"meta": {"a": 1}}


@pytest.mark.cpu
@pytest.mark.level0
def test_last_object():
    assert parse_openclaw_result('{"old":1}\n{"new":{"b":2}}') == {"new": {"b": 2}}
