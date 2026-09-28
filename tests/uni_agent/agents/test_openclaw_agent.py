"""Host adapter tests without GPU or provider dependencies."""

from __future__ import annotations

import asyncio
import json
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
        if argv[0] == "python3":
            return SimpleNamespace(exit_code=0, stdout=json.dumps({"verified": self.mode != "audit_failure"}), stderr="")
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


def run_agent(mode):
    sandbox = FakeSandbox(mode)
    cfg = OpenClawConfig(model=ModelConfig(base_url="http://endpoint/v1", model_name="policy"))
    result = asyncio.run(
        OpenClawAgent(cfg).run(
            sandbox=sandbox,
            messages=[
                {"role": "system", "content": "follow repository policy"},
                {"role": "user", "content": "solve 中文 ' $(false)"},
            ],
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
    assert len(sandbox.commands) == 2
    launch_command = " ".join(sandbox.commands[0][0])
    assert "--message-file" in launch_command
    message_path = sandbox.commands[0][0][-1]
    message_path = message_path.split("--message-file", 1)[1].split()[0].strip("'\"")
    assert "System instructions:\nfollow repository policy" in sandbox.files[message_path]
    assert "solve 中文 ' $(false)" in sandbox.files[message_path]
    assert "solve" not in launch_command


@pytest.mark.cpu
@pytest.mark.level0
def test_audit_failure():
    assert not run_agent("audit_failure")[0].finished


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
    assert parse_openclaw_result("noise\n{\"meta\":{\"a\":1}}") == {"meta": {"a": 1}}


@pytest.mark.cpu
@pytest.mark.level0
def test_last_object():
    assert parse_openclaw_result('{"old":1}\n{"new":{"b":2}}') == {"new": {"b": 2}}
