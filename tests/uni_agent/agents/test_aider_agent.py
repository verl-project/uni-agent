"""Aider host adapter tests without GPU or provider dependencies."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from uni_agent.agents.aider.agent import AiderAgent, AiderConfig, _diagnostic
from uni_agent.agents.base import ModelConfig
from uni_agent.agents.registry import build_agent


class FakeSandbox:
    def __init__(self, mode="success"):
        self.mode = mode
        self.shells, self.commands, self.files = [], [], {}

    async def exec_shell(self, command, **kwargs):
        self.shells.append(command)
        if self.mode == "mkdir_failure":
            return SimpleNamespace(exit_code=1, stdout="", stderr="read-only fs")
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    async def write_file(self, path, content):
        self.files[path] = content

    async def exec(self, argv, **kwargs):
        self.commands.append((argv, kwargs))
        if self.mode == "timeout_exception":
            raise TimeoutError("deadline")
        if self.mode == "startup":
            raise FileNotFoundError("aider")
        if self.mode == "exit_timeout":
            return SimpleNamespace(exit_code=-1, stdout="", stderr="deadline")
        secret = "sk-test-secret"
        return SimpleNamespace(
            exit_code=1 if self.mode == "exit_failure" else 0,
            stdout=f"Applied edit. Tokens used with {secret}" if self.mode == "leaky" else "Applied edit to hello.py",
            stderr="",
        )


def make_config(**overrides):
    return AiderConfig(
        model=ModelConfig(base_url="http://endpoint/v1", api_key="sk-test-secret", model_name="deepseek-chat"),
        **overrides,
    )


def run_agent(mode, config=None, messages=None):
    sandbox = FakeSandbox(mode)
    if messages is None:
        messages = [{"role": "user", "content": "solve 中文 ' $(false)"}]
    result = asyncio.run(AiderAgent(config or make_config()).run(sandbox=sandbox, messages=messages, workdir="/repo"))
    return result, sandbox


@pytest.mark.cpu
@pytest.mark.level0
def test_success_launches_headless_aider():
    result, sandbox = run_agent("success")
    assert result.finished is True
    argv, options = sandbox.commands[0]
    assert argv[0] == "aider"
    assert "--message-file" in argv
    assert "openai/deepseek-chat" in argv
    assert "http://endpoint/v1" in argv
    assert "--no-auto-commits" in argv
    assert "--no-restore-chat-history" in argv
    assert options["timeout"] == 1800
    assert options["workdir"] == "/repo"
    # The prompt only travels by file, never through argv.
    prompt_path = argv[argv.index("--message-file") + 1]
    assert sandbox.files[prompt_path] == "solve 中文 ' $(false)"
    assert not any("solve" in arg for arg in argv)
    # The API key uses the private exec environment, never argv.
    assert options["env"]["OPENAI_API_KEY"] == "sk-test-secret"
    assert not any("sk-test-secret" in arg for arg in argv)
    assert result.info["exit_code"] == 0
    assert result.info["session_id"]


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
    with pytest.raises(ValueError, match="aider requires"):
        asyncio.run(AiderAgent(make_config()).run(sandbox=sandbox, messages=messages))
    assert not sandbox.shells and not sandbox.commands and not sandbox.files


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("model", [{}, {"base_url": "http://endpoint/v1"}, {"model_name": "policy"}])
def test_missing_model_fields_raise(model):
    sandbox = FakeSandbox()
    with pytest.raises(ValueError, match="aider requires config.model"):
        asyncio.run(
            AiderAgent(AiderConfig(model=ModelConfig(**model))).run(
                sandbox=sandbox, messages=[{"role": "user", "content": "task"}]
            )
        )
    assert not sandbox.commands


@pytest.mark.cpu
@pytest.mark.level0
def test_exit_failure_marks_unfinished():
    result, _ = run_agent("exit_failure")
    assert result.finished is False
    assert result.info["error_kind"] == "agent_failure"
    assert result.info["exit_code"] == 1


@pytest.mark.cpu
@pytest.mark.level0
def test_sandbox_timeout_exit_marks_timeout():
    result, _ = run_agent("exit_timeout")
    assert result.finished is False
    assert result.info["error_kind"] == "timeout"
    assert result.info["exit_code"] == -1


@pytest.mark.cpu
@pytest.mark.level0
def test_exec_timeout_exception_marks_timeout():
    result, _ = run_agent("timeout_exception")
    assert result.finished is False
    assert result.info["error_kind"] == "timeout"
    assert result.info["error_type"] == "TimeoutError"


@pytest.mark.cpu
@pytest.mark.level0
def test_missing_cli_marks_startup_failure():
    result, _ = run_agent("startup")
    assert result.finished is False
    assert result.info["error_kind"] == "startup_failure"
    assert result.info["error_type"] == "FileNotFoundError"


@pytest.mark.cpu
@pytest.mark.level0
def test_prepare_failure_short_circuits():
    result, sandbox = run_agent("mkdir_failure")
    assert result.finished is False
    assert result.info["error_kind"] == "startup_failure"
    assert not sandbox.commands and not sandbox.files


@pytest.mark.cpu
@pytest.mark.level0
def test_optional_flags_are_forwarded():
    result, sandbox = run_agent(
        "success", config=make_config(edit_format="udiff", extra_args=["--show-diffs"], run_timeout=42)
    )
    argv, options = sandbox.commands[0]
    assert options["timeout"] == 42
    assert argv[-3:] == ["--edit-format", "udiff", "--show-diffs"]


@pytest.mark.cpu
@pytest.mark.level0
def test_diagnostics_redact_api_key():
    result, _ = run_agent("leaky")
    assert "sk-test-secret" not in result.info["stdout_tail"]
    assert "<redacted>" in result.info["stdout_tail"]
    assert _diagnostic("token sk-test-secret", "sk-test-secret") == "token <redacted>"


@pytest.mark.cpu
@pytest.mark.level0
def test_registry_builds_aider_agent():
    agent = build_agent(make_config())
    assert isinstance(agent, AiderAgent)
    assert agent.name == "aider"
