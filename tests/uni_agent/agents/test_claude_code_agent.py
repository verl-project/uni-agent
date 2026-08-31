from __future__ import annotations

import asyncio

import pytest

from uni_agent.agents.base import ModelConfig
from uni_agent.agents.claude_code.agent import (
    _CLAUDE_NATIVE_INSTALL_COMMAND,
    _CLAUDE_NPM_INSTALL_COMMAND,
    ClaudeCodeAgent,
    ClaudeCodeConfig,
)
from uni_agent.sandbox.base import ExecResult


class _FakeSandbox:
    def __init__(
        self,
        *,
        probe_results: list[int],
        npm_available: bool = True,
        install_exit_code: int = 0,
        install_stderr: str = "install failed",
        process_exit_code: int = 0,
    ):
        self.probe_results = list(probe_results)
        self.npm_available = npm_available
        self.install_exit_code = install_exit_code
        self.install_stderr = install_stderr
        self.process_exit_code = process_exit_code
        self.calls: list[dict] = []
        self.exec_calls: list[dict] = []

    async def exec_shell(self, script: str, *, timeout=None, workdir=None, env=None) -> ExecResult:
        self.calls.append({"script": script, "timeout": timeout})
        if script.startswith("command -v claude"):
            return ExecResult(exit_code=self.probe_results.pop(0), stdout="", stderr="")
        if script.startswith("command -v npm"):
            return ExecResult(exit_code=0 if self.npm_available else 1, stdout="", stderr="")
        stderr = self.install_stderr if self.install_exit_code else ""
        return ExecResult(exit_code=self.install_exit_code, stdout="", stderr=stderr)

    async def exec(self, argv, *, timeout=None, workdir=None, env=None) -> ExecResult:
        self.exec_calls.append({"argv": argv, "timeout": timeout, "workdir": workdir, "env": env})
        return ExecResult(exit_code=self.process_exit_code, stdout="done", stderr="")


def _agent() -> ClaudeCodeAgent:
    return ClaudeCodeAgent(ClaudeCodeConfig())


@pytest.mark.cpu
@pytest.mark.level0
def test_ensure_claude_skips_install_when_already_available():
    sandbox = _FakeSandbox(probe_results=[0])

    asyncio.run(_agent()._ensure_claude(sandbox))

    assert len(sandbox.calls) == 1
    assert sandbox.calls[0]["script"].startswith("command -v claude")


@pytest.mark.cpu
@pytest.mark.level0
def test_ensure_claude_installs_and_rechecks_path():
    sandbox = _FakeSandbox(probe_results=[1, 0])

    asyncio.run(_agent()._ensure_claude(sandbox))

    assert [call["script"] for call in sandbox.calls] == [
        "command -v claude >/dev/null 2>&1",
        "command -v npm >/dev/null 2>&1",
        _CLAUDE_NPM_INSTALL_COMMAND,
        "command -v claude >/dev/null 2>&1",
    ]
    assert sandbox.calls[2]["timeout"] == 600


@pytest.mark.cpu
@pytest.mark.level0
def test_ensure_claude_uses_native_installer_when_npm_is_missing():
    sandbox = _FakeSandbox(probe_results=[1, 0], npm_available=False)

    asyncio.run(_agent()._ensure_claude(sandbox))

    assert [call["script"] for call in sandbox.calls] == [
        "command -v claude >/dev/null 2>&1",
        "command -v npm >/dev/null 2>&1",
        _CLAUDE_NATIVE_INSTALL_COMMAND,
        "command -v claude >/dev/null 2>&1",
    ]
    assert sandbox.calls[2]["timeout"] == 600


@pytest.mark.cpu
@pytest.mark.level0
def test_ensure_claude_surfaces_install_failure():
    sandbox = _FakeSandbox(probe_results=[1], install_exit_code=1, install_stderr="npm failed")

    with pytest.raises(RuntimeError, match="failed to install Claude Code with npm: npm failed"):
        asyncio.run(_agent()._ensure_claude(sandbox))


@pytest.mark.cpu
@pytest.mark.level0
def test_ensure_claude_surfaces_native_install_failure():
    sandbox = _FakeSandbox(
        probe_results=[1],
        npm_available=False,
        install_exit_code=1,
        install_stderr="curl failed",
    )

    with pytest.raises(RuntimeError, match="failed to install Claude Code with native installer: curl failed"):
        asyncio.run(_agent()._ensure_claude(sandbox))


@pytest.mark.cpu
@pytest.mark.level0
def test_ensure_claude_requires_binary_on_path_after_install():
    sandbox = _FakeSandbox(probe_results=[1, 1])

    with pytest.raises(RuntimeError, match="not available on PATH"):
        asyncio.run(_agent()._ensure_claude(sandbox))


@pytest.mark.cpu
@pytest.mark.level0
def test_run_forwards_workdir():
    config = ClaudeCodeConfig(
        model=ModelConfig(
            base_url="https://ark.example/api/compatible",
            api_key="ark-test-api-key",
            model_name="policy",
        ),
        run_timeout=123.0,
    )
    sandbox = _FakeSandbox(probe_results=[0])

    result = asyncio.run(
        ClaudeCodeAgent(config).run(
            sandbox=sandbox,
            messages=[
                {"role": "system", "content": "ignored system prompt"},
                {"role": "user", "content": "fix the bug"},
            ],
            workdir="/testbed",
        )
    )

    assert result.finished is True
    assert len(sandbox.exec_calls) == 1
    assert sandbox.exec_calls[0]["workdir"] == "/testbed"
    assert sandbox.exec_calls[0]["timeout"] == 123.0
    argv = sandbox.exec_calls[0]["argv"]
    assert argv[:2] == ["claude", "-p"]
    assert argv[2] == "fix the bug"
    assert argv[argv.index("--model") + 1] == "policy"
    assert argv[argv.index("--permission-mode") + 1] == "bypassPermissions"
    assert "--bare" not in argv
    assert "--no-session-persistence" not in argv
    assert "--disable-slash-commands" in argv
    assert "--append-system-prompt" not in argv
    assert "--dangerously-skip-permissions" not in argv
    disallowed_tools = argv[argv.index("--disallowedTools") + 1].split(",")
    assert set(disallowed_tools) == {"Agent", "Task", "WebFetch", "WebSearch", "AskUserQuestion"}
    assert sandbox.exec_calls[0]["env"]["ANTHROPIC_BASE_URL"] == "https://ark.example/api/compatible"
    assert "ANTHROPIC_API_KEY" not in sandbox.exec_calls[0]["env"]
    assert sandbox.exec_calls[0]["env"]["ANTHROPIC_AUTH_TOKEN"] == "ark-test-api-key"
    assert sandbox.exec_calls[0]["env"]["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] == "1"
    assert sandbox.exec_calls[0]["env"]["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
    assert sandbox.exec_calls[0]["env"]["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"
    assert sandbox.exec_calls[0]["env"]["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] == "1"
    assert sandbox.exec_calls[0]["env"]["CLAUDE_CODE_DISABLE_TERMINAL_TITLE"] == "1"


@pytest.mark.cpu
@pytest.mark.level0
def test_run_accepts_user_only_message_without_rewriting():
    config = ClaudeCodeConfig(
        model=ModelConfig(base_url="http://gateway:8000/v1", model_name="policy"),
    )
    sandbox = _FakeSandbox(probe_results=[0])
    prompt = (
        "Inspect the repository in /testbed and resolve the following issue:\n\n"
        "<issue_description>\nfix the bug\n</issue_description>\n\n"
        "Run the relevant tests before finishing."
    )

    asyncio.run(
        ClaudeCodeAgent(config).run(
            sandbox=sandbox,
            messages=[{"role": "user", "content": prompt}],
        )
    )

    assert sandbox.exec_calls[0]["argv"][2] == prompt


@pytest.mark.cpu
@pytest.mark.level0
def test_run_requires_exactly_one_user_message():
    config = ClaudeCodeConfig(
        model=ModelConfig(base_url="http://gateway:8000/v1", model_name="policy"),
    )
    sandbox = _FakeSandbox(probe_results=[])

    with pytest.raises(ValueError, match="exactly one 'user' message"):
        asyncio.run(
            ClaudeCodeAgent(config).run(
                sandbox=sandbox,
                messages=[{"role": "system", "content": "system prompt"}],
            )
        )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("enable_web_tools", "enable_subagents", "expected_disallowed"),
    [
        (True, False, {"Agent", "Task", "AskUserQuestion"}),
        (False, True, {"WebFetch", "WebSearch", "AskUserQuestion"}),
        (True, True, {"AskUserQuestion"}),
    ],
)
def test_claude_argv_controls_web_tools_and_subagents_independently(
    enable_web_tools,
    enable_subagents,
    expected_disallowed,
):
    config = ClaudeCodeConfig(
        model=ModelConfig(model_name="policy"),
        enable_web_tools=enable_web_tools,
        enable_subagents=enable_subagents,
    )

    argv = ClaudeCodeAgent(config)._claude_argv("fix the bug")

    disallowed_tools = argv[argv.index("--disallowedTools") + 1].split(",")
    assert set(disallowed_tools) == expected_disallowed


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(("enable_subagents", "expected_model"), [(False, None), (True, "policy")])
def test_claude_env_pins_enabled_subagents_to_policy_model(enable_subagents, expected_model):
    config = ClaudeCodeConfig(
        model=ModelConfig(model_name="policy"),
        enable_subagents=enable_subagents,
    )

    env = ClaudeCodeAgent(config)._claude_env("http://gateway:8000")

    assert env.get("CLAUDE_CODE_SUBAGENT_MODEL") == expected_model


@pytest.mark.cpu
@pytest.mark.level0
def test_claude_argv_can_keep_slash_commands_enabled():
    config = ClaudeCodeConfig(
        model=ModelConfig(model_name="policy"),
        disable_slash_commands=False,
    )

    argv = ClaudeCodeAgent(config)._claude_argv("fix the bug")

    assert "--disable-slash-commands" not in argv


@pytest.mark.cpu
@pytest.mark.level0
def test_run_reports_nonzero_process_exit():
    config = ClaudeCodeConfig(
        model=ModelConfig(base_url="http://gateway:8000/v1", model_name="policy"),
    )
    sandbox = _FakeSandbox(probe_results=[0], process_exit_code=2)

    result = asyncio.run(
        ClaudeCodeAgent(config).run(
            sandbox=sandbox,
            messages=[
                {"role": "system", "content": "ignored system prompt"},
                {"role": "user", "content": "fix the bug"},
            ],
        )
    )

    assert result.finished is False
    assert result.info["exit_code"] == 2


@pytest.mark.cpu
@pytest.mark.level0
def test_claude_env_uses_placeholders_for_session_gateway():
    config = ClaudeCodeConfig(model=ModelConfig(base_url="http://gateway:8000/v1", model_name="policy"))

    env = ClaudeCodeAgent(config)._claude_env("http://gateway:8000")

    assert "ANTHROPIC_API_KEY" not in env
    assert env["ANTHROPIC_AUTH_TOKEN"]
    assert env["ANTHROPIC_AUTH_TOKEN"] != "EMPTY"
