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
    ):
        self.probe_results = list(probe_results)
        self.npm_available = npm_available
        self.install_exit_code = install_exit_code
        self.install_stderr = install_stderr
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
        return ExecResult(exit_code=0, stdout="done", stderr="")


def _agent() -> ClaudeCodeAgent:
    return ClaudeCodeAgent(ClaudeCodeConfig())


def test_ensure_claude_skips_install_when_already_available():
    sandbox = _FakeSandbox(probe_results=[0])

    asyncio.run(_agent()._ensure_claude(sandbox))

    assert len(sandbox.calls) == 1
    assert sandbox.calls[0]["script"].startswith("command -v claude")


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


def test_ensure_claude_surfaces_install_failure():
    sandbox = _FakeSandbox(probe_results=[1], install_exit_code=1, install_stderr="npm failed")

    with pytest.raises(RuntimeError, match="failed to install Claude Code with npm: npm failed"):
        asyncio.run(_agent()._ensure_claude(sandbox))


def test_ensure_claude_surfaces_native_install_failure():
    sandbox = _FakeSandbox(
        probe_results=[1],
        npm_available=False,
        install_exit_code=1,
        install_stderr="curl failed",
    )

    with pytest.raises(RuntimeError, match="failed to install Claude Code with native installer: curl failed"):
        asyncio.run(_agent()._ensure_claude(sandbox))


def test_ensure_claude_requires_binary_on_path_after_install():
    sandbox = _FakeSandbox(probe_results=[1, 1])

    with pytest.raises(RuntimeError, match="not available on PATH"):
        asyncio.run(_agent()._ensure_claude(sandbox))


def test_run_uses_sandbox_default_workdir():
    config = ClaudeCodeConfig(
        model=ModelConfig(
            base_url="https://ark.example/api/compatible",
            api_key="ark-test-api-key",
            model_name="policy",
        )
    )
    sandbox = _FakeSandbox(probe_results=[0])

    asyncio.run(
        ClaudeCodeAgent(config).run(
            sandbox=sandbox,
            messages=[{"role": "user", "content": "fix the bug"}],
        )
    )

    assert len(sandbox.exec_calls) == 1
    assert sandbox.exec_calls[0]["workdir"] is None
    assert sandbox.exec_calls[0]["env"]["ANTHROPIC_BASE_URL"] == "https://ark.example/api/compatible"
    assert sandbox.exec_calls[0]["env"]["ANTHROPIC_API_KEY"] == ""
    assert sandbox.exec_calls[0]["env"]["ANTHROPIC_AUTH_TOKEN"] == "ark-test-api-key"


def test_claude_env_uses_placeholders_for_session_gateway():
    config = ClaudeCodeConfig(model=ModelConfig(base_url="http://gateway:8000/v1", model_name="policy"))

    env = ClaudeCodeAgent(config)._claude_env("http://gateway:8000")

    assert env["ANTHROPIC_API_KEY"] == "sk-ant-uni-agent-placeholder"
    assert env["ANTHROPIC_AUTH_TOKEN"]
    assert env["ANTHROPIC_AUTH_TOKEN"] != "EMPTY"
