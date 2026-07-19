from __future__ import annotations

import asyncio

import pytest

from uni_agent.agents.base import ModelConfig
from uni_agent.agents.claude_code.agent import ClaudeCodeAgent, ClaudeCodeConfig
from uni_agent.sandbox.base import ExecResult


class _FakeSandbox:
    def __init__(self, *, probe_results: list[int], install_exit_code: int = 0):
        self.probe_results = list(probe_results)
        self.install_exit_code = install_exit_code
        self.calls: list[dict] = []
        self.exec_calls: list[dict] = []

    async def exec_shell(self, script: str, *, timeout=None, workdir=None, env=None) -> ExecResult:
        self.calls.append({"script": script, "timeout": timeout})
        if script.startswith("command -v claude"):
            return ExecResult(exit_code=self.probe_results.pop(0), stdout="", stderr="")
        stderr = "npm failed" if self.install_exit_code else ""
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
        "npm install -g @anthropic-ai/claude-code --no-audit --no-fund",
        "command -v claude >/dev/null 2>&1",
    ]
    assert sandbox.calls[1]["timeout"] == 600


def test_ensure_claude_surfaces_install_failure():
    sandbox = _FakeSandbox(probe_results=[1], install_exit_code=1)

    with pytest.raises(RuntimeError, match="failed to install Claude Code: npm failed"):
        asyncio.run(_agent()._ensure_claude(sandbox))


def test_ensure_claude_requires_binary_on_path_after_install():
    sandbox = _FakeSandbox(probe_results=[1, 1])

    with pytest.raises(RuntimeError, match="not available on PATH"):
        asyncio.run(_agent()._ensure_claude(sandbox))


def test_run_uses_sandbox_default_workdir():
    config = ClaudeCodeConfig(model=ModelConfig(base_url="http://gateway:8000/v1", model_name="policy"))
    sandbox = _FakeSandbox(probe_results=[0])

    asyncio.run(
        ClaudeCodeAgent(config).run(
            sandbox=sandbox,
            messages=[{"role": "user", "content": "fix the bug"}],
        )
    )

    assert len(sandbox.exec_calls) == 1
    assert sandbox.exec_calls[0]["workdir"] is None
