from __future__ import annotations

import asyncio

import pytest

from uni_agent.sandbox.base import ExecResult, SandboxConfig
from uni_agent.sandbox.registry import build_sandbox


@pytest.mark.cpu
@pytest.mark.level0
def test_startup_commands_run_after_start_and_before_sandbox_is_returned(monkeypatch):
    sandbox = build_sandbox(
        SandboxConfig(
            provider="local",
            startup_commands=["first command", "second command"],
        )
    )
    events: list[str] = []

    async def start() -> None:
        events.append("start")

    async def stop() -> None:
        events.append("stop")

    async def exec_shell(command: str, **kwargs) -> ExecResult:
        events.append(command)
        return ExecResult(exit_code=0, stdout="", stderr="")

    monkeypatch.setattr(sandbox, "start", start)
    monkeypatch.setattr(sandbox, "stop", stop)
    monkeypatch.setattr(sandbox, "exec_shell", exec_shell)

    async def run() -> None:
        async with sandbox:
            events.append("ready")

    asyncio.run(run())

    assert events == ["start", "first command", "second command", "ready", "stop"]


@pytest.mark.cpu
@pytest.mark.level0
def test_failed_startup_command_fails_start_and_cleans_up(monkeypatch):
    sandbox = build_sandbox(
        SandboxConfig(provider="local", startup_commands=["failing command"])
    )
    stopped = False

    async def start() -> None:
        pass

    async def stop() -> None:
        nonlocal stopped
        stopped = True

    async def exec_shell(command: str, **kwargs) -> ExecResult:
        return ExecResult(exit_code=2, stdout="", stderr="link failed")

    monkeypatch.setattr(sandbox, "start", start)
    monkeypatch.setattr(sandbox, "stop", stop)
    monkeypatch.setattr(sandbox, "exec_shell", exec_shell)

    with pytest.raises(RuntimeError, match="startup command 1/1 failed: link failed"):
        asyncio.run(sandbox.__aenter__(retry=1))

    assert stopped is True


@pytest.mark.cpu
@pytest.mark.level0
def test_startup_commands_reject_empty_values():
    with pytest.raises(ValueError, match="startup_commands must contain only non-empty commands"):
        SandboxConfig(provider="local", startup_commands=["  "])
