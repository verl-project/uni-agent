from __future__ import annotations

import pytest

from uni_agent.agents.base import AgentResult
from uni_agent.sandbox.base import ExecResult
from uni_agent.tasks import get_task
from uni_agent.tasks.gui.task import GuiTask, GuiTaskConfig


class _Sandbox:
    def __init__(self, exit_codes: dict[str, int]):
        self.exit_codes = exit_codes
        self.scripts: list[tuple[str, dict | None]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def exec_shell(self, script, *, timeout=None, workdir=None, env=None):
        self.scripts.append((script, env))
        return ExecResult(exit_code=self.exit_codes.get(script, 0), stdout="", stderr=f"err:{script}")


class _Agent:
    def __init__(self):
        self.messages = None

    async def run(self, *, sandbox, messages, workdir=None):
        self.messages = messages
        return AgentResult(finished=True, info={"steps": 3})


def _task(monkeypatch, sandbox, agent, **overrides) -> GuiTask:
    config = GuiTaskConfig(
        setup_command="launch-app",
        verify_command="check-result",
        sandbox={"provider": "local"},
        metadata={"instruction": "Click the button."},
        display=":5",
        **overrides,
    )
    task = GuiTask(config)
    monkeypatch.setattr(task, "build_sandbox", lambda: sandbox)
    monkeypatch.setattr(task, "build_agent", lambda: agent)
    return task


@pytest.mark.cpu
@pytest.mark.level0
def test_gui_task_is_registered_and_requires_a_verify_command():
    task = get_task({"name": "gui", "verify_command": "true", "sandbox": {"provider": "local"}})

    assert isinstance(task, GuiTask)
    with pytest.raises(ValueError):
        GuiTaskConfig(sandbox={"provider": "local"})


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize(("verify_exit", "reward"), [(0, 1.0), (1, 0.0)])
async def test_reward_follows_verify_exit_code(monkeypatch, verify_exit, reward):
    sandbox = _Sandbox({"check-result": verify_exit})
    agent = _Agent()

    result = await _task(monkeypatch, sandbox, agent).run()

    assert result.reward == reward
    assert result.finished is True
    assert result.extra_info["verify_exit_code"] == verify_exit
    assert result.extra_info["steps"] == 3
    # setup runs before the agent, verify after, both on the configured display
    assert [script for script, _ in sandbox.scripts] == ["launch-app", "check-result"]
    assert all(env == {"DISPLAY": ":5"} for _, env in sandbox.scripts)
    assert agent.messages == [{"role": "user", "content": "Click the button."}]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_failed_setup_raises_before_the_agent_runs(monkeypatch):
    sandbox = _Sandbox({"launch-app": 2})
    agent = _Agent()

    with pytest.raises(RuntimeError, match="setup_command failed"):
        await _task(monkeypatch, sandbox, agent).run()

    assert agent.messages is None


def _config_with_tools(tools, **kwargs):
    return GuiTaskConfig(
        verify_command="true",
        sandbox={"provider": "local"},
        agent={"name": "react", "tools": tools},
        **kwargs,
    )


@pytest.mark.cpu
@pytest.mark.level0
def test_task_display_is_propagated_to_gui_tools_only():
    config = _config_with_tools(
        [{"name": "gui_click"}, {"name": "finish"}, {"name": "gui_type", "display": ":9"}], display=":9"
    )

    assert config.agent.tools == [
        {"name": "gui_click", "display": ":9"},
        {"name": "finish"},
        {"name": "gui_type", "display": ":9"},
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_conflicting_tool_display_is_rejected():
    with pytest.raises(ValueError, match="uses display ':1' but the task display is ':9'"):
        _config_with_tools([{"name": "gui_click", "display": ":1"}], display=":9")
