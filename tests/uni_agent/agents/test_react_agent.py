from __future__ import annotations

import pytest

import uni_agent.agents.react.agent as react_module
from uni_agent.agents.base import AgentResult, ModelConfig
from uni_agent.agents.react.agent import ReActAgent, ReActConfig


class _FakeToolbox:
    def schemas(self) -> list[dict]:
        return []

    def entered(self, *, retry: int, timeout: float):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeModel:
    def __init__(self, **kwargs):
        pass

    async def aclose(self) -> None:
        pass


def _agent() -> ReActAgent:
    model = ModelConfig(base_url="http://gateway:8000/v1", model_name="policy")
    return ReActAgent(ReActConfig(model=model, tools=[], max_steps=1))


def test_agent_result_defaults_to_unfinished():
    assert AgentResult().finished is False


@pytest.mark.parametrize(
    ("termination_reason", "expected"),
    [
        ("finished", True),
        ("token_limit", False),
        ("timeout_limit", False),
    ],
)
@pytest.mark.asyncio
async def test_react_reports_completion(monkeypatch, termination_reason: str, expected: bool):
    monkeypatch.setattr(react_module.Toolbox, "from_specs", lambda specs, *, sandbox: _FakeToolbox())
    monkeypatch.setattr(react_module, "OpenAICompatibleChatModel", _FakeModel)

    agent = _agent()

    async def stop_with_reason(*args, **kwargs):
        return termination_reason

    monkeypatch.setattr(agent, "step", stop_with_reason)

    result = await agent.run(sandbox=object(), messages=[])

    assert result.finished is expected
