from __future__ import annotations

import pytest

import uni_agent.agents.react.agent as react_module
from uni_agent.agents.base import AgentResult, ModelConfig
from uni_agent.agents.react.agent import ReActAgent, ReActConfig
from uni_agent.agents.react.model import OpenAICompatibleChatModel
from uni_agent.tools import ToolResult


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


class _StepModel:
    def __init__(
        self,
        *,
        tool_calls: list[dict] | None = None,
        finish_reason: str = "stop",
        prompt_tokens: int = 100,
        completion_tokens: int = 1,
    ):
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.sampling_params: dict | None = None

    async def query(self, messages, *, sampling_params):
        self.sampling_params = sampling_params
        return (
            "answer",
            self.tool_calls,
            {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "finish_reason": self.finish_reason,
            },
        )


class _StepToolbox:
    def __init__(self, result: ToolResult | None = None):
        self.result = result or ToolResult(text="ok")
        self.calls: list[tuple[str, object]] = []

    async def call(self, name, args, *, timeout=None):
        self.calls.append((name, args))
        return self.result


def _step_info(*, total_tokens: int = 0) -> dict:
    return {
        "steps": 1,
        "num_tool_calls": 0,
        "timeouts": 0,
        "errors": 0,
        "total_tokens": total_tokens,
    }


def _agent() -> ReActAgent:
    model = ModelConfig(base_url="http://gateway:8000/v1", model_name="policy")
    return ReActAgent(ReActConfig(model=model, tools=[], max_steps=1))


def test_agent_result_defaults_to_unfinished():
    assert AgentResult().finished is False


@pytest.mark.asyncio
async def test_model_forwards_finish_reason(monkeypatch):
    model = OpenAICompatibleChatModel(
        base_url="http://gateway:8000/v1",
        model_name="policy",
    )

    async def fake_completion(body):
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "truncated"},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }

    monkeypatch.setattr(model, "_post_chat_completion", fake_completion)

    _, _, generation_info = await model.query([{"role": "user", "content": "test"}])

    assert generation_info["finish_reason"] == "length"


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


@pytest.mark.asyncio
async def test_length_finish_reason_is_unfinished():
    agent = _agent()
    cfg: ReActConfig = agent.config  # type: ignore[assignment]

    reason = await agent.step(cfg, _StepModel(finish_reason="length"), _StepToolbox(), [], _step_info())

    assert reason == "token_limit"


@pytest.mark.asyncio
async def test_total_tokens_tracks_current_context_size():
    agent = _agent()
    cfg: ReActConfig = agent.config  # type: ignore[assignment]
    step_model = _StepModel(prompt_tokens=100, completion_tokens=2)
    info = _step_info(total_tokens=3)

    reason = await agent.step(cfg, step_model, _StepToolbox(), [], info)

    assert reason == "finished"
    assert info["total_tokens"] == 102


@pytest.mark.asyncio
async def test_failed_finish_tool_does_not_finish():
    agent = _agent()
    cfg: ReActConfig = agent.config  # type: ignore[assignment]
    model = _StepModel(
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "finish", "arguments": "{}"},
            }
        ],
        finish_reason="tool_calls",
    )

    info = _step_info()
    reason = await agent.step(
        cfg,
        model,
        _StepToolbox(ToolResult(text="invalid", status="format_error")),
        [],
        info,
    )

    assert reason == "completed"
    assert info["errors"] == 1


@pytest.mark.asyncio
async def test_truncated_tool_call_is_not_dispatched():
    agent = _agent()
    cfg: ReActConfig = agent.config  # type: ignore[assignment]
    toolbox = _StepToolbox()
    model = _StepModel(
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "finish", "arguments": "{}"},
            }
        ],
        finish_reason="length",
    )

    reason = await agent.step(cfg, model, toolbox, [], _step_info())

    assert reason == "token_limit"
    assert toolbox.calls == []
