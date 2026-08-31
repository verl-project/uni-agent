from __future__ import annotations

import pytest

import uni_agent.agents.react.agent as react_module
from uni_agent.agents.base import AgentResult, ModelConfig
from uni_agent.agents.react.agent import ReActAgent, ReActConfig
from uni_agent.agents.react.model import OpenAICompatibleChatModel
from uni_agent.tools import ToolResult


class _FakeToolbox:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def schemas(self) -> list[dict]:
        return []

    def names(self) -> list[str]:
        return ["shell"]

    async def call(self, name, args, *, timeout=None):
        self.calls.append((name, args))
        return ToolResult(text="ok")

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


@pytest.mark.cpu
@pytest.mark.level0
def test_agent_result_defaults_to_unreported_completion():
    assert AgentResult().finished is None


@pytest.mark.cpu
@pytest.mark.level0
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


@pytest.mark.cpu
@pytest.mark.level0
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
    toolbox = _FakeToolbox()
    monkeypatch.setattr(react_module.Toolbox, "from_specs", lambda specs, *, sandbox: toolbox)
    monkeypatch.setattr(react_module, "OpenAICompatibleChatModel", _FakeModel)

    agent = _agent()

    async def stop_with_reason(*args, **kwargs):
        return termination_reason

    monkeypatch.setattr(agent, "step", stop_with_reason)

    result = await agent.run(sandbox=object(), messages=[], workdir="/testbed")

    assert result.finished is expected
    assert toolbox.calls == [("shell", {"command": "cd -- /testbed"})]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_length_finish_reason_is_unfinished():
    agent = _agent()
    cfg: ReActConfig = agent.config  # type: ignore[assignment]

    reason = await agent.step(cfg, _StepModel(finish_reason="length"), _StepToolbox(), [], _step_info())

    assert reason == "token_limit"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_length_exhausted_empty_response_ends_the_episode():
    # What the Gateway returns once the session's response budget is spent: an
    # empty assistant message, finish_reason=length, zero completion tokens.
    agent = _agent()
    cfg: ReActConfig = agent.config  # type: ignore[assignment]
    model = _StepModel(finish_reason="length", completion_tokens=0)

    reason = await agent.step(cfg, model, _StepToolbox(), [], _step_info())

    assert reason == "token_limit"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_total_tokens_tracks_current_context_size():
    agent = _agent()
    cfg: ReActConfig = agent.config  # type: ignore[assignment]
    step_model = _StepModel(prompt_tokens=100, completion_tokens=2)
    info = _step_info(total_tokens=3)

    reason = await agent.step(cfg, step_model, _StepToolbox(), [], info)

    assert reason == "finished"
    assert info["total_tokens"] == 102


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_plain_text_without_tool_call_finishes_episode():
    agent = _agent()
    cfg: ReActConfig = agent.config  # type: ignore[assignment]
    transcript: list[dict] = []

    reason = await agent.step(cfg, _StepModel(), _StepToolbox(), transcript, _step_info())

    assert reason == "finished"
    assert [message["role"] for message in transcript] == ["assistant"]


@pytest.mark.cpu
@pytest.mark.level0
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


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_truncated_tool_call_is_not_dispatched():
    # The Gateway reports tool_calls, never length, whenever it parsed a tool call
    # out of the response; a third-party endpoint may report both, and a tool call
    # decoded from a cut-off response can carry half-written arguments.
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


def _per_turn_capped_agent(*, max_tokens_per_turn: int, max_total_tokens: int | None = None) -> ReActAgent:
    model = ModelConfig(
        base_url="http://gateway:8000/v1",
        model_name="policy",
        max_tokens_per_turn=max_tokens_per_turn,
        max_total_tokens=max_total_tokens,
    )
    return ReActAgent(ReActConfig(model=model, tools=[], max_steps=2))


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_per_turn_cap_truncation_ends_the_episode():
    agent = _per_turn_capped_agent(max_tokens_per_turn=8)
    cfg: ReActConfig = agent.config  # type: ignore[assignment]
    model = _StepModel(finish_reason="length", completion_tokens=8)

    reason = await agent.step(cfg, model, _StepToolbox(), [], _step_info())

    assert model.sampling_params["max_tokens"] == 8
    assert reason == "token_limit"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_max_tokens_is_clamped_to_the_remaining_episode_budget():
    agent = _per_turn_capped_agent(max_tokens_per_turn=8, max_total_tokens=105)
    cfg: ReActConfig = agent.config  # type: ignore[assignment]
    # remaining episode budget (5) is below the per-turn cap (8), so the turn asks
    # for the smaller of the two.
    model = _StepModel(finish_reason="length", completion_tokens=5, prompt_tokens=99)

    reason = await agent.step(cfg, model, _StepToolbox(), [], _step_info(total_tokens=100))

    assert model.sampling_params["max_tokens"] == 5
    assert reason == "token_limit"
