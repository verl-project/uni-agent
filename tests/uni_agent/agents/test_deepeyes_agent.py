from __future__ import annotations

import asyncio

from PIL import Image

import uni_agent.agents.deepeyes.agent as agent_module
from uni_agent.agents import ModelConfig, get_agent_cls
from uni_agent.agents.deepeyes import DeepEyesAgent, DeepEyesAgentConfig, ImageZoomInTool


class _FakeModel:
    instances: list[_FakeModel] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.responses = [
            (
                "",
                [
                    {
                        "id": "zoom-1",
                        "type": "function",
                        "function": {
                            "name": "image_zoom_in_tool",
                            "arguments": '{"bbox_2d": [0, 0, 500, 500], "label": "object"}',
                        },
                    }
                ],
                {"prompt_tokens": 20, "completion_tokens": 5, "finish_reason": "tool_calls"},
            ),
            (
                "<answer>cat</answer>",
                [],
                {"prompt_tokens": 30, "completion_tokens": 4, "finish_reason": "stop"},
            ),
        ]
        self.instances.append(self)

    async def query(self, messages, *, sampling_params=None):
        self.last_sampling_params = sampling_params
        return self.responses.pop(0)

    async def aclose(self) -> None:
        self.closed = True


def test_deepeyes_agent_is_registered():
    assert get_agent_cls("deepeyes") is DeepEyesAgent
    assert DeepEyesAgent.config_model is DeepEyesAgentConfig


def test_image_zoom_tool_crops_normalized_bbox():
    tool = ImageZoomInTool(object(), image=Image.new("RGB", (100, 80), "white"))

    result = asyncio.run(tool.run({"bbox_2d": [0, 0, 500, 500], "label": "corner"}))

    assert result.status == "ok"
    assert result.info["success"] is True
    assert result.info["bbox"] == (0, 0, 50, 40)
    assert result.images[0].size == (50, 40)


def test_image_zoom_tool_reports_invalid_bbox():
    tool = ImageZoomInTool(object(), image=Image.new("RGB", (100, 80), "white"))

    result = asyncio.run(tool.run({"bbox_2d": [500, 500, 100, 100]}))

    assert result.status == "error"
    assert result.info == {"success": False, "error": "invalid_bbox"}


def test_deepeyes_agent_round_trips_crop_and_closes_model(monkeypatch):
    _FakeModel.instances.clear()
    monkeypatch.setattr(agent_module, "OpenAICompatibleChatModel", _FakeModel)
    agent = DeepEyesAgent(
        DeepEyesAgentConfig(
            model=ModelConfig(base_url="http://gateway/v1", model_name="policy"),
            max_turns=4,
        )
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (100, 80), "white")},
                {"type": "text", "text": "What is shown?"},
            ],
        }
    ]

    result = asyncio.run(agent.run(sandbox=object(), messages=messages, workdir="/testbed"))

    assert result.finished is True
    assert result.output == {"final_answer": "<answer>cat</answer>", "termination_reason": "finished"}
    assert result.info["steps"] == 2
    assert result.info["tool_calls"] == 1
    assert result.info["tool_successes"] == 1
    assert result.info["tool_errors"] == 0
    assert result.info["prompt_tokens"] == 50
    assert result.info["completion_tokens"] == 9
    assert [message["role"] for message in result.transcript] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result.transcript[0]["content"] == agent_module.DEFAULT_SYSTEM_PROMPT
    tool_content = result.transcript[3]["content"]
    assert any(part.get("type") == "image_url" for part in tool_content)
    assert _FakeModel.instances[0].kwargs["sampling_params"]["tool_choice"] == "auto"
    assert _FakeModel.instances[0].closed is True


def test_deepeyes_agent_enforces_per_turn_and_total_token_budgets(monkeypatch):
    _FakeModel.instances.clear()
    monkeypatch.setattr(agent_module, "OpenAICompatibleChatModel", _FakeModel)
    agent = DeepEyesAgent(
        DeepEyesAgentConfig(
            model=ModelConfig(
                base_url="http://gateway/v1",
                max_tokens_per_turn=8,
                max_total_tokens=10,
            ),
            request_max_retries=0,
        )
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (100, 80), "white")},
                {"type": "text", "text": "What is shown?"},
            ],
        }
    ]

    result = asyncio.run(agent.run(sandbox=object(), messages=messages))

    model = _FakeModel.instances[0]
    assert model.kwargs["sampling_params"]["max_tokens"] == 8
    assert model.kwargs["max_retries"] == 0
    # The first fake completion consumes five tokens, leaving five for turn two.
    assert model.last_sampling_params["max_tokens"] == 5
    assert result.info["completion_tokens"] == 9
