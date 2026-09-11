from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from tests.uni_agent.support import FakeTokenizer, QueuedBackend
from uni_agent.gateway.adapters.responses import (
    responses_build_response,
    responses_stream_response,
)
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.gateway import _GatewayActor
from uni_agent.gateway.session.codec import MessageCodec

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def _outcome(*, finish_reason: str, content: str = "partial", tool_calls=None):
    return SimpleNamespace(
        assistant_msg={
            "role": "assistant",
            "content": content,
            **({"tool_calls": tool_calls} if tool_calls is not None else {}),
        },
        finish_reason=finish_reason,
        prompt_tokens=4,
        completion_tokens=len(content),
    )


def test_responses_stop_is_completed():
    response = responses_build_response(_outcome(finish_reason="stop"), model="policy")

    assert response["status"] == "completed"
    assert "incomplete_details" not in response
    assert response["output"][0]["status"] == "completed"


def test_responses_length_is_incomplete_and_marks_output_items():
    response = responses_build_response(_outcome(finish_reason="length"), model="policy")

    assert response["status"] == "incomplete"
    assert response["incomplete_details"] == {"reason": "max_output_tokens"}
    assert response["output"][0]["status"] == "incomplete"


@pytest.mark.asyncio
async def test_responses_sse_ends_with_incomplete_event_for_length():
    response = responses_stream_response(_outcome(finish_reason="length"), model="policy")
    chunks = [chunk async for chunk in response.body_iterator]
    body = b"".join(chunks)

    assert b"event: response.incomplete\n" in body
    assert b"event: response.completed\n" not in body
    assert b'"status": "incomplete"' in body
    assert b'"reason": "max_output_tokens"' in body


def test_responses_complete_function_call_remains_completed():
    response = responses_build_response(
        _outcome(
            finish_reason="tool_calls",
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "exec_command", "arguments": "{}"},
                }
            ],
        ),
        model="policy",
    )

    assert response["status"] == "completed"
    assert response["output"][0]["type"] == "function_call"
    assert response["output"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_responses_http_routes_match_json_and_sse_terminal_statuses():
    actor = _GatewayActor(
        config=GatewayActorConfig(tokenizer=FakeTokenizer()),
        backend=QueuedBackend(["ANSWER", "ANSWER"]),
    )
    actor._server_base_url = "http://gateway.local"
    await actor.create_session("json")
    await actor.create_session("sse")
    transport = httpx.ASGITransport(app=actor._app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway.local") as client:
        json_response = await client.post(
            "/sessions/json/v1/responses",
            json={"model": "policy", "input": "inspect"},
        )
        sse_response = await client.post(
            "/sessions/sse/v1/responses",
            json={"model": "policy", "input": "inspect", "stream": True},
        )

    assert json_response.status_code == 200
    assert json_response.json()["status"] == "completed"
    assert sse_response.status_code == 200
    assert "event: response.completed" in sse_response.text
    assert "event: response.incomplete" not in sse_response.text


@pytest.mark.asyncio
async def test_codec_does_not_turn_length_limited_tool_text_into_tool_call(monkeypatch):
    codec = MessageCodec(FakeTokenizer())

    async def unexpected_parser(*_args, **_kwargs):
        raise AssertionError("tool parser must not run for a length-limited response")

    monkeypatch.setattr(codec, "_extract_tool_calls", unexpected_parser)
    message, finish_reason = await codec.decode_response(
        [ord(char) for char in "partial tool call"],
        tools=[{"type": "function", "function": {"name": "exec_command", "parameters": {}}}],
        stop_reason="length",
    )

    assert finish_reason == "length"
    assert message["content"] == "partial tool call"
    assert "tool_calls" not in message
