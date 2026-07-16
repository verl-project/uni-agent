import json

import pytest

ALLOWED_SAMPLING_KEYS = frozenset({"temperature", "top_p", "top_k", "max_tokens", "stop"})


def test_openai_build_response_shape():
    """A completed internal outcome serializes to the basic OpenAI chat
    completion envelope with ids, model, choices, and token usage."""
    from uni_agent.gateway.adapters.openai import openai_build_response
    from uni_agent.gateway.session.session import GenerationOutcome

    outcome = GenerationOutcome(
        assistant_msg={"role": "assistant", "content": "hello"},
        finish_reason="stop",
        prompt_tokens=3,
        completion_tokens=2,
    )
    body = openai_build_response(outcome, model="m")
    assert body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert isinstance(body["created"], int)
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == 5
    assert body["model"] == "m"


@pytest.mark.asyncio
async def test_openai_stream_response_emits_compatible_sse_chunks():
    """A completed internal outcome is synthesized into OpenAI-compatible SSE
    chunks for role, reasoning, content, tool calls, final usage, and DONE."""
    from uni_agent.gateway.adapters.openai import openai_stream_response
    from uni_agent.gateway.session.session import GenerationOutcome

    resp = openai_stream_response(
        GenerationOutcome(
            assistant_msg={
                "role": "assistant",
                "reasoning_content": "thinking",
                "content": "hello",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"a":1}'},
                    }
                ],
            },
            finish_reason="tool_calls",
            prompt_tokens=3,
            completion_tokens=2,
        ),
        model="m",
    )

    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["connection"] == "keep-alive"
    assert resp.headers["x-accel-buffering"] == "no"

    text = (b"".join([chunk async for chunk in resp.body_iterator])).decode()
    data_lines = [line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: ")]
    assert data_lines[-1] == "[DONE]"
    chunks = [json.loads(line) for line in data_lines[:-1]]

    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[1]["choices"][0]["delta"] == {"reasoning_content": "thinking"}
    assert chunks[2]["choices"][0]["delta"] == {"content": "hello"}
    assert chunks[3]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert chunks[3]["choices"][0]["delta"]["tool_calls"][0]["id"] == "call-1"
    assert chunks[4]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[4]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }


def test_openai_to_internal_normalizes_messages_sampling_and_tools():
    """OpenAI wire requests lower to the internal shape: sampling params use the
    gateway allowlist, JSON tool arguments are parsed, malformed argument
    strings are preserved, and tool_choice=none clears tool schemas."""
    from uni_agent.gateway.adapters.openai import openai_to_internal

    payload = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "x", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}},
                    {"id": "y", "type": "function", "function": {"name": "g", "arguments": "not json"}},
                ],
            },
        ],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "max_tokens": 32,
        "temperature": 0.7,
        "stop": ["</s>"],
        "chat_template_kwargs": {"enable_thinking": True},
        "ignored_field": 1,
    }
    req = openai_to_internal(
        payload,
        base_sampling_params={"top_p": 0.9},
        allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
    )
    assert set(req) == {"messages", "tools", "sampling_params"}
    assert req["messages"][0] == {"role": "user", "content": "hi"}
    assert req["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"x": 1}
    assert req["messages"][1]["tool_calls"][1]["function"]["arguments"] == "not json"
    assert req["tools"][0]["function"]["name"] == "f"
    assert req["sampling_params"]["max_tokens"] == 32
    assert req["sampling_params"]["temperature"] == 0.7
    assert req["sampling_params"]["top_p"] == 0.9
    assert req["sampling_params"]["stop"] == ["</s>"]
    assert "ignored_field" not in req["sampling_params"]

    without_tools = openai_to_internal(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "tool_choice": "none",
        },
        base_sampling_params={},
        allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
    )
    assert without_tools["tools"] is None
