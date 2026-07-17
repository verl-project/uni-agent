import json

import pytest

from uni_agent.gateway.adapters.anthropic import anthropic_to_internal
from uni_agent.gateway.adapters.types import MalformedRequestError

ALLOWED_SAMPLING_KEYS = frozenset({"temperature", "top_p", "top_k", "max_tokens", "stop"})
BASE = dict(base_sampling_params={}, allowed_sampling_keys=ALLOWED_SAMPLING_KEYS)


def test_system_string_becomes_first_message():
    """Top-level Anthropic system text becomes the leading internal system
    message and max_tokens is forwarded into sampling params."""
    req = anthropic_to_internal(
        {"system": "Be concise.", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
        **BASE,
    )
    assert req["messages"][0] == {"role": "system", "content": "Be concise."}
    assert req["sampling_params"]["max_tokens"] == 16


def test_user_text_image_text_order_preserved():
    """Mixed Anthropic user text/image blocks lower to OpenAI-compatible parts
    without reordering the multimodal content."""
    req = anthropic_to_internal(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                        {"type": "text", "text": "b"},
                    ],
                }
            ],
            "max_tokens": 8,
        },
        **BASE,
    )
    assert req["messages"][0]["content"] == [
        {"type": "text", "text": "a"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "text", "text": "b"},
    ]


def test_assistant_tool_use_to_tool_calls_dict_args():
    """Anthropic assistant tool_use blocks become internal OpenAI-style
    tool_calls while preserving dict arguments."""
    req = anthropic_to_internal(
        {
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}}],
                },
            ],
            "max_tokens": 8,
        },
        **BASE,
    )
    tc = req["messages"][-1]["tool_calls"][0]
    assert tc["function"]["name"] == "lookup"
    assert tc["function"]["arguments"] == {"q": "x"}


def test_malformed_anthropic_content_blocks_rejected():
    """Malformed or unsupported Anthropic content blocks fail at the adapter
    boundary before the session codec sees corrupted history."""
    payloads = [
        {"messages": [{"role": "user", "content": [{"type": "audio", "data": "x"}]}], "max_tokens": 8},
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "name": "lookup", "input": {"q": "x"}}],
                }
            ],
            "max_tokens": 8,
        },
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": 123, "name": "lookup", "input": {"q": "x"}}],
                }
            ],
            "max_tokens": 8,
        },
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": "bad"}],
                }
            ],
            "max_tokens": 8,
        },
        {"messages": [{"role": "assistant", "content": [{"type": "citation", "text": "x"}]}], "max_tokens": 8},
    ]

    for payload in payloads:
        with pytest.raises(MalformedRequestError):
            anthropic_to_internal(payload, **BASE)


def test_user_tool_result_text_becomes_tool_message():
    """A text-only Anthropic tool_result lowers to an internal tool message with
    the original tool_use_id."""
    req = anthropic_to_internal(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "text", "text": "found"}],
                        }
                    ],
                }
            ],
            "max_tokens": 8,
        },
        **BASE,
    )
    assert req["messages"][0]["role"] == "tool"
    assert req["messages"][0]["content"] == "found"


def test_malformed_tool_result_blocks_rejected():
    """Tool results without a usable id or with unsupported nested blocks are
    rejected instead of corrupting tool-return history."""
    payloads = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "", "content": [{"type": "text", "text": "found"}]}
                    ],
                }
            ],
            "max_tokens": 8,
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "json", "value": {}}],
                        }
                    ],
                }
            ],
            "max_tokens": 8,
        },
    ]

    for payload in payloads:
        with pytest.raises(MalformedRequestError):
            anthropic_to_internal(payload, **BASE)


def test_tool_result_image_appends_user_message():
    """Tool-result images are preserved as a following user image message while
    text remains in the correlated tool message."""
    req = anthropic_to_internal(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t",
                            "content": [
                                {"type": "text", "text": "see"},
                                {
                                    "type": "image",
                                    "source": {"type": "base64", "media_type": "image/png", "data": "BBBB"},
                                },
                            ],
                        }
                    ],
                }
            ],
            "max_tokens": 8,
        },
        **BASE,
    )
    assert req["messages"][0]["role"] == "tool"
    assert req["messages"][1]["role"] == "user"
    assert req["messages"][1]["content"][0]["type"] == "image_url"


def test_tool_result_image_only_preserves_empty_tool_message():
    """Image-only tool_result content still emits an empty tool message sentinel
    before the downgraded user image message."""
    req = anthropic_to_internal(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {"type": "base64", "media_type": "image/png", "data": "BBBB"},
                                }
                            ],
                        }
                    ],
                }
            ],
            "max_tokens": 8,
        },
        **BASE,
    )
    assert req["messages"][0] == {"role": "tool", "tool_call_id": "t", "content": ""}
    assert req["messages"][1]["role"] == "user"
    assert req["messages"][1]["content"][0]["type"] == "image_url"


def test_anthropic_request_control_fields_lower_or_reject():
    """Small Anthropic request-level controls stay adapter-owned: unsupported
    tool choices are rejected, while stop_sequences lowers to sampling stop."""
    with pytest.raises(MalformedRequestError):
        anthropic_to_internal(
            {"messages": [{"role": "user", "content": "hi"}], "tool_choice": {"type": "any"}, "max_tokens": 8},
            **BASE,
        )

    req = anthropic_to_internal(
        {"messages": [{"role": "user", "content": "hi"}], "stop_sequences": ["</s>"], "max_tokens": 8},
        **BASE,
    )
    assert req["sampling_params"]["stop"] == ["</s>"]


def test_mid_list_system_folded_into_user():
    """Mid-conversation system reminders are folded into user content so chat
    templates never see a system role after the first message."""
    req = anthropic_to_internal(
        {
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "system", "content": "reminder"},
                {"role": "user", "content": "b"},
            ],
            "max_tokens": 8,
        },
        **BASE,
    )
    roles = [m["role"] for m in req["messages"]]
    assert len(req["messages"]) == 2
    assert roles == ["user", "user"]
    assert "system" not in roles[1:]
    assert any("reminder" in str(m.get("content")) for m in req["messages"])
    assert not any(m.get("content") == "<system-reminder>\nreminder\n</system-reminder>" for m in req["messages"])


def test_mid_list_system_before_tool_result_does_not_cross_tool_message():
    """A system reminder before a tool_result folds into the prior user message
    and does not break the assistant/tool message adjacency."""
    req = anthropic_to_internal(
        {
            "messages": [
                {"role": "user", "content": "a"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}}],
                },
                {"role": "system", "content": "reminder"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "text", "text": "found"}],
                        }
                    ],
                },
            ],
            "max_tokens": 8,
        },
        **BASE,
    )
    assert req["messages"][0]["role"] == "user"
    assert "reminder" in req["messages"][0]["content"]
    assert req["messages"][-1]["role"] == "tool"
    assert not any(
        m["role"] == "user" and m.get("content") == "<system-reminder>\nreminder\n</system-reminder>"
        for m in req["messages"][1:]
    )


def test_billing_header_stripped_from_system():
    """Claude Code billing headers embedded in system text are stripped before
    prompt construction."""
    req = anthropic_to_internal(
        {
            "system": "x-anthropic-billing-header: cch=abc\nBe concise.",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        },
        **BASE,
    )
    assert req["messages"][0] == {"role": "system", "content": "Be concise."}


def test_provider_specific_tool_metadata_lowers_to_template_function_schema():
    """Provider-specific Anthropic tool metadata is ignored while the tool name
    remains available to the local chat template as an OpenAI function schema."""
    req = anthropic_to_internal(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        **BASE,
    )

    assert req["tools"] == [{"type": "function", "function": {"name": "web_search", "parameters": {}}}]


def test_tool_input_schema_passes_through_json_schema():
    """Anthropic JSON schema is copied into OpenAI function parameters without
    gateway-specific parser normalization."""
    schema = {
        "type": "object",
        "properties": {
            "target": {"anyOf": [{"const": "file"}, {"type": "string"}]},
            "mode": {"anyOf": [{"const": "read"}, {"const": "write"}]},
            "nested": {"type": "object", "properties": {"x": {"type": "integer"}}},
        },
        "required": ["target"],
    }
    req = anthropic_to_internal(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "tools": [{"name": "edit", "description": "edit files", "input_schema": schema}],
        },
        **BASE,
    )
    params = req["tools"][0]["function"]["parameters"]
    assert params is not schema
    assert params == schema
    assert "type" not in params["properties"]["target"]
    assert "enum" not in params["properties"]["target"]


def test_tool_input_schema_anyof_preserves_heterogeneous_branches_without_inferred_type():
    """Heterogeneous anyOf branches keep their distinct JSON types instead of
    being collapsed into a compatibility hint for a specific parser."""
    schema = {
        "type": "object",
        "properties": {
            "value": {"anyOf": [{"type": "integer"}, {"type": "number"}]},
        },
    }
    req = anthropic_to_internal(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "tools": [{"name": "edit", "description": "edit files", "input_schema": schema}],
        },
        **BASE,
    )
    assert req["tools"][0]["function"]["parameters"]["properties"]["value"] == {
        "anyOf": [{"type": "integer"}, {"type": "number"}]
    }


def test_thinking_block_dropped_with_warning(caplog):
    """Inbound Anthropic thinking blocks are dropped with a warning while
    neighboring assistant text remains in the prompt history."""
    with caplog.at_level("WARNING", logger="gateway"):
        req = anthropic_to_internal(
            {
                "messages": [
                    {"role": "user", "content": "go"},
                    {
                        "role": "assistant",
                        "content": [{"type": "thinking", "thinking": "reasoning..."}, {"type": "text", "text": "done"}],
                    },
                ],
                "max_tokens": 8,
            },
            **BASE,
        )
    assert req["messages"][-1]["content"] == "done"
    assert any("thinking" in r.message for r in caplog.records)


def test_unsupported_system_and_redacted_thinking_blocks_rejected():
    """System non-text blocks and encrypted redacted_thinking blocks are rejected
    because neither has a faithful local template representation."""
    with pytest.raises(MalformedRequestError):
        anthropic_to_internal(
            {
                "system": [{"type": "audio", "data": "x"}],
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
            },
            **BASE,
        )

    with pytest.raises(MalformedRequestError):
        anthropic_to_internal(
            {
                "messages": [
                    {"role": "user", "content": "go"},
                    {
                        "role": "assistant",
                        "content": [{"type": "redacted_thinking", "data": "xxx"}, {"type": "text", "text": "done"}],
                    },
                ],
                "max_tokens": 8,
            },
            **BASE,
        )


def test_anthropic_build_response_text():
    """A text GenerationOutcome serializes to the Anthropic Messages response
    shape with end_turn stop reason and usage counts."""
    from uni_agent.gateway.adapters.anthropic import anthropic_build_response
    from uni_agent.gateway.session.session import GenerationOutcome

    body = anthropic_build_response(
        GenerationOutcome(
            assistant_msg={"role": "assistant", "content": "hi"},
            finish_reason="stop",
            prompt_tokens=3,
            completion_tokens=1,
        ),
        model="claude-x",
    )
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"] == [{"type": "text", "text": "hi"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 3, "output_tokens": 1}


def test_anthropic_build_response_tool_use():
    """Internal OpenAI-style tool_calls serialize back to Anthropic tool_use
    blocks and map finish_reason to tool_use."""
    from uni_agent.gateway.adapters.anthropic import anthropic_build_response
    from uni_agent.gateway.session.session import GenerationOutcome

    body = anthropic_build_response(
        GenerationOutcome(
            assistant_msg={
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": {"a": 1}}}],
            },
            finish_reason="tool_calls",
            prompt_tokens=2,
            completion_tokens=4,
        ),
        model="claude-x",
    )
    tu = [b for b in body["content"] if b["type"] == "tool_use"][0]
    assert tu == {"type": "tool_use", "id": "c1", "name": "f", "input": {"a": 1}}
    assert body["stop_reason"] == "tool_use"


def test_anthropic_build_response_length_maps_to_max_tokens():
    """Internal length finish reasons are exposed as Anthropic max_tokens stop
    reasons."""
    from uni_agent.gateway.adapters.anthropic import anthropic_build_response
    from uni_agent.gateway.session.session import GenerationOutcome

    body = anthropic_build_response(
        GenerationOutcome(
            assistant_msg={"role": "assistant", "content": ""},
            finish_reason="length",
            prompt_tokens=1,
            completion_tokens=0,
        ),
        model="claude-x",
    )
    assert body["stop_reason"] == "max_tokens"


def test_anthropic_build_response_normalizes_tool_use_input():
    """Tool parser arguments are normalized for Anthropic clients: JSON-string
    objects become dicts and invalid strings become empty input."""
    from uni_agent.gateway.adapters.anthropic import anthropic_build_response
    from uni_agent.gateway.session.session import GenerationOutcome

    for arguments, expected_input in (('{"a": 1}', {"a": 1}), ("not json", {})):
        body = anthropic_build_response(
            GenerationOutcome(
                assistant_msg={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": arguments}}],
                },
                finish_reason="tool_calls",
                prompt_tokens=2,
                completion_tokens=4,
            ),
            model="claude-x",
        )
        tu = [b for b in body["content"] if b["type"] == "tool_use"][0]
        assert tu["input"] == expected_input


@pytest.mark.asyncio
async def test_anthropic_stream_event_sequence():
    """A synthesized text response follows the Anthropic Messages SSE event
    order and reports stop reason plus final usage in message_delta."""
    from uni_agent.gateway.adapters.anthropic import anthropic_stream_response
    from uni_agent.gateway.session.session import GenerationOutcome

    resp = anthropic_stream_response(
        GenerationOutcome(
            assistant_msg={"role": "assistant", "content": "hi"},
            finish_reason="stop",
            prompt_tokens=3,
            completion_tokens=1,
        ),
        model="claude-x",
    )
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["connection"] == "keep-alive"
    assert resp.headers["x-accel-buffering"] == "no"

    text = (b"".join([c async for c in resp.body_iterator])).decode()
    assert "event: message_start" in text
    assert "event: content_block_start" in text
    assert "text_delta" in text
    assert "event: content_block_stop" in text
    assert "event: message_delta" in text
    assert '"stop_reason": "end_turn"' in text or '"stop_reason":"end_turn"' in text
    assert (
        '"usage": {"input_tokens": 3, "output_tokens": 1}' in text
        or '"usage":{"input_tokens":3,"output_tokens":1}' in text
    )
    assert "event: message_stop" in text


@pytest.mark.asyncio
async def test_anthropic_stream_response_emits_tool_use_delta():
    """A synthesized tool-call response emits Anthropic tool_use start data and
    streams the tool input through an input_json_delta event."""
    from uni_agent.gateway.adapters.anthropic import anthropic_stream_response
    from uni_agent.gateway.session.session import GenerationOutcome

    resp = anthropic_stream_response(
        GenerationOutcome(
            assistant_msg={
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": {"a": 1}}}],
            },
            finish_reason="tool_calls",
            prompt_tokens=2,
            completion_tokens=4,
        ),
        model="claude-x",
    )

    text = (b"".join([c async for c in resp.body_iterator])).decode()
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        events.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))

    start = next(data for event, data in events if event == "content_block_start")
    delta = next(data for event, data in events if event == "content_block_delta")
    message_delta = next(data for event, data in events if event == "message_delta")

    assert start["content_block"] == {"type": "tool_use", "id": "c1", "name": "f", "input": {}}
    assert delta["delta"]["type"] == "input_json_delta"
    assert json.loads(delta["delta"]["partial_json"]) == {"a": 1}
    assert message_delta["delta"]["stop_reason"] == "tool_use"
