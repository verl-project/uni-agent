import pytest

from uni_agent.gateway.adapters.anthropic import anthropic_to_internal
from uni_agent.gateway.adapters.types import MalformedRequestError

ALLOWED_SAMPLING_KEYS = frozenset({"temperature", "top_p", "top_k", "max_tokens", "stop"})
BASE = dict(base_sampling_params={}, allowed_sampling_keys=ALLOWED_SAMPLING_KEYS)


def test_system_string_becomes_first_message():
    req = anthropic_to_internal(
        {"system": "Be concise.", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
        **BASE,
    )
    assert req["messages"][0] == {"role": "system", "content": "Be concise."}
    assert req["sampling_params"]["max_tokens"] == 16


def test_user_image_block_to_image_url():
    req = anthropic_to_internal(
        {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
            ]}],
            "max_tokens": 8,
        },
        **BASE,
    )
    parts = req["messages"][0]["content"]
    assert {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}} in parts


def test_user_text_image_text_order_preserved():
    req = anthropic_to_internal(
        {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "a"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                {"type": "text", "text": "b"},
            ]}],
            "max_tokens": 8,
        },
        **BASE,
    )
    assert req["messages"][0]["content"] == [
        {"type": "text", "text": "a"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "text", "text": "b"},
    ]


def test_unknown_user_block_rejected():
    with pytest.raises(MalformedRequestError):
        anthropic_to_internal(
            {"messages": [{"role": "user", "content": [{"type": "audio", "data": "x"}]}], "max_tokens": 8},
            **BASE,
        )


def test_assistant_tool_use_to_tool_calls_dict_args():
    req = anthropic_to_internal(
        {"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}}]},
        ], "max_tokens": 8},
        **BASE,
    )
    tc = req["messages"][-1]["tool_calls"][0]
    assert tc["function"]["name"] == "lookup"
    assert tc["function"]["arguments"] == {"q": "x"}


@pytest.mark.parametrize("tool_id", [None, 123])
def test_assistant_tool_use_requires_string_id(tool_id):
    block = {"type": "tool_use", "name": "lookup", "input": {"q": "x"}}
    if tool_id is not None:
        block["id"] = tool_id
    with pytest.raises(MalformedRequestError):
        anthropic_to_internal(
            {"messages": [{"role": "assistant", "content": [block]}], "max_tokens": 8},
            **BASE,
        )


def test_assistant_tool_use_input_must_be_object():
    with pytest.raises(MalformedRequestError):
        anthropic_to_internal(
            {"messages": [{"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": "bad"}]}],
             "max_tokens": 8},
            **BASE,
        )


def test_unknown_assistant_block_rejected():
    with pytest.raises(MalformedRequestError):
        anthropic_to_internal(
            {"messages": [{"role": "assistant", "content": [{"type": "citation", "text": "x"}]}], "max_tokens": 8},
            **BASE,
        )


def test_user_tool_result_text_becomes_tool_message():
    req = anthropic_to_internal(
        {"messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "found"}]}]}],
         "max_tokens": 8},
        **BASE,
    )
    assert req["messages"][0]["role"] == "tool"
    assert req["messages"][0]["content"] == "found"


def test_tool_result_requires_non_empty_tool_use_id():
    with pytest.raises(MalformedRequestError):
        anthropic_to_internal(
            {"messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "", "content": [{"type": "text", "text": "found"}]}]}],
             "max_tokens": 8},
            **BASE,
        )


def test_unknown_tool_result_block_rejected():
    with pytest.raises(MalformedRequestError):
        anthropic_to_internal(
            {"messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "json", "value": {}}]}]}],
             "max_tokens": 8},
            **BASE,
        )


def test_tool_result_image_appends_user_message():
    req = anthropic_to_internal(
        {"messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "content": [
                {"type": "text", "text": "see"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "BBBB"}}]}]}],
         "max_tokens": 8},
        **BASE,
    )
    assert req["messages"][0]["role"] == "tool"
    assert req["messages"][1]["role"] == "user"
    assert req["messages"][1]["content"][0]["type"] == "image_url"


def test_tool_result_image_only_preserves_empty_tool_message():
    req = anthropic_to_internal(
        {"messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "BBBB"}}]}]}],
         "max_tokens": 8},
        **BASE,
    )
    assert req["messages"][0] == {"role": "tool", "tool_call_id": "t", "content": ""}
    assert req["messages"][1]["role"] == "user"
    assert req["messages"][1]["content"][0]["type"] == "image_url"


def test_tool_choice_any_rejected():
    with pytest.raises(MalformedRequestError):
        anthropic_to_internal(
            {"messages": [{"role": "user", "content": "hi"}], "tool_choice": {"type": "any"}, "max_tokens": 8},
            **BASE,
        )


def test_stop_sequences_mapped_to_stop():
    req = anthropic_to_internal(
        {"messages": [{"role": "user", "content": "hi"}], "stop_sequences": ["</s>"], "max_tokens": 8},
        **BASE,
    )
    assert req["sampling_params"]["stop"] == ["</s>"]


def test_mid_list_system_folded_into_user():
    req = anthropic_to_internal(
        {"messages": [
            {"role": "user", "content": "a"},
            {"role": "system", "content": "reminder"},
            {"role": "user", "content": "b"}],
         "max_tokens": 8},
        **BASE,
    )
    roles = [m["role"] for m in req["messages"]]
    assert len(req["messages"]) == 2
    assert roles == ["user", "user"]
    assert "system" not in roles[1:]
    assert any("reminder" in str(m.get("content")) for m in req["messages"])
    assert not any(m.get("content") == "<system-reminder>\nreminder\n</system-reminder>" for m in req["messages"])


def test_mid_list_system_before_tool_result_does_not_cross_tool_message():
    req = anthropic_to_internal(
        {"messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}}]},
            {"role": "system", "content": "reminder"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "found"}]}]}],
         "max_tokens": 8},
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
    req = anthropic_to_internal(
        {"system": "x-anthropic-billing-header: cch=abc\nBe concise.",
         "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
        **BASE,
    )
    assert req["messages"][0] == {"role": "system", "content": "Be concise."}


def test_unknown_system_block_rejected():
    with pytest.raises(MalformedRequestError):
        anthropic_to_internal(
            {"system": [{"type": "audio", "data": "x"}],
             "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
            **BASE,
        )


def test_server_tool_rejected():
    with pytest.raises(MalformedRequestError):
        anthropic_to_internal(
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
             "tools": [{"type": "web_search_20250305", "name": "web_search"}]},
            **BASE,
        )


def test_tool_input_schema_minimally_normalized_for_qwen_parser():
    from verl.tools.schemas import OpenAIFunctionToolSchema

    schema = {
        "type": "object",
        "properties": {
            "target": {"anyOf": [{"const": "file"}, {"type": "string"}]},
            "nested": {"type": "object", "properties": {"x": {"type": "integer"}}},
        },
        "required": ["target"],
    }
    req = anthropic_to_internal(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
         "tools": [{"name": "edit", "description": "edit files", "input_schema": schema}]},
        **BASE,
    )
    params = req["tools"][0]["function"]["parameters"]
    assert params is not schema
    assert params["properties"]["target"]["type"] == "string"
    assert params["properties"]["target"]["enum"] == ["file"]
    assert params["properties"]["target"]["anyOf"] == [{"const": "file"}, {"type": "string"}]
    assert params["properties"]["nested"]["properties"] == {"x": {"type": "integer"}}
    OpenAIFunctionToolSchema(**req["tools"][0])


def test_tool_input_schema_anyof_preserves_heterogeneous_types():
    schema = {
        "type": "object",
        "properties": {
            "value": {"anyOf": [{"type": "integer"}, {"type": "number"}]},
        },
    }
    req = anthropic_to_internal(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
         "tools": [{"name": "edit", "description": "edit files", "input_schema": schema}]},
        **BASE,
    )
    assert req["tools"][0]["function"]["parameters"]["properties"]["value"]["type"] == ["integer", "number"]


def test_thinking_block_dropped_with_warning(caplog):
    with caplog.at_level("WARNING", logger="gateway"):
        req = anthropic_to_internal(
            {"messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "reasoning..."},
                    {"type": "text", "text": "done"}]}],
             "max_tokens": 8},
            **BASE,
        )
    assert req["messages"][-1]["content"] == "done"
    assert any("thinking" in r.message for r in caplog.records)


def test_redacted_thinking_rejected():
    with pytest.raises(MalformedRequestError):
        anthropic_to_internal(
            {"messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [
                    {"type": "redacted_thinking", "data": "xxx"},
                    {"type": "text", "text": "done"}]}],
             "max_tokens": 8},
            **BASE,
        )


def test_anthropic_build_response_text():
    from uni_agent.gateway.adapters.anthropic import anthropic_build_response
    from uni_agent.gateway.session.session import GenerationOutcome

    body = anthropic_build_response(
        GenerationOutcome(assistant_msg={"role": "assistant", "content": "hi"},
                          finish_reason="stop", prompt_tokens=3, completion_tokens=1),
        model="claude-x",
    )
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"] == [{"type": "text", "text": "hi"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 3, "output_tokens": 1}


def test_anthropic_build_response_tool_use():
    from uni_agent.gateway.adapters.anthropic import anthropic_build_response
    from uni_agent.gateway.session.session import GenerationOutcome

    body = anthropic_build_response(
        GenerationOutcome(
            assistant_msg={"role": "assistant", "content": "",
                           "tool_calls": [{"id": "c1", "type": "function",
                                           "function": {"name": "f", "arguments": {"a": 1}}}]},
            finish_reason="tool_calls", prompt_tokens=2, completion_tokens=4),
        model="claude-x",
    )
    tu = [b for b in body["content"] if b["type"] == "tool_use"][0]
    assert tu == {"type": "tool_use", "id": "c1", "name": "f", "input": {"a": 1}}
    assert body["stop_reason"] == "tool_use"


def test_anthropic_build_response_invalid_json_args_becomes_empty_input():
    from uni_agent.gateway.adapters.anthropic import anthropic_build_response
    from uni_agent.gateway.session.session import GenerationOutcome

    body = anthropic_build_response(
        GenerationOutcome(
            assistant_msg={"role": "assistant", "content": "",
                           "tool_calls": [{"id": "c1", "type": "function",
                                           "function": {"name": "f", "arguments": "not json"}}]},
            finish_reason="tool_calls", prompt_tokens=2, completion_tokens=4),
        model="claude-x",
    )
    tu = [b for b in body["content"] if b["type"] == "tool_use"][0]
    assert tu["input"] == {}


def test_anthropic_build_response_length_maps_to_max_tokens():
    from uni_agent.gateway.adapters.anthropic import anthropic_build_response
    from uni_agent.gateway.session.session import GenerationOutcome

    body = anthropic_build_response(
        GenerationOutcome(assistant_msg={"role": "assistant", "content": ""},
                          finish_reason="length", prompt_tokens=1, completion_tokens=0),
        model="claude-x",
    )
    assert body["stop_reason"] == "max_tokens"


def test_anthropic_build_response_parses_json_string_args():
    """tool parser may emit JSON-string arguments; Anthropic tool_use.input must
    be a dict (zqz A2: robust dict | JSON-string | invalid->{})."""
    from uni_agent.gateway.adapters.anthropic import anthropic_build_response
    from uni_agent.gateway.session.session import GenerationOutcome

    body = anthropic_build_response(
        GenerationOutcome(
            assistant_msg={"role": "assistant", "content": "",
                           "tool_calls": [{"id": "c1", "type": "function",
                                           "function": {"name": "f", "arguments": '{"a": 1}'}}]},
            finish_reason="tool_calls", prompt_tokens=2, completion_tokens=4),
        model="claude-x",
    )
    tu = [b for b in body["content"] if b["type"] == "tool_use"][0]
    assert tu["input"] == {"a": 1}


@pytest.mark.asyncio
async def test_anthropic_stream_event_sequence():
    from uni_agent.gateway.adapters.anthropic import anthropic_stream_response
    from uni_agent.gateway.session.session import GenerationOutcome

    resp = anthropic_stream_response(
        GenerationOutcome(assistant_msg={"role": "assistant", "content": "hi"},
                          finish_reason="stop", prompt_tokens=3, completion_tokens=1),
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
