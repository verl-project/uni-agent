from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest


ALLOWED_SAMPLING_KEYS = frozenset({"temperature", "top_p", "top_k", "max_tokens", "stop"})


def _request(**overrides):
    request = {
        "model": "policy",
        "instructions": "Use the repository tools.",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "inspect"}]}
        ],
        "tools": [
            {
                "type": "custom",
                "name": "exec",
                "description": "Run JavaScript tools",
                "format": {"type": "grammar", "syntax": "lark", "definition": "..."},
            }
        ],
        "max_output_tokens": 128,
        "stream": True,
    }
    request.update(overrides)
    return request


def _outcome(*, content: str = "", tool_calls=None, finish_reason: str = "tool_calls"):
    return SimpleNamespace(
        assistant_msg={"role": "assistant", "content": content, "tool_calls": tool_calls or []},
        finish_reason=finish_reason,
        prompt_tokens=10,
        completion_tokens=3,
    )


def test_custom_exec_round_trips_raw_input_and_tool_output():
    from uni_agent.gateway.adapters.responses import responses_build_response, responses_to_internal

    javascript = 'const r = await tools.exec_command({"cmd":"git status --short"});\ntext(r.output);'
    payload = _request(
        input=[
            {"type": "message", "role": "user", "content": "inspect"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "need status"}]},
            {"type": "custom_tool_call", "call_id": "call-1", "name": "exec", "input": javascript},
            {"type": "custom_tool_call_output", "call_id": "call-1", "output": " M file.py"},
        ]
    )
    internal = responses_to_internal(
        payload,
        base_sampling_params={"top_p": 0.9},
        allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
    )

    assert internal["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "exec",
                "description": "Run JavaScript tools",
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    assistant = internal["messages"][2]
    assert assistant["reasoning_content"] == "need status"
    assert assistant["tool_calls"][0]["function"]["arguments"] == {"input": javascript}
    assert internal["messages"][3] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": " M file.py",
    }
    assert internal["sampling_params"] == {"top_p": 0.9, "max_tokens": 128}

    body = responses_build_response(
        _outcome(
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "exec", "arguments": {"input": javascript}},
                }
            ]
        ),
        payload=payload,
        model="policy",
    )
    assert body["output"][0]["type"] == "custom_tool_call"
    assert body["output"][0]["input"] == javascript


def test_responses_preserves_images_and_namespaces_for_lowering():
    from uni_agent.gateway.adapters.responses import responses_to_internal

    payload = _request(
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "inspect"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                ],
            }
        ],
        tools=[
            {
                "type": "namespace",
                "name": "repo",
                "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
            }
        ],
    )
    internal = responses_to_internal(
        payload,
        base_sampling_params={},
        allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
    )
    assert internal["messages"][1]["content"] == [
        {"type": "text", "text": "inspect"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    assert internal["tools"][0]["function"]["name"] == "repo__lookup"


@pytest.mark.parametrize("field,value", [("background", True), ("store", True), ("previous_response_id", "resp_x")])
def test_responses_rejects_unsupported_stateful_features(field, value):
    from uni_agent.gateway.adapters.responses import responses_to_internal
    from uni_agent.gateway.adapters.types import MalformedRequestError

    with pytest.raises(MalformedRequestError):
        responses_to_internal(
            _request(**{field: value}),
            base_sampling_params={},
            allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
        )


def test_responses_length_status_and_error_envelope():
    from uni_agent.gateway.adapters.responses import responses_build_response, responses_error_body

    body = responses_build_response(
        _outcome(content="", tool_calls=[], finish_reason="length"),
        payload=_request(),
        model="policy",
    )
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}
    assert responses_error_body(400, "bad", param="input")["error"]["type"] == "invalid_request_error"


def test_responses_tool_choice_none_disables_tools():
    from uni_agent.gateway.adapters.responses import responses_to_internal

    internal = responses_to_internal(
        _request(tool_choice="none"),
        base_sampling_params={},
        allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
    )
    assert internal["tools"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"text": {"format": {"type": "json_schema", "name": "result", "schema": {}}}},
    ],
)
def test_responses_rejects_unsupported_capability_flags(overrides):
    from uni_agent.gateway.adapters.responses import responses_to_internal
    from uni_agent.gateway.adapters.types import MalformedRequestError

    with pytest.raises(MalformedRequestError):
        responses_to_internal(
            _request(**overrides),
            base_sampling_params={},
            allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
        )


def test_responses_accepts_parallel_tool_calls_false_for_single_call():
    from uni_agent.gateway.adapters.responses import responses_to_internal

    internal = responses_to_internal(
        _request(parallel_tool_calls=False),
        base_sampling_params={},
        allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
    )
    assert internal["messages"][1]["role"] == "user"


def test_responses_rejects_multiple_tool_calls_when_parallel_disabled():
    from uni_agent.gateway.adapters.responses import responses_build_response
    from uni_agent.gateway.adapters.types import MalformedRequestError

    with pytest.raises(MalformedRequestError):
        responses_build_response(
            _outcome(
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "exec", "arguments": {"input": "pwd"}},
                    },
                    {
                        "id": "call-2",
                        "type": "function",
                        "function": {"name": "exec", "arguments": {"input": "ls"}},
                    },
                ]
            ),
            payload=_request(parallel_tool_calls=False),
            model="policy",
        )


def test_responses_rejects_unknown_tool_type_instead_of_dropping_it():
    from uni_agent.gateway.adapters.responses import responses_to_internal
    from uni_agent.gateway.adapters.types import MalformedRequestError

    with pytest.raises(MalformedRequestError):
        responses_to_internal(
            _request(tools=[{"type": "future_hosted_tool", "name": "lookup"}]),
            base_sampling_params={},
            allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
        )


def test_responses_preserves_known_hosted_tool_as_parser_declaration():
    from uni_agent.gateway.adapters.responses import responses_to_internal

    internal = responses_to_internal(
        _request(tools=[{"type": "web_search", "description": "Search the web"}]),
        base_sampling_params={},
        allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
    )
    assert internal["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_responses_rejects_unknown_input_item_instead_of_degrading_it():
    from uni_agent.gateway.adapters.responses import responses_to_internal
    from uni_agent.gateway.adapters.types import MalformedRequestError

    with pytest.raises(MalformedRequestError):
        responses_to_internal(
            _request(input=[{"type": "future_input_item", "content": "not understood"}]),
            base_sampling_params={},
            allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
        )


def test_responses_rejects_unknown_content_block_instead_of_dropping_it():
    from uni_agent.gateway.adapters.responses import responses_to_internal
    from uni_agent.gateway.adapters.types import MalformedRequestError

    with pytest.raises(MalformedRequestError):
        responses_to_internal(
            _request(
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "future_content_block", "value": "not understood"}],
                    }
                ]
            ),
            base_sampling_params={},
            allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
        )


@pytest.mark.asyncio
async def test_responses_stream_emits_heartbeats_and_custom_events():
    from uni_agent.gateway.adapters.responses import responses_stream_response

    async def generate():
        await asyncio.sleep(0.03)
        return _outcome(
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "exec", "arguments": {"input": "pwd"}},
                }
            ]
        )

    response = responses_stream_response(
        generate,
        payload=_request(),
        model="policy",
        heartbeat_interval_s=0.005,
    )
    text = (b"".join([chunk async for chunk in response.body_iterator])).decode()
    events = [line.removeprefix("event: ") for line in text.splitlines() if line.startswith("event: ")]
    data = [json.loads(line.removeprefix("data: ")) for line in text.splitlines() if line.startswith("data: ")]

    assert events[:2] == ["response.created", "response.in_progress"]
    assert events.count("response.in_progress") >= 2
    assert "response.custom_tool_call_input.delta" in events
    assert events[-1] == "response.completed"
    assert [item["sequence_number"] for item in data] == list(range(len(data)))
