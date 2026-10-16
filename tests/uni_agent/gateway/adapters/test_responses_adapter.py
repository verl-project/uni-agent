from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytestmark = [pytest.mark.cpu, pytest.mark.level0]

ALLOWED_SAMPLING_KEYS = frozenset({"temperature", "top_p", "top_k", "max_tokens", "stop"})
_MISSING_ROLE = object()


def _request(**overrides):
    request = {
        "model": "policy",
        "instructions": "Use the repository tools.",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "inspect"}]}],
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
    from uni_agent.gateway.adapters.openai_response import responses_build_response, responses_to_internal

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
    from uni_agent.gateway.adapters.openai_response import responses_to_internal

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


def _namespace_tool(tool, namespaces):
    for namespace in reversed(namespaces):
        tool = {"type": "namespace", "name": namespace, "tools": [tool]}
    return tool


@pytest.mark.parametrize("kind", ["function", "custom"])
@pytest.mark.parametrize(
    "namespaces,name,expected_namespace,qualified_name",
    [
        ([], "look__up", None, "look__up"),
        (["repo"], "lookup", "repo", "repo__lookup"),
        (["repo"], "look__up", "repo", "repo__look__up"),
        (["outer__inner"], "lookup", "outer__inner", "outer__inner__lookup"),
        (["outer", "inner"], "lookup", "outer__inner", "outer__inner__lookup"),
        (["outer", "functions", "inner"], "look__up", "outer__inner", "outer__inner__look__up"),
    ],
)
def test_responses_preserves_declared_namespace_and_tool_name(
    kind, namespaces, name, expected_namespace, qualified_name
):
    from uni_agent.gateway.adapters.openai_response import responses_build_response, responses_to_internal

    payload = _request(tools=[_namespace_tool({"type": kind, "name": name}, namespaces)])
    internal = responses_to_internal(payload, base_sampling_params={}, allowed_sampling_keys=ALLOWED_SAMPLING_KEYS)
    assert internal["tools"][0]["function"]["name"] == qualified_name

    body = responses_build_response(
        _outcome(
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": qualified_name, "arguments": {"input": "pwd"}},
                }
            ]
        ),
        payload=payload,
        model="policy",
    )
    call = body["output"][0]
    assert call["type"] == ("custom_tool_call" if kind == "custom" else "function_call")
    assert call["name"] == name
    assert call.get("namespace") == expected_namespace
    replay = responses_to_internal(
        _request(tools=payload["tools"], input=[{"role": "user", "content": "inspect"}, call]),
        base_sampling_params={},
        allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
    )
    assert replay["messages"][-1]["tool_calls"][0]["function"]["name"] == qualified_name


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("kind", ["function", "custom"])
@pytest.mark.parametrize("namespaces", [[], ["repo"]])
@pytest.mark.asyncio
async def test_responses_deferred_tool_name_errors_precede_generation(stream, kind, namespaces):
    from fastapi import HTTPException

    from uni_agent.gateway.gateway import _GatewayActor

    generate = AsyncMock(return_value=_outcome(content="done", finish_reason="stop"))
    actor = object.__new__(_GatewayActor)
    actor._sessions = {"s1": SimpleNamespace(sampling_params={}, run_generation=generate)}
    actor._allowed_request_sampling_param_keys = ALLOWED_SAMPLING_KEYS
    actor._backend = object()
    tool = _namespace_tool({"type": kind, "defer_loading": True}, namespaces)
    with pytest.raises(HTTPException) as error:
        await actor._handle_openai_responses("s1", _request(stream=stream, tools=[tool]))
    assert error.value.status_code == 400
    assert "require a non-empty name" in error.value.detail
    generate.assert_not_called()


@pytest.mark.parametrize("kind", ["function", "custom"])
@pytest.mark.parametrize("namespaces", [[], ["repo"]])
def test_responses_valid_deferred_tools_keep_output_metadata(kind, namespaces):
    from uni_agent.gateway.adapters.openai_response import responses_build_response, responses_to_internal

    tool = _namespace_tool({"type": kind, "name": "lookup", "defer_loading": True}, namespaces)
    payload = _request(tools=[tool])
    internal = responses_to_internal(payload, base_sampling_params={}, allowed_sampling_keys=ALLOWED_SAMPLING_KEYS)
    assert internal["tools"] is None
    qualified_name = "repo__lookup" if namespaces else "lookup"
    body = responses_build_response(
        _outcome(
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": qualified_name, "arguments": {"input": "pwd"}},
                }
            ]
        ),
        payload=payload,
        model="policy",
    )
    call = body["output"][0]
    assert call["type"] == ("custom_tool_call" if kind == "custom" else "function_call")
    assert call["name"] == "lookup"
    assert call.get("namespace") == ("repo" if namespaces else None)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "left_namespaces,right_namespaces,right_name,left_deferred,right_deferred",
    [
        (["a__b"], ["a"], "b__c", False, False),
        (["a", "b"], ["a"], "b__c", False, False),
        (["a__b"], [], "a__b__c", False, False),
        (["a__b"], ["a"], "b__c", True, False),
        (["a__b"], ["a"], "b__c", False, True),
    ],
)
@pytest.mark.asyncio
async def test_responses_ambiguous_flat_tool_names_precede_generation(
    stream, left_namespaces, right_namespaces, right_name, left_deferred, right_deferred
):
    from fastapi import HTTPException

    from uni_agent.gateway.gateway import _GatewayActor

    tools = [
        _namespace_tool({"type": "function", "name": "c", "defer_loading": left_deferred}, left_namespaces),
        _namespace_tool({"type": "custom", "name": right_name, "defer_loading": right_deferred}, right_namespaces),
    ]
    generate = AsyncMock(return_value=_outcome(content="done", finish_reason="stop"))
    actor = object.__new__(_GatewayActor)
    actor._sessions = {"s1": SimpleNamespace(sampling_params={}, run_generation=generate)}
    actor._allowed_request_sampling_param_keys = ALLOWED_SAMPLING_KEYS
    actor._backend = object()
    with pytest.raises(HTTPException) as error:
        await actor._handle_openai_responses("s1", _request(stream=stream, tools=tools))
    assert error.value.status_code == 400
    assert "Ambiguous Responses tool name" in error.value.detail
    generate.assert_not_called()


@pytest.mark.parametrize(
    "source,expected_url",
    [
        ({"image_url": "https://example.test/image.png"}, "https://example.test/image.png"),
        ({"image_url": {"url": "https://example.test/image.png"}}, "https://example.test/image.png"),
        ({"url": "https://example.test/image.png"}, "https://example.test/image.png"),
        ({"image_url": "data:image/png;base64,AAAA"}, "data:image/png;base64,AAAA"),
        ({"data": "AAAA"}, "data:image/png;base64,AAAA"),
        ({"data": "AAAA", "mime_type": "image/jpeg"}, "data:image/jpeg;base64,AAAA"),
        ({"image_data": "AAAA", "media_type": "image/webp"}, "data:image/webp;base64,AAAA"),
        ({"data": "data:image/png;base64,AAAA"}, "data:image/png;base64,AAAA"),
    ],
)
def test_responses_normalizes_supported_image_sources(source, expected_url):
    from uni_agent.gateway.adapters.openai_response import responses_to_internal

    internal = responses_to_internal(
        _request(
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "inspect"}, {"type": "input_image", **source}],
                }
            ]
        ),
        base_sampling_params={},
        allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
    )
    assert internal["messages"][1]["content"] == [
        {"type": "text", "text": "inspect"},
        {"type": "image_url", "image_url": {"url": expected_url}},
    ]


@pytest.mark.parametrize(
    "source,field",
    [
        ({"file_id": "file-secret-id"}, "file_id"),
        ({}, "none"),
        ({"image_url": ""}, "image_url"),
        ({"image_url": {}}, "image_url"),
        ({"url": 1}, "url"),
        ({"data": []}, "data"),
        ({"image_data": {}}, "image_data"),
    ],
)
def test_responses_rejects_images_without_supported_sources(source, field):
    from uni_agent.gateway.adapters.openai_response import responses_to_internal
    from uni_agent.gateway.adapters.types import MalformedRequestError

    with pytest.raises(MalformedRequestError, match=r"input\[0\].content\[1\]") as error:
        responses_to_internal(
            _request(
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "inspect"}, {"type": "input_image", **source}],
                    }
                ]
            ),
            base_sampling_params={},
            allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
        )
    assert field in str(error.value)
    assert "file-secret-id" not in str(error.value)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("source", [{"file_id": "file-secret-id"}, {}])
@pytest.mark.asyncio
async def test_responses_image_source_errors_precede_generation(stream, source):
    from fastapi import HTTPException

    from uni_agent.gateway.gateway import _GatewayActor

    generate = AsyncMock(return_value=_outcome(content="done", finish_reason="stop"))
    actor = object.__new__(_GatewayActor)
    actor._sessions = {"s1": SimpleNamespace(sampling_params={}, run_generation=generate)}
    actor._allowed_request_sampling_param_keys = ALLOWED_SAMPLING_KEYS
    actor._backend = object()
    with pytest.raises(HTTPException) as error:
        await actor._handle_openai_responses(
            "s1", _request(stream=stream, input=[{"role": "user", "content": [{"type": "input_image", **source}]}])
        )
    assert error.value.status_code == 400
    assert "input[0].content[0]" in error.value.detail
    generate.assert_not_called()


@pytest.mark.parametrize("field,value", [("background", True), ("store", True), ("previous_response_id", "resp_x")])
def test_responses_rejects_unsupported_stateful_features(field, value):
    from uni_agent.gateway.adapters.openai_response import responses_to_internal
    from uni_agent.gateway.adapters.types import MalformedRequestError

    with pytest.raises(MalformedRequestError):
        responses_to_internal(
            _request(**{field: value}),
            base_sampling_params={},
            allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
        )


def test_responses_length_status_and_error_envelope():
    from uni_agent.gateway.adapters.openai_response import responses_build_response, responses_error_body

    body = responses_build_response(
        _outcome(content="", tool_calls=[], finish_reason="length"),
        payload=_request(),
        model="policy",
    )
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}
    assert responses_error_body(400, "bad", param="input")["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize(
    "overrides",
    [
        {"text": {"format": {"type": "json_schema", "name": "result", "schema": {}}}},
    ],
)
def test_responses_rejects_unsupported_capability_flags(overrides):
    from uni_agent.gateway.adapters.openai_response import responses_to_internal
    from uni_agent.gateway.adapters.types import MalformedRequestError

    with pytest.raises(MalformedRequestError):
        responses_to_internal(
            _request(**overrides),
            base_sampling_params={},
            allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
        )


def test_responses_rejects_parallel_tool_calls_false_at_request_entry():
    from uni_agent.gateway.adapters.openai_response import responses_to_internal
    from uni_agent.gateway.adapters.types import MalformedRequestError

    with pytest.raises(MalformedRequestError, match="parallel_tool_calls=false is not supported"):
        responses_to_internal(
            _request(parallel_tool_calls=False),
            base_sampling_params={},
            allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
        )


@pytest.mark.parametrize("parallel", [True, "omitted"])
def test_responses_accepts_parallel_tool_calls_true_or_default(parallel):
    from uni_agent.gateway.adapters.openai_response import responses_to_internal

    payload = _request(**({"parallel_tool_calls": parallel} if parallel != "omitted" else {}))
    assert responses_to_internal(payload, base_sampling_params={}, allowed_sampling_keys=ALLOWED_SAMPLING_KEYS)["tools"]


def test_responses_rejects_unknown_tool_type_instead_of_dropping_it():
    from uni_agent.gateway.adapters.openai_response import responses_to_internal
    from uni_agent.gateway.adapters.types import MalformedRequestError

    with pytest.raises(MalformedRequestError):
        responses_to_internal(
            _request(tools=[{"type": "future_hosted_tool", "name": "lookup"}]),
            base_sampling_params={},
            allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
        )


def test_responses_preserves_known_hosted_tool_as_parser_declaration():
    from uni_agent.gateway.adapters.openai_response import responses_to_internal

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
    from uni_agent.gateway.adapters.openai_response import responses_to_internal
    from uni_agent.gateway.adapters.types import MalformedRequestError

    with pytest.raises(MalformedRequestError):
        responses_to_internal(
            _request(input=[{"type": "future_input_item", "content": "not understood"}]),
            base_sampling_params={},
            allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
        )


def test_responses_rejects_unknown_content_block_instead_of_dropping_it():
    from uni_agent.gateway.adapters.openai_response import responses_to_internal
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


@pytest.fixture(params=["text", "function", "custom", "reasoning_length", "reasoning_only_length"])
def stream_case(request):
    kind = request.param
    payload = _request()
    outcome = _outcome(
        content="done" if kind in {"text", "reasoning_length"} else "",
        finish_reason="length" if kind.startswith("reasoning") else "stop",
    )
    if kind in {"function", "custom"}:
        outcome.assistant_msg["tool_calls"] = [
            {"id": "call-1", "type": "function", "function": {"name": "exec", "arguments": {"input": "pwd"}}}
        ]
        if kind == "function":
            payload["tools"] = [{"type": "function", "name": "exec", "parameters": {"type": "object"}}]
    if kind.startswith("reasoning"):
        outcome.assistant_msg["reasoning_content"] = "plan"
    return outcome, payload


async def _stream_bytes(outcome, payload):
    from uni_agent.gateway.adapters.openai_response import responses_stream_response

    response = responses_stream_response(outcome, payload=payload, model="policy")
    return b"".join([chunk async for chunk in response.body_iterator])


@pytest.mark.asyncio
async def test_responses_stream_preserves_identity_and_content_parts(stream_case, monkeypatch):
    from uni_agent.gateway.adapters import openai_response

    timestamps = iter(range(1700000000, 1700000010))
    monkeypatch.setattr(openai_response, "time", SimpleNamespace(time=lambda: next(timestamps)))
    outcome, payload = stream_case
    raw = await _stream_bytes(outcome, payload)
    data = [json.loads(line.removeprefix("data: ")) for line in raw.decode().splitlines() if line.startswith("data: ")]
    types = [event["type"] for event in data]
    assert types[:2] == ["response.created", "response.in_progress"]
    assert types.count("response.in_progress") == 1
    assert [event["sequence_number"] for event in data] == list(range(len(data)))
    final = data[-1]["response"]
    assert types[-1] == ("response.incomplete" if outcome.finish_reason == "length" else "response.completed")
    responses = [event["response"] for event in data if "response" in event]
    assert {response["id"] for response in responses} == {final["id"]}
    assert {response["created_at"] for response in responses} == {final["created_at"]}
    done_items = [event["item"] for event in data if event["type"] == "response.output_item.done"]
    assert done_items == final["output"]
    for event in data:
        if "item_id" in event:
            assert event["item_id"] == final["output"][event["output_index"]]["id"]
        if event["type"] == "response.output_item.added":
            assert event["item"]["id"] == final["output"][event["output_index"]]["id"]
    if outcome.assistant_msg["content"]:
        assert types.index("response.content_part.added") < types.index("response.output_text.delta")
        assert types.index("response.output_text.done") < types.index("response.content_part.done")
        part_done = next(event for event in data if event["type"] == "response.content_part.done")
        assert part_done["part"] == final["output"][part_done["output_index"]]["content"][0]


@pytest.mark.asyncio
async def test_responses_stream_is_consumed_by_openai_sdk(stream_case):
    openai = pytest.importorskip("openai", minversion="2.6.1")
    import httpx

    outcome, payload = stream_case
    raw = await _stream_bytes(outcome, payload)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=raw)
    )
    with openai.OpenAI(api_key="test-key", http_client=httpx.Client(transport=transport)) as client:
        with client.responses.stream(model="policy", input="hello") as stream:
            events = list(stream)
            terminal = events[-1].response
            assert terminal.status == ("incomplete" if outcome.finish_reason == "length" else "completed")
            if terminal.status == "completed":
                assert stream.get_final_response().id == terminal.id
            assert terminal.output_text == outcome.assistant_msg["content"]
            for event in events:
                if event.type == "response.output_item.done":
                    assert event.item.id == terminal.output[event.output_index].id


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_responses_handler_waits_for_generation_before_returning(stream):
    from uni_agent.gateway.gateway import _GatewayActor

    started, release = asyncio.Event(), asyncio.Event()

    async def generate(*args):
        started.set()
        await release.wait()
        return _outcome(content="done", finish_reason="stop")

    actor = object.__new__(_GatewayActor)
    generate_mock = AsyncMock(side_effect=generate)
    actor._sessions = {"s1": SimpleNamespace(sampling_params={}, run_generation=generate_mock)}
    actor._allowed_request_sampling_param_keys = ALLOWED_SAMPLING_KEYS
    actor._backend = object()
    task = asyncio.create_task(actor._handle_openai_responses("s1", _request(stream=stream)))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        assert not task.done()
    finally:
        release.set()
        await task
    generate_mock.assert_awaited_once()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_responses_generation_errors_precede_stream_headers(stream):
    from uni_agent.gateway.gateway import _GatewayActor

    actor = object.__new__(_GatewayActor)
    generate = AsyncMock(side_effect=RuntimeError("backend unavailable"))
    actor._sessions = {"s1": SimpleNamespace(sampling_params={}, run_generation=generate)}
    actor._allowed_request_sampling_param_keys = ALLOWED_SAMPLING_KEYS
    actor._backend = object()
    with pytest.raises(RuntimeError, match="backend unavailable"):
        await actor._handle_openai_responses("s1", _request(stream=stream))
    generate.assert_awaited_once()


def test_responses_input_accumulates_assistant_fragments_in_order():
    from uni_agent.gateway.adapters.openai_response import responses_to_internal

    internal = responses_to_internal(
        _request(
            input=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "plan"}]},
                {"type": "function_call", "call_id": "call-1", "name": "first", "arguments": "{}"},
                {"type": "custom_tool_call", "call_id": "call-2", "name": "exec", "input": "pwd"},
                {"role": "assistant", "content": "b"},
            ]
        ),
        base_sampling_params={},
        allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
    )
    assistant = internal["messages"][-1]
    assert assistant["content"] == "ab"
    assert assistant["reasoning_content"] == "plan"
    assert [call["id"] for call in assistant["tool_calls"]] == ["call-1", "call-2"]


@pytest.mark.parametrize(
    "boundary",
    [
        {"type": "custom_tool_call_output", "call_id": "call-1", "output": "ok"},
        {"role": "user", "content": "compacted context"},
        {"role": "system", "content": "policy"},
    ],
)
def test_responses_keeps_assistant_groups_across_input_boundaries(boundary):
    from uni_agent.gateway.adapters.openai_response import responses_to_internal

    internal = responses_to_internal(
        _request(input=[{"role": "assistant", "content": "a"}, boundary, {"role": "assistant", "content": "b"}]),
        base_sampling_params={},
        allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
    )
    assert [m["content"] for m in internal["messages"] if m["role"] == "assistant"] == ["a", "b"]


def test_responses_system_normalization_preserves_mixed_image_content():
    from copy import deepcopy

    from uni_agent.gateway.adapters.openai_response import responses_to_internal

    payload = _request(
        instructions="policy",
        input=[
            {"role": "system", "content": ""},
            {"role": "developer", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]},
            {"role": "system", "content": "last"},
            {"role": "user", "content": "q"},
        ],
    )
    original = deepcopy(payload)
    internal = responses_to_internal(payload, base_sampling_params={}, allowed_sampling_keys=ALLOWED_SAMPLING_KEYS)
    assert payload == original
    assert internal["messages"][0] == {
        "role": "system",
        "content": [
            {"type": "text", "text": "policy"},
            {"type": "text", "text": "\n"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "\n"},
            {"type": "text", "text": "last"},
        ],
    }


@pytest.mark.parametrize("choice", ["auto", "AUTO", "none", "NoNe"])
def test_responses_tool_choice_strings_are_normalized(choice):
    from uni_agent.gateway.adapters.openai_response import responses_build_response, responses_to_internal

    payload = _request(tool_choice=choice)
    internal = responses_to_internal(payload, base_sampling_params={}, allowed_sampling_keys=ALLOWED_SAMPLING_KEYS)
    assert (internal["tools"] is None) == (choice.lower() == "none")
    body = responses_build_response(_outcome(content="done", finish_reason="stop"), payload=payload, model="policy")
    assert body["tool_choice"] == choice.lower()


@pytest.mark.parametrize(
    "choice",
    [
        "required",
        "unknown",
        {"type": "auto"},
        {"type": "none"},
        {"type": "function", "name": "exec"},
        {"type": "custom", "name": "exec"},
        None,
        [],
        1,
    ],
)
def test_responses_rejects_unsupported_tool_choice_shapes(choice):
    from uni_agent.gateway.adapters.openai_response import responses_to_internal
    from uni_agent.gateway.adapters.types import MalformedRequestError

    with pytest.raises(MalformedRequestError, match="tool_choice"):
        responses_to_internal(
            _request(tool_choice=choice), base_sampling_params={}, allowed_sampling_keys=ALLOWED_SAMPLING_KEYS
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_tokens": 64},
        {"max_completion_tokens": 64},
        {"max_completion_tokens": None},
        {"max_tokens": None},
        {"max_tokens": 64, "max_output_tokens": None},
        {"max_tokens": 128, "max_output_tokens": 128},
    ],
)
def test_responses_rejects_chat_token_limit_field(overrides):
    from uni_agent.gateway.adapters.openai_response import responses_to_internal
    from uni_agent.gateway.adapters.types import MalformedRequestError

    with pytest.raises(MalformedRequestError, match="use max_output_tokens"):
        responses_to_internal(
            _request(**overrides), base_sampling_params={}, allowed_sampling_keys=ALLOWED_SAMPLING_KEYS
        )


@pytest.mark.parametrize("limit,expected", [(128, 128), (None, 2048)])
def test_responses_maps_only_canonical_output_token_limit(limit, expected):
    from uni_agent.gateway.adapters.openai_response import responses_to_internal

    internal = responses_to_internal(
        _request(max_output_tokens=limit),
        base_sampling_params={"max_tokens": 2048},
        allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
    )
    assert internal["sampling_params"]["max_tokens"] == expected


@pytest.mark.parametrize(
    "role,expected", [("system", "system"), ("developer", "system"), ("user", "user"), ("assistant", "assistant")]
)
def test_responses_accepts_standard_message_roles(role, expected):
    from uni_agent.gateway.adapters.openai_response import responses_to_internal

    internal = responses_to_internal(
        _request(instructions=None, input=[{"type": "message", "role": role, "content": "body"}]),
        base_sampling_params={},
        allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
    )
    assert any(message["role"] == expected and message["content"] == "body" for message in internal["messages"])


@pytest.mark.parametrize(
    "role", ["moderator", "tool", "USER", "", None, {}, [], pytest.param(_MISSING_ROLE, id="missing")]
)
def test_responses_rejects_unknown_or_invalid_message_roles(role):
    from uni_agent.gateway.adapters.openai_response import responses_to_internal
    from uni_agent.gateway.adapters.types import MalformedRequestError

    message = {"type": "message", "content": "body", "tool_call_id": "call-1"}
    if role is not _MISSING_ROLE:
        message["role"] = role
    with pytest.raises(MalformedRequestError, match=r"input\[0\].role"):
        responses_to_internal(
            _request(input=[message]),
            base_sampling_params={},
            allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
        )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "overrides",
    [
        {"parallel_tool_calls": False},
        {"tool_choice": {"type": "auto"}},
        {"max_tokens": 128},
        {"input": [{"role": "moderator", "content": "body"}]},
    ],
)
@pytest.mark.asyncio
async def test_responses_capability_errors_precede_generation(stream, overrides):
    from fastapi import HTTPException

    from uni_agent.gateway.gateway import _GatewayActor

    generate = AsyncMock(return_value=_outcome(content="done", finish_reason="stop"))
    actor = object.__new__(_GatewayActor)
    actor._sessions = {"s1": SimpleNamespace(sampling_params={}, run_generation=generate)}
    actor._allowed_request_sampling_param_keys = ALLOWED_SAMPLING_KEYS
    actor._backend = object()
    with pytest.raises(HTTPException) as error:
        await actor._handle_openai_responses("s1", _request(stream=stream, **overrides))
    assert error.value.status_code == 400
    generate.assert_not_called()


@pytest.mark.parametrize("item_type", ["function_call_output", "custom_tool_call_output"])
def test_tool_output_images_remain_chat_content_parts(item_type):
    from copy import deepcopy

    from uni_agent.gateway.adapters.openai_response import responses_to_internal

    output = [
        {"type": "input_text", "text": "Screenshot:"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        {"type": "input_text", "text": "Second image:"},
        {"type": "input_image", "data": "BBBB", "mime_type": "image/png"},
    ]
    payload = _request(
        input=[{"role": "user", "content": "inspect"}, {"type": item_type, "call_id": "call-image", "output": output}]
    )
    original = deepcopy(payload)
    request = responses_to_internal(payload, base_sampling_params={}, allowed_sampling_keys=ALLOWED_SAMPLING_KEYS)
    assert request["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-image",
        "content": [
            {"type": "text", "text": "Screenshot:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "Second image:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
        ],
    }
    assert payload == original


@pytest.mark.parametrize(
    "output,expected",
    [
        ("unchanged", "unchanged"),
        ('{"type":"input_image","image_url":"literal text"}', '{"type":"input_image","image_url":"literal text"}'),
        ([{"type": "input_text", "text": "one"}, {"type": "input_text", "text": "two"}], "onetwo"),
        ([], ""),
        ({"count": 1}, '{"count": 1}'),
    ],
)
def test_tool_output_text_and_json_results_keep_their_text_semantics(output, expected):
    from uni_agent.gateway.adapters.openai_response import responses_to_internal

    request = responses_to_internal(
        _request(input=[{"type": "function_call_output", "call_id": "call-1", "output": output}]),
        base_sampling_params={},
        allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
    )
    assert request["messages"][0 if request["messages"][0]["role"] == "tool" else 1]["content"] == expected


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "output",
    [
        [{"type": "input_image", "file_id": "file-unsupported"}],
        [{"type": "input_image"}],
        [{"type": "input_file", "file_data": "base64", "text": "must not hide the file"}],
        [{"type": "unknown", "text": "must not hide unsupported content"}],
        ["not a content object"],
        [{"type": []}],
        [{"type": {}}],
        {"type": []},
        {"type": {}},
        {"type": "input_image", "image_url": "https://example.test/image.png"},
        {"type": "computer_screenshot", "image_url": "https://example.test/image.png"},
    ],
)
def test_unsupported_tool_output_media_is_rejected_before_generation(output, stream):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from uni_agent.gateway.gateway import _GatewayActor

    async def run():
        session = SimpleNamespace(
            sampling_params={},
            run_generation=AsyncMock(return_value=_outcome(content="done", finish_reason="stop")),
        )
        gateway = _GatewayActor.__new__(_GatewayActor)
        gateway._sessions = {"s1": session}
        gateway._allowed_request_sampling_param_keys = ALLOWED_SAMPLING_KEYS
        gateway._backend = object()
        gateway._app = FastAPI()
        gateway._register_routes()
        async with AsyncClient(transport=ASGITransport(app=gateway._app), base_url="http://test") as client:
            response = await client.post(
                "/sessions/s1/v1/responses",
                json=_request(
                    stream=stream,
                    input=[
                        {"role": "user", "content": "inspect"},
                        {"type": "function_call_output", "call_id": "call-1", "output": output},
                    ],
                ),
            )
        assert response.status_code == 400
        assert "input[1].output" in response.json()["error"]["message"]
        session.run_generation.assert_not_awaited()

    asyncio.run(run())


async def _assert_responses_request_fails_before_generation(payload):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from uni_agent.gateway.gateway import _GatewayActor

    generate = AsyncMock(return_value=_outcome(content="done", finish_reason="stop"))
    actor = object.__new__(_GatewayActor)
    actor._sessions = {"s1": SimpleNamespace(sampling_params={}, run_generation=generate)}
    actor._allowed_request_sampling_param_keys = ALLOWED_SAMPLING_KEYS
    actor._backend = object()
    actor._app = FastAPI()
    actor._register_routes()
    async with AsyncClient(transport=ASGITransport(app=actor._app), base_url="http://test") as client:
        response = await client.post("/sessions/s1/v1/responses", json=payload)
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    generate.assert_not_called()
    return error["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("item_type", ["compaction", "compaction_summary"])
@pytest.mark.parametrize("encrypted", [False, True])
async def test_remote_compaction_items_are_rejected_before_generation(stream, item_type, encrypted):
    item = {"type": item_type}
    if encrypted:
        item["encrypted_content"] = "opaque-content-must-not-be-echoed"
    message = await _assert_responses_request_fails_before_generation(
        _request(stream=stream, input=[{"role": "user", "content": "q"}, item])
    )
    assert item_type in message and "input[1]" in message
    assert "opaque-content-must-not-be-echoed" not in message


@pytest.mark.parametrize("notice", ["context_compaction", "compaction_trigger"])
def test_compaction_notifications_do_not_split_assistant_fragments(notice):
    from uni_agent.gateway.adapters.openai_response import responses_to_internal

    request = responses_to_internal(
        _request(
            input=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
                {"type": notice},
                {"role": "assistant", "content": "b"},
            ]
        ),
        base_sampling_params={},
        allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
    )
    assert [m["content"] for m in request["messages"] if m["role"] == "assistant"] == ["ab"]


def test_text_compaction_summary_remains_an_ordinary_user_message():
    from uni_agent.gateway.adapters.openai_response import responses_to_internal

    summary = "Compacted context:\nKeep this previous progress and tool evidence."
    request = responses_to_internal(
        _request(input=[{"role": "user", "content": summary}]),
        base_sampling_params={},
        allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
    )
    assert request["messages"][-1] == {"role": "user", "content": summary}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location,with_source",
    [
        ("summary", False),
        ("reasoning_content", True),
        ("summary_text", True),
        ("output", True),
        ("text", True),
        ("tool_text", True),
        ("image_with_output", True),
    ],
)
async def test_images_in_text_only_fields_are_rejected_without_omission(location, with_source):
    image = {"type": "input_image"}
    if with_source:
        image["image_url"] = "data:image/png;base64,AAAA"
    if location == "summary":
        item = {"type": "reasoning", "summary": [image]}
    elif location == "reasoning_content":
        item = {"type": "reasoning", "content": [image]}
    elif location == "summary_text":
        item = {"type": "reasoning", "summary": [{"type": "summary_text", "text": [image]}]}
    elif location == "output":
        item = {"role": "user", "content": [{"output": [image]}]}
    elif location == "text":
        item = {"role": "user", "content": [{"type": "input_text", "text": [image]}]}
    elif location == "tool_text":
        item = {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": [{"type": "input_text", "text": [image]}],
        }
    else:
        item = {"type": "reasoning", "summary": {**image, "output": "must-not-bypass-image-validation"}}
    message = await _assert_responses_request_fails_before_generation(
        _request(input=[{"role": "user", "content": "q"}, item])
    )
    assert "input_image" in message and "input[1]" in message
    assert "[Image omitted]" not in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "placement,reverse,deferred",
    [("flat", False, False), ("namespace", False, False), ("separate_namespaces", False, False), ("flat", True, True)],
)
async def test_conflicting_tool_kinds_are_rejected_before_generation(placement, reverse, deferred):
    tools = [{"type": "function", "name": "same", "parameters": {"type": "object"}}, {"type": "custom", "name": "same"}]
    if deferred:
        tools[0]["defer_loading"] = True
    if reverse:
        tools.reverse()
    if placement == "namespace":
        tools = [{"type": "namespace", "name": "group", "tools": tools}]
    elif placement == "separate_namespaces":
        tools = [{"type": "namespace", "name": "group", "tools": [tool]} for tool in tools]
    message = await _assert_responses_request_fails_before_generation(_request(tools=tools))
    assert "tool kind" in message and "same" in message


@pytest.mark.parametrize("kind", ["function", "custom"])
def test_duplicate_tool_declarations_of_the_same_kind_keep_existing_behavior(kind):
    from uni_agent.gateway.adapters.openai_response import responses_to_internal

    declaration = {"type": kind, "name": "same"}
    request = responses_to_internal(
        _request(tools=[declaration, dict(declaration)]),
        base_sampling_params={},
        allowed_sampling_keys=ALLOWED_SAMPLING_KEYS,
    )
    assert [tool["function"]["name"] for tool in request["tools"]] == ["same", "same"]


def test_responses_forwards_configured_sampling_keys_without_mutating_inputs():
    from uni_agent.gateway.adapters.openai_response import responses_to_internal

    payload = _request(presence_penalty=1.5, seed=42, frequency_penalty=1.8, temperature=0.25)
    original = copy.deepcopy(payload)
    base = {"presence_penalty": 0, "seed": 1, "frequency_penalty": 0.3, "temperature": 0.7, "top_p": 0.9}
    allowed = frozenset({"presence_penalty", "seed", "max_tokens", "top_p"})
    request = responses_to_internal(payload, base_sampling_params=base, allowed_sampling_keys=allowed)

    assert request["sampling_params"] == {
        "presence_penalty": 1.5,
        "seed": 42,
        "frequency_penalty": 0.3,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 128,
    }
    assert payload == original
    assert base == {"presence_penalty": 0, "seed": 1, "frequency_penalty": 0.3, "temperature": 0.7, "top_p": 0.9}


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, {"presence_penalty": 0.5, "seed": 1}),
        ({"presence_penalty": 0, "seed": 0}, {"presence_penalty": 0, "seed": 0}),
        ({"presence_penalty": None, "seed": None}, {"presence_penalty": None, "seed": None}),
    ],
    ids=["missing", "zero", "null"],
)
def test_responses_sampling_merge_matches_chat(overrides, expected):
    from uni_agent.gateway.adapters.openai_chat import openai_to_internal
    from uni_agent.gateway.adapters.openai_response import responses_to_internal

    base = {"presence_penalty": 0.5, "seed": 1}
    allowed = frozenset(base)
    responses = responses_to_internal(_request(**overrides), base_sampling_params=base, allowed_sampling_keys=allowed)
    chat = openai_to_internal(
        {"messages": [{"role": "user", "content": "q"}], **overrides},
        base_sampling_params=base,
        allowed_sampling_keys=allowed,
    )
    assert responses["sampling_params"] == chat["sampling_params"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_responses_allowed_sampling_fields_reach_generation_like_chat(stream):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from uni_agent.gateway.gateway import _GatewayActor

    base = {"presence_penalty": 0, "seed": 1, "frequency_penalty": 0.3}
    generate = AsyncMock(return_value=_outcome(content="done", finish_reason="stop"))
    actor = object.__new__(_GatewayActor)
    actor._sessions = {"s1": SimpleNamespace(sampling_params=base, run_generation=generate)}
    actor._warned_discarded_request_sampling_param_keys = set()
    actor._allowed_request_sampling_param_keys = ALLOWED_SAMPLING_KEYS | {"presence_penalty", "seed"}
    actor._backend = object()
    actor._app = FastAPI()
    actor._register_routes()
    values = {"presence_penalty": 1.5, "seed": 42, "frequency_penalty": 1.8, "stream": stream}
    async with AsyncClient(transport=ASGITransport(app=actor._app), base_url="http://test") as client:
        response = await client.post("/sessions/s1/v1/responses", json=_request(**values))
        chat = await client.post(
            "/sessions/s1/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "q"}], "max_tokens": 128, **values},
        )
    assert response.status_code == chat.status_code == 200
    assert generate.await_count == 2
    expected = {"presence_penalty": 1.5, "seed": 42, "frequency_penalty": 0.3, "max_tokens": 128}
    assert [call.args[0]["sampling_params"] for call in generate.call_args_list] == [expected, expected]
    assert base == {"presence_penalty": 0, "seed": 1, "frequency_penalty": 0.3}
