from types import SimpleNamespace

import pytest

from tests.uni_agent.support import FakeTokenizer
from uni_agent.gateway.adapters.responses import responses_build_response, responses_to_internal
from uni_agent.gateway.session.codec import MessageCodec

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_responses_input_lowering():
    internal = responses_to_internal(
        {
            "instructions": "system rule",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix"}]},
                {"type": "function_call", "call_id": "call_1", "name": "exec_command", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
            ],
            "tools": [
                {"type": "function", "name": "exec_command", "description": "run", "parameters": {"type": "object"}},
                {"type": "web_search", "external_web_access": True},
            ],
            "max_output_tokens": 64,
        },
        base_sampling_params={},
        allowed_sampling_keys=frozenset({"max_tokens"}),
    )
    assert internal["messages"][0] == {"role": "system", "content": "system rule"}
    assert internal["messages"][1]["content"] == "fix"
    assert internal["messages"][2]["tool_calls"][0]["function"]["name"] == "exec_command"
    assert internal["messages"][3]["role"] == "tool"
    assert internal["sampling_params"]["max_tokens"] == 64
    assert internal["tools"] == [
        {
            "type": "function",
            "function": {"name": "exec_command", "description": "run", "parameters": {"type": "object"}},
        }
    ]


def test_responses_continuation_coalesces_reasoning_and_function_call():
    internal = responses_to_internal(
        {
            "instructions": "system rule",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "inspect"}]},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "inspect first"}]},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": '{"cmd":"find /testbed"}',
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "file.py"},
            ],
        },
        base_sampling_params={},
        allowed_sampling_keys=frozenset(),
    )

    continuation = internal["messages"][2:]
    assert [message["role"] for message in continuation] == ["assistant", "tool"]
    assert continuation[0]["reasoning_content"] == "inspect first"
    assert continuation[0]["tool_calls"][0]["function"]["name"] == "exec_command"
    MessageCodec(FakeTokenizer()).encode_incremental(continuation)


def test_responses_canonicalizes_developer_and_late_system_messages():
    internal = responses_to_internal(
        {
            "instructions": "base system",
            "input": [
                {"type": "message", "role": "user", "content": "inspect"},
                {"type": "message", "role": "developer", "content": "late system"},
            ],
        },
        base_sampling_params={},
        allowed_sampling_keys=frozenset(),
    )
    assert [message["role"] for message in internal["messages"]] == ["system", "user"]
    assert internal["messages"][0]["content"] == "base system\nlate system"


def test_responses_build_response_for_text_and_tool_call():
    text = responses_build_response(
        SimpleNamespace(
            assistant_msg={"role": "assistant", "content": "fixed"},
            finish_reason="stop",
            prompt_tokens=3,
            completion_tokens=2,
        ),
        model="policy",
    )
    assert text["object"] == "response"
    assert text["output_text"] == "fixed"
    assert text["output"][0]["type"] == "message"

    tool = responses_build_response(
        SimpleNamespace(
            assistant_msg={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "exec_command", "arguments": "{}"}}
                ],
            },
            finish_reason="tool_calls",
            prompt_tokens=3,
            completion_tokens=2,
        ),
        model="policy",
    )
    assert tool["output"][0]["type"] == "function_call"
    assert tool["output"][0]["call_id"] == "call_1"
