from __future__ import annotations

import pytest

from uni_agent.gateway.adapters.responses import responses_to_internal
from uni_agent.gateway.adapters.types import MalformedRequestError

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def _request(arguments):
    return {
        "instructions": "system",
        "input": [
            {"type": "message", "role": "user", "content": "inspect"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "exec_command",
                "arguments": arguments,
            },
        ],
    }


def test_responses_function_call_arguments_are_lowered_to_mapping():
    internal = responses_to_internal(
        _request('{"cmd":"true"}'),
        base_sampling_params={},
        allowed_sampling_keys=frozenset(),
    )

    arguments = internal["messages"][2]["tool_calls"][0]["function"]["arguments"]
    assert arguments == {"cmd": "true"}


@pytest.mark.parametrize("arguments", ["not-json", "[]", "null", 1])
def test_responses_rejects_non_object_function_call_arguments(arguments):
    with pytest.raises(MalformedRequestError, match="function_call.arguments"):
        responses_to_internal(
            _request(arguments),
            base_sampling_params={},
            allowed_sampling_keys=frozenset(),
        )
