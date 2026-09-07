from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.uni_agent.support import FakeTokenizer
from uni_agent.gateway.message_normalization import (
    canonicalize_messages,
    coalesce_consecutive_assistant_messages,
)
from uni_agent.gateway.session.codec import MessageCodec
from uni_agent.gateway.session.session import GatewaySession
from uni_agent.gateway.session.types import SessionHandle

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def _messages():
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_a", "type": "function", "function": {"name": "a", "arguments": "{}"}}],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_b", "type": "function", "function": {"name": "b", "arguments": "{}"}}],
        },
    ]


def test_coalescing_does_not_mutate_input_or_accumulate_on_reuse():
    messages = _messages()
    before = deepcopy(messages)

    first = coalesce_consecutive_assistant_messages(messages)
    second = coalesce_consecutive_assistant_messages(messages)

    assert messages == before
    assert first == second
    assert [call["id"] for call in first[0]["tool_calls"]] == ["call_a", "call_b"]


def test_canonicalizing_an_existing_result_is_idempotent():
    messages = _messages()

    normalized = canonicalize_messages(messages)
    again = canonicalize_messages(normalized)

    assert normalized == again
    assert messages == _messages()


def test_encode_incremental_reuse_preserves_token_ids_and_messages():
    messages = [
        {"role": "assistant", "content": "", "reasoning_content": "think"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": {}}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    before = deepcopy(messages)
    codec = MessageCodec(FakeTokenizer())

    first = codec.encode_incremental(messages)
    second = codec.encode_incremental(messages)

    assert first == second
    assert messages == before


class _SessionBackend:
    def __init__(self):
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        token = ord("A") + len(self.calls) - 1
        return SimpleNamespace(token_ids=[token], stop_reason="stop", extra_fields={}, log_probs=None)


@pytest.mark.asyncio
async def test_session_continuation_reuses_chain_without_mutating_request_history():
    codec = MessageCodec(FakeTokenizer())
    backend = _SessionBackend()
    session = GatewaySession(SessionHandle("r2"), codec)
    first_request = {"messages": [{"role": "user", "content": "start"}], "tools": None, "sampling_params": {}}
    second_request = {
        "messages": [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "continue"},
        ],
        "tools": None,
        "sampling_params": {},
    }
    first_before = deepcopy(first_request)
    second_before = deepcopy(second_request)

    await session.run_generation(first_request, backend)
    await session.run_generation(second_request, backend)

    assert first_request == first_before
    assert second_request == second_before
    assert len(session.active_chains) == 1
    assert session.active_chains[0].buffer.response_mask[-1] == 1
    assert 0 in session.active_chains[0].buffer.response_mask
    assert backend.calls[1]["sampling_params"] == {}
