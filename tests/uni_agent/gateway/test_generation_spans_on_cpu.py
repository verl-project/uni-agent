"""Generation metadata must match the native token arrays exposed to consumers."""

from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from tests.uni_agent.support import FakeTokenizer, SequencedBackend
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.gateway import _GatewayActor
from uni_agent.gateway.session import GatewaySession, MessageCodec, SessionHandle
from uni_agent.gateway.session.tree_session import TreeGatewaySession
from uni_agent.gateway.session.types import AgentContext

TOOL_NAMES = ("lookup", "fetch")


@pytest.fixture(params=[GatewaySession, TreeGatewaySession], ids=["linear", "tree"])
def session_type(request):
    return request.param


@pytest.fixture
def synthetic_tool_parser(monkeypatch):
    async def extract(codec, response_ids, tools, parser_name):
        text = codec._tokenizer.decode(response_ids)
        if text == "~":
            names = ["lookup"]
        elif text.startswith("tool:"):
            names = text.removeprefix("tool:").split(",")
        else:
            return text, []
        return "", [SimpleNamespace(name=name, arguments={}) for name in names]

    monkeypatch.setattr(MessageCodec, "_extract_tool_calls", extract)


def assert_spans(trajectory, session_id, expected):
    coverage = [0] * len(trajectory.response_ids)
    last_end = 0
    for span, (index, text, tool_ids, context) in zip(
        trajectory.extra_fields["generation_spans"], expected, strict=True
    ):
        start, end = span["response_start"], span["response_end"]
        assert last_end <= start <= end <= len(coverage)
        assert start < end or tool_ids
        assert span == {
            "session_id": session_id,
            "chain_id": trajectory.chain_id,
            "generation_index": index,
            "response_start": start,
            "response_end": end,
            "tool_call_ids": tool_ids,
            "agent_context": context,
        }
        assert FakeTokenizer().decode(trajectory.response_ids[start:end]) == text
        coverage[start:end] = [1] * (end - start)
        last_end = end
    assert coverage == trajectory.response_mask
    assert trajectory.response_logprobs == [-0.1 if mask else 0.0 for mask in coverage]


async def generate(session, backend, messages, *, with_tools=False):
    tools = (
        [{"type": "function", "function": {"name": name, "parameters": {"type": "object"}}} for name in TOOL_NAMES]
        if with_tools
        else None
    )
    return await session.run_generation(
        {"messages": messages, "tools": tools, "sampling_params": {"max_tokens": 64, "logprobs": True}},
        backend,
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_span_uses_wire_tool_ids_and_native_token_ranges(session_type, monkeypatch, synthetic_tool_parser):
    monkeypatch.setattr("uni_agent.gateway.gateway.ray.util.get_node_ip_address", lambda: "127.0.0.1")
    tool_output = "tool:lookup,fetch"
    backend = SequencedBackend([tool_output, "answer"])
    actor = _GatewayActor(
        GatewayActorConfig(
            tokenizer=FakeTokenizer(),
            tool_parser_name="hermes",
            use_tree_trajectory_manager=session_type is TreeGatewaySession,
        ),
        backend,
    )
    context = {
        "provider": "claude_code",
        "session_id": "harness-session",
        "agent_id": "child-agent",
        "parent_agent_id": "parent-agent",
    }
    headers = {
        "x-claude-code-session-id": "harness-session",
        "x-claude-code-agent-id": "child-agent",
        "x-claude-code-parent-agent-id": "parent-agent",
    }
    request = {
        "model": "synthetic-model",
        "max_tokens": 64,
        "tools": [{"name": name, "input_schema": {"type": "object"}} for name in TOOL_NAMES],
        "messages": [{"role": "user", "content": "Use both tools"}],
    }
    await actor.start()
    try:
        handle = await actor.create_session("gateway-session", sampling_params={"logprobs": True})
        async with httpx.AsyncClient(timeout=5, trust_env=False, headers=headers) as client:
            endpoint = f"{handle.base_url}/messages"
            first = await client.post(endpoint, json=request)
            first.raise_for_status()
            blocks = first.json()["content"]
            assert [(block["type"], block["name"]) for block in blocks] == [("tool_use", name) for name in TOOL_NAMES]
            tool_ids = [block["id"] for block in blocks]
            assert len(set(tool_ids)) == len(TOOL_NAMES)
            request["messages"].extend(
                [
                    {"role": "assistant", "content": blocks},
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": tool_id, "content": "tool observation"}
                            for tool_id in reversed(tool_ids)
                        ],
                    },
                ]
            )
            second = await client.post(endpoint, json=request)
            second.raise_for_status()
            assert second.json()["content"] == [{"type": "text", "text": "answer"}]

        [trajectory] = await actor.finalize_session(handle.session_id)
        assert trajectory.prompt_ids + trajectory.response_ids == backend.calls[1]["prompt_ids"] + list(
            map(ord, "answer")
        )
        assert "tool observation" in FakeTokenizer().decode(trajectory.response_ids)
        assert_spans(trajectory, handle.session_id, [(0, tool_output, tool_ids, context), (1, "answer", [], context)])
    finally:
        await actor.shutdown()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_empty_generation_keeps_later_generation_index(session_type):
    session = session_type(SessionHandle("empty-generation"), MessageCodec(FakeTokenizer()))
    backend = SequencedBackend(["first", "", "third"])
    messages = [{"role": "user", "content": "start"}]
    for follow_up in ("continue", "try again"):
        outcome = await generate(session, backend, messages)
        messages.extend([outcome.assistant_msg, {"role": "user", "content": follow_up}])
    await generate(session, backend, messages)

    [trajectory] = await session.finalize()
    assert_spans(trajectory, session.handle.session_id, [(0, "first", [], {}), (2, "third", [], {})])


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_generation_span_snapshots_request_context(session_type):
    session = session_type(SessionHandle("context-copy"), MessageCodec(FakeTokenizer()))
    context = AgentContext(provider="test", session_id="original", agent_id=None, parent_agent_id=None)
    original_context = dict(context)
    await session.run_generation(
        {
            "messages": [{"role": "user", "content": "start"}],
            "tools": None,
            "sampling_params": {"logprobs": True},
            "agent_context": context,
        },
        SequencedBackend(["first"]),
    )
    context["session_id"] = "updated"

    [trajectory] = await session.finalize()
    assert_spans(trajectory, session.handle.session_id, [(0, "first", [], original_context)])


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("retry", ["failed", "successful", "capacity"])
async def test_assistant_rollback_tracks_only_surviving_generations(session_type, retry, synthetic_tool_parser):
    original_text = "tool:lookup"
    replacement_text = "tool:fetch"
    backend = SequencedBackend(["first", original_text, RuntimeError("backend failure"), replacement_text])
    session = session_type(SessionHandle("rewrite"), MessageCodec(FakeTokenizer(), tool_parser_name="hermes"))
    first_messages = [{"role": "user", "content": "start"}]
    first = await generate(session, backend, first_messages, with_tools=True)
    second_messages = [*first_messages, first.assistant_msg, {"role": "user", "content": "Use a tool"}]
    original = await generate(session, backend, second_messages, with_tools=True)
    [original_tool] = original.assistant_msg["tool_calls"]
    rewrite = [*second_messages, {"role": "user", "content": "Try again"}]
    with pytest.raises(HTTPException, match="backend failure"):
        await generate(session, backend, rewrite, with_tools=True)

    expected = [(0, "first", [], {})]
    if retry == "successful":
        replacement = await generate(session, backend, rewrite, with_tools=True)
        [replacement_tool] = replacement.assistant_msg["tool_calls"]
        expected.append((1, replacement_text, [replacement_tool["id"]], {}))
    elif retry == "capacity":
        session._trajectory_capacity = 1
        capped = await generate(session, backend, rewrite, with_tools=True)
        assert capped.finish_reason == "length"
        assert backend.steps == [replacement_text]
    else:
        expected.append((1, original_text, [original_tool["id"]], {}))

    [trajectory] = await session.finalize()
    assert_spans(trajectory, session.handle.session_id, expected)


class GLMTokenizer(FakeTokenizer):
    def convert_tokens_to_ids(self, token):
        return {"<|observation|>": ord("~"), "<|user|>": ord("`")}.get(token)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("first_text,with_tools", [("AB~", False), ("~", False), ("~", True)])
async def test_spans_match_final_tokens_after_continuous_token_trim(
    session_type, first_text, with_tools, synthetic_tool_parser
):
    session = session_type(
        SessionHandle("context-trim"),
        MessageCodec(GLMTokenizer(), hf_model_type="glm4_moe", tool_parser_name="hermes"),
    )
    backend = SequencedBackend([first_text, "C"])
    messages = [{"role": "user", "content": "start"}]
    first = await generate(session, backend, messages, with_tools=with_tools)
    await generate(
        session,
        backend,
        [*messages, first.assistant_msg, {"role": "user", "content": "continue"}],
        with_tools=with_tools,
    )

    [trajectory] = await session.finalize()
    assert ord("~") not in trajectory.response_ids
    surviving_text = first_text.removesuffix("~")
    tool_ids = [call["id"] for call in first.assistant_msg.get("tool_calls", [])]
    expected = [(0, surviving_text, tool_ids, {})] if surviving_text or tool_ids else []
    assert_spans(trajectory, session.handle.session_id, [*expected, (1, "C", [], {})])


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_tree_branch_and_rewrite_exclude_shared_prompt_generations():
    session = TreeGatewaySession(SessionHandle("tree-fork"), MessageCodec(FakeTokenizer()))
    backend = SequencedBackend(["shared", "original", "abandoned", "replacement", "continued"])
    first_messages = [{"role": "user", "content": "start"}]
    first = await generate(session, backend, first_messages)
    history = [*first_messages, first.assistant_msg]
    await generate(session, backend, [*history, {"role": "user", "content": "original branch"}])
    fork = [*history, {"role": "user", "content": "new branch"}]
    await generate(session, backend, fork)
    rewrite = [*fork, {"role": "user", "content": "rewrite branch reply"}]
    replacement = await generate(session, backend, rewrite)
    await generate(session, backend, [*rewrite, replacement.assistant_msg, {"role": "user", "content": "continue"}])

    trajectories = {trajectory.chain_id: trajectory for trajectory in await session.finalize()}
    assert set(trajectories) == {1, 2}
    assert_spans(trajectories[1], session.handle.session_id, [(0, "shared", [], {}), (1, "original", [], {})])
    assert "shared" in FakeTokenizer().decode(trajectories[2].prompt_ids)
    assert_spans(trajectories[2], session.handle.session_id, [(0, "replacement", [], {}), (1, "continued", [], {})])
