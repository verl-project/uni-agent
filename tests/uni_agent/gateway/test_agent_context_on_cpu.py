"""Harness identity survives HTTP adaptation without mixing independent agents."""

import asyncio

import httpx
import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers

from tests.uni_agent.support import FakeTokenizer, RecordingConcurrentBackend, SequencedBackend
from uni_agent.gateway.adapters.agent_context import agent_context_from_headers
from uni_agent.gateway.adapters.types import MalformedRequestError
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.gateway import _GatewayActor

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def _headers(agent="child", parent=None, session="claude-session"):
    headers = {"x-claude-code-session-id": session}
    if agent is not None:
        headers["x-claude-code-agent-id"] = agent
    if parent is not None:
        headers["x-claude-code-parent-agent-id"] = parent
    return headers


@pytest.mark.parametrize(
    "agent, parent", [(None, None), ("a", None), ("a", "claude-session"), ("c", "a"), ("a\xa0b", None), ("a\xff", None)]
)
def test_header_identity_preserves_explicit_parent(agent, parent):
    headers = Headers({key.upper(): value for key, value in _headers(agent, parent).items()})
    assert agent_context_from_headers(headers) == {
        "provider": "claude_code",
        "session_id": "claude-session",
        "agent_id": agent,
        "parent_agent_id": parent,
    }


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"authorization": "test-only", "x-dynamo-session-id": "backend-owned"},
        {"x-claude-code-session-id": "  "},
    ],
)
def test_unrelated_or_empty_headers_have_no_agent_context(headers):
    assert agent_context_from_headers(Headers(headers)) is None


@pytest.mark.parametrize(
    "headers",
    [
        {"x-claude-code-agent-id": "orphan"},
        _headers("a", "a"),
        _headers(None, "a"),
        _headers("a\n"),
        _headers("a\x7f"),
        _headers("a\x80"),
        _headers("a\x81"),
        _headers("a\x85"),
        _headers("a\x9f"),
        _headers(session="s\x81"),
        _headers(parent="p\x81"),
        _headers("a" * 513),
    ],
)
def test_malformed_identity_is_rejected(headers):
    with pytest.raises(MalformedRequestError):
        agent_context_from_headers(Headers(headers))


class ContextBackend(SequencedBackend):
    supports_agent_context = True

    def __init__(self, steps):
        super().__init__(steps)
        self.contexts = []

    async def generate(self, *args, agent_context=None, **kwargs):
        self.contexts.append(agent_context.copy() if agent_context else None)
        if agent_context is not None:
            # A backend owns its argument; mutation must not alter the collector.
            agent_context["agent_id"] = "backend-local"
        return await super().generate(*args, **kwargs)


async def _actor(backend):
    actor = _GatewayActor(GatewayActorConfig(tokenizer=FakeTokenizer()), backend)
    actor._server_base_url = "http://test"
    await actor.create_session("rollout")
    return actor


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["messages", "chat/completions", "responses"])
@pytest.mark.parametrize("stream", [False, True])
async def test_http_identity_reaches_backend_and_trajectory(endpoint, stream):
    backend = ContextBackend(["OK", "PLAIN"])
    actor = await _actor(backend)
    headers = _headers("c", "a")
    headers["authorization"] = "test-only-not-forwarded"
    payload = {"model": "test", "stream": stream}
    payload["input" if endpoint == "responses" else "messages"] = [{"role": "user", "content": "hello"}]
    # The identity comes from the transport, not a model request body field.
    payload["agent_context"] = {"agent_id": "spoofed"}
    if endpoint == "messages":
        payload["max_tokens"] = 8
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=actor._app), base_url="http://test") as client:
        for request_headers in (headers, {}):
            response = await client.post(f"/sessions/rollout/v1/{endpoint}", json=payload, headers=request_headers)
            assert response.status_code == 200, response.text
            if stream:
                assert response.headers["content-type"].startswith("text/event-stream")
    expected = agent_context_from_headers(Headers(headers))
    assert backend.contexts == [expected, None]
    assert [call["request_id"] for call in backend.calls] == ["rollout", "rollout"]
    assert all("agent_context" not in call["sampling_params"] for call in backend.calls)
    trajectories = await actor.finalize_session("rollout")
    assert trajectories[0].extra_fields["agent_context"] == expected
    assert "agent_context" not in trajectories[1].extra_fields


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["messages", "chat/completions", "responses"])
@pytest.mark.parametrize("headers", [_headers("a", "a"), _headers("a\x81")])
async def test_invalid_identity_returns_provider_error_before_generation(endpoint, headers):
    backend = ContextBackend([])
    actor = await _actor(backend)
    payload = {"model": "test"}
    payload["input" if endpoint == "responses" else "messages"] = [{"role": "user", "content": "hello"}]
    if endpoint == "messages":
        payload["max_tokens"] = 8
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=actor._app), base_url="http://test") as client:
        raw_headers = [(name.encode("ascii"), value.encode("latin-1")) for name, value in headers.items()]
        response = await client.post(f"/sessions/rollout/v1/{endpoint}", json=payload, headers=raw_headers)
    assert response.status_code == 400
    assert "error" in response.json()
    assert backend.calls == []
    await actor.abort_session("rollout")


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", [{}, {"agent": "other"}, {"session": "other-session"}])
async def test_inflight_coalescing_is_scoped_to_agent_identity(identity):
    # This strict backend does not accept agent_context: identity isolation is
    # also useful when the backend has not opted into receiving it.
    backend = RecordingConcurrentBackend(["FIRST", "SECOND"], delay=0.05)
    actor = await _actor(backend)
    context = agent_context_from_headers(Headers(_headers()))
    other_context = agent_context_from_headers(Headers(_headers(**identity)))
    payload = {"messages": [{"role": "user", "content": "identical prompt"}]}
    await asyncio.wait_for(
        asyncio.gather(
            actor._handle_openai_chat_completions("rollout", payload, agent_context=context),
            actor._handle_openai_chat_completions("rollout", payload, agent_context=other_context),
        ),
        timeout=5,
    )
    expected_count = 2 if identity else 1
    assert len(backend.call_windows) == expected_count
    trajectories = await actor.finalize_session("rollout")
    assert len(trajectories) == expected_count
    assert trajectories[0].extra_fields["agent_context"] == context
    if not identity:
        assert trajectories[0].extra_fields["num_coalesced_requests"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("other_headers", [_headers("other"), _headers(session="other-session"), {}])
async def test_continuation_cannot_attach_to_another_agents_identical_history(other_headers):
    backend = SequencedBackend(["SAME", "SAME", "NEXT"])
    actor = await _actor(backend)
    context = agent_context_from_headers(Headers(_headers()))
    other_context = agent_context_from_headers(Headers(other_headers))
    prompt = [{"role": "user", "content": "hello"}]
    for current in (context, other_context):
        await actor._handle_openai_chat_completions("rollout", {"messages": prompt}, agent_context=current)
    await actor._handle_openai_chat_completions(
        "rollout",
        {"messages": [*prompt, {"role": "assistant", "content": "SAME"}, {"role": "user", "content": "next"}]},
        agent_context=context,
    )
    trajectories = await actor.finalize_session("rollout")
    own = next(t for t in trajectories if t.extra_fields.get("agent_context") == context)
    assert len(trajectories) == 2
    assert sum(own.response_mask) == len("SAMENEXT")
    assert len(own.response_ids) == len(own.response_mask)
    generated_ids = [token for token, mask in zip(own.response_ids, own.response_mask, strict=True) if mask]
    assert FakeTokenizer().decode(generated_ids) == "SAMENEXT"
    assert all(FakeTokenizer().decode(t.response_ids) == "SAME" for t in trajectories if t is not own)


@pytest.mark.asyncio
async def test_failed_generation_can_retry_with_the_same_identity():
    backend = ContextBackend([RuntimeError("temporary failure"), "OK"])
    actor = await _actor(backend)
    context = agent_context_from_headers(Headers(_headers()))
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    with pytest.raises(HTTPException, match="temporary failure"):
        await actor._handle_openai_chat_completions("rollout", payload, agent_context=context)
    await actor._handle_openai_chat_completions("rollout", payload, agent_context=context)
    trajectories = await actor.finalize_session("rollout")
    assert backend.contexts == [context, context]
    assert len(trajectories) == 1
    assert trajectories[0].extra_fields["agent_context"] == context
    assert FakeTokenizer().decode(trajectories[0].response_ids) == "OK"


@pytest.mark.asyncio
async def test_reused_harness_identity_remains_scoped_to_outer_rollout():
    backend = ContextBackend(["FIRST", "SECOND"])
    actor = await _actor(backend)
    await actor.create_session("another-rollout")
    context = agent_context_from_headers(Headers(_headers()))
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    trajectories = []
    for rollout in ("rollout", "another-rollout"):
        await actor._handle_openai_chat_completions(rollout, payload, agent_context=context)
        trajectories.extend(await actor.finalize_session(rollout))
    assert [call["request_id"] for call in backend.calls] == ["rollout", "another-rollout"]
    assert [FakeTokenizer().decode(t.response_ids) for t in trajectories] == ["FIRST", "SECOND"]
    assert all(t.extra_fields["agent_context"] == context for t in trajectories)
