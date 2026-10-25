import json
from types import SimpleNamespace

import httpx
import pytest

from tests.uni_agent.support import FakeTokenizer
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.gateway import _GatewayActor
from uni_agent.gateway.session import GatewaySession


class Backend:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[str, dict]] = []

    async def generate(self, request_id: str, **kwargs) -> SimpleNamespace:
        self.calls.append((request_id, kwargs))
        ids = [ord(c) for c in self.text]
        return SimpleNamespace(
            token_ids=ids,
            log_probs=[-0.1] * len(ids),
            stop_reason="stop",
            extra_fields={"min_global_steps": 2, "max_global_steps": 2},
        )


def make_actor() -> tuple[_GatewayActor, dict[str, Backend]]:
    backends = {"p1": Backend("plan"), "p2": Backend("answer")}
    actor = _GatewayActor(
        config={
            p: GatewayActorConfig(
                tokenizer=FakeTokenizer(), prompt_length=1000, response_length=100
            )
            for p in backends
        },
        backend=backends,
    )
    actor._server_base_url = "http://test"
    return actor, backends


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_each_role_session_has_its_own_url_and_policy_backend():
    actor, backends = make_actor()
    planner = await actor.create_session(
        "r:planner",
        policy_name="p1",
        sampling_params={"temperature": 0.7, "logprobs": True},
    )
    answerer = await actor.create_session(
        "r:answerer",
        policy_name="p2",
        sampling_params={"temperature": 1.0, "logprobs": True},
    )
    assert planner.base_url.endswith("/sessions/r:planner/v1")
    assert answerer.base_url.endswith("/sessions/r:answerer/v1")
    assert planner.base_url != answerer.base_url
    assert all(isinstance(s, GatewaySession) for s in actor._sessions.values())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=actor._app), base_url="http://test"
    ) as client:
        for handle, role in ((planner, "planner"), (answerer, "answerer")):
            response = await client.post(
                f"{handle.base_url}/chat/completions",
                json={
                    "model": role,
                    "messages": [{"role": "user", "content": "question"}],
                },
            )
            assert response.status_code == 200, response.text

    assert len(backends["p1"].calls) == len(backends["p2"].calls) == 1
    assert backends["p1"].calls[0][1]["sampling_params"]["temperature"] == 0.7
    assert backends["p2"].calls[0][1]["sampling_params"]["temperature"] == 1.0

    planner_trajectories = await actor.finalize_session("r:planner")
    answerer_trajectories = await actor.finalize_session("r:answerer")
    assert len(planner_trajectories) == len(answerer_trajectories) == 1
    assert not actor._sessions


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_roles_may_share_a_policy_with_independent_sessions():
    actor, backends = make_actor()
    first = await actor.create_session("r:agent_1", policy_name="p1")
    second = await actor.create_session("r:agent_2", policy_name="p1")
    assert first.base_url != second.base_url
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=actor._app), base_url="http://test"
    ) as client:
        for handle in (first, second):
            response = await client.post(
                f"{handle.base_url}/chat/completions",
                json={"messages": [{"role": "user", "content": "q"}]},
            )
            assert response.status_code == 200
    assert backends["p1"].calls[0][0] == "r:agent_1"
    assert backends["p1"].calls[1][0] == "r:agent_2"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_unknown_policy_is_rejected():
    actor, _ = make_actor()
    with pytest.raises(ValueError, match="Unknown policy"):
        await actor.create_session("r:a", policy_name="missing")
    assert not actor._sessions


@pytest.mark.cpu
@pytest.mark.level0
def test_allowed_sampling_keys_are_precomputed_per_policy():
    configs = {
        "p1": GatewayActorConfig(
            tokenizer=FakeTokenizer(),
            allowed_request_sampling_param_keys={"temperature"},
        ),
        "p2": GatewayActorConfig(
            tokenizer=FakeTokenizer(),
            allowed_request_sampling_param_keys={"top_p"},
        ),
    }
    actor = _GatewayActor(
        config=configs,
        backend={"p1": Backend("plan"), "p2": Backend("answer")},
    )

    assert actor._policy_allowed_request_sampling_param_keys == {
        "p1": {"max_tokens", "stop", "temperature"},
        "p2": {"max_tokens", "stop", "top_p"},
    }
    assert all(
        isinstance(keys, frozenset)
        for keys in actor._policy_allowed_request_sampling_param_keys.values()
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_create_session_requires_started_server():
    actor, _ = make_actor()
    actor._server_base_url = None

    with pytest.raises(RuntimeError, match="start\\(\\) must be called"):
        await actor.create_session("r:planner", policy_name="p1")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_unhandled_route_errors_use_openai_error_shape(monkeypatch, caplog):
    actor, _ = make_actor()
    handle = await actor.create_session("r:planner", policy_name="p1")

    async def fail_completion(*args, **kwargs):
        raise RuntimeError("unexpected route failure")

    monkeypatch.setattr(actor, "_handle_openai_chat_completions", fail_completion)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=actor._app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"{handle.base_url}/chat/completions",
            json={"messages": []},
        )
    assert response.status_code == 500
    assert response.json()["error"]["message"] == "Internal server error"
    assert "unexpected route failure" not in response.text

    assert any(record.exc_info for record in caplog.records)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_openai_stream_and_http_errors():
    actor, _ = make_actor()
    handle = await actor.create_session("r:planner", policy_name="p1")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=actor._app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"{handle.base_url}/chat/completions",
            json={
                "model": "planner",
                "messages": [{"role": "user", "content": "question"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "data: [DONE]" in response.text
        chunks = [
            json.loads(line[len("data: ") :])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert all(chunk["model"] == "planner" for chunk in chunks)
        assert (
            "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
            == "plan"
        )
        invalid = await client.post(
            f"{handle.base_url}/chat/completions",
            content="{",
            headers={"Content-Type": "application/json"},
        )
        assert invalid.status_code == 400
        assert invalid.json()["error"]["message"] == "Invalid JSON body"
        # Anthropic Messages is served by the shared core; an empty payload
        # is rejected as a malformed request.
        unsupported = await client.post(
            f"{handle.base_url}/messages", json={"model": "planner", "messages": []}
        )
        assert unsupported.status_code == 400
    await actor.abort_session("r:planner")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_ignored_sampling_parameter_warning_is_emitted_once(caplog):
    actor, _ = make_actor()
    planner = await actor.create_session(
        "r:planner",
        policy_name="p1",
        sampling_params={"temperature": 0.7, "top_p": 0.9},
    )
    answerer = await actor.create_session(
        "r:answerer",
        policy_name="p2",
        sampling_params={"temperature": 0.8, "top_p": 0.9},
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=actor._app),
        base_url="http://test",
    ) as client:
        for handle in (planner, answerer, planner, answerer):
            response = await client.post(
                f"{handle.base_url}/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "question"}],
                    "temperature": 0.2,
                    "top_p": 0.5,
                },
            )
            assert response.status_code == 200

    assert actor._warned_discarded_request_sampling_param_keys == {
        ("p1", "temperature"),
        ("p1", "top_p"),
        ("p2", "temperature"),
        ("p2", "top_p"),
    }
    warnings = [
        record
        for record in caplog.records
        if "Ignoring request sampling parameters" in record.message
    ]
    assert len(warnings) == 2
    assert all("temperature, top_p" in record.message for record in warnings)
    assert "policy p1" in warnings[0].message
    assert "policy p2" in warnings[1].message


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_invalid_json_shapes_return_bad_request():
    actor, _ = make_actor()
    handle = await actor.create_session("r:planner", policy_name="p1")
    async with httpx.AsyncClient(
        # The non-object body raises inside the handler and is converted by
        # the app's exception handler; let the transport read that response.
        transport=httpx.ASGITransport(app=actor._app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        non_object = await client.post(
            f"{handle.base_url}/chat/completions", json=["not", "an", "object"]
        )
        unhashable_role = await client.post(
            f"{handle.base_url}/chat/completions",
            json={"model": ["planner"], "messages": []},
        )
        non_string_role = await client.post(
            f"{handle.base_url}/chat/completions",
            json={"model": 3, "messages": []},
        )

    # A non-object body surfaces as an internal error, matching the
    # pre-refactor actor. ``model`` is no longer a routing key, so the two
    # remaining 400s come from protocol validation (empty ``messages``).
    assert [
        non_object.status_code,
        unhashable_role.status_code,
        non_string_role.status_code,
    ] == [500, 400, 400]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_failed_finalize_leaves_session_retryable_then_abortable(
    monkeypatch, caplog
):
    actor, _ = make_actor()
    await actor.create_session("r:planner", policy_name="p1")
    failing_session = actor._sessions["r:planner"]

    async def fail_finalize():
        raise RuntimeError("finalize failed")

    monkeypatch.setattr(failing_session, "finalize", fail_finalize)
    with pytest.raises(RuntimeError, match="finalize failed"):
        await actor.finalize_session("r:planner")
    assert "r:planner" in actor._sessions
    await actor.abort_session("r:planner")
    assert not actor._sessions


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("max_tokens", [None, True, 0, -1])
async def test_invalid_max_tokens_returns_openai_bad_request(max_tokens):
    actor, _ = make_actor()
    handle = await actor.create_session("r:planner", policy_name="p1")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=actor._app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"{handle.base_url}/chat/completions",
            json={
                "model": "planner",
                "messages": [{"role": "user", "content": "question"}],
                "max_tokens": max_tokens,
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "max_tokens must be a positive integer"
    )
