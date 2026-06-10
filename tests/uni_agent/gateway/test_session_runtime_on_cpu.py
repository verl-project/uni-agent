from __future__ import annotations

import httpx
import pytest
import ray

from tests.uni_agent.support import FakeTokenizer, RecordingLLMClient


def test_gateway_serving_runtime_rejects_zero_gateway_count():
    """``GatewayServingRuntime`` raises ``ValueError`` when ``gateway_count=0``
    and no external ``gateway_manager`` is supplied, rather than silently
    carrying a half-initialized runtime."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.runtime import GatewayServingRuntime

    with pytest.raises(ValueError, match="gateway_count must be positive"):
        GatewayServingRuntime(
            llm_client=object(),
            gateway_count=0,
            gateway_actor_config=GatewayActorConfig(tokenizer=object()),
        )


@pytest.fixture(scope="session")
def ray_runtime():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.mark.asyncio
async def test_gateway_serving_runtime_owns_gateway_lifecycle_and_session_runtime(ray_runtime):
    """Full lifecycle through the runtime: create, chat (via HTTP), complete
    (with reward_info), wait, finalize. Verifies the runtime→manager→actor
    chain works end-to-end and the trajectory carries the reward."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.runtime import GatewayServingRuntime

    llm_client = RecordingLLMClient("OWNER")
    runtime = GatewayServingRuntime(
        llm_client=llm_client,
        gateway_count=1,
        gateway_actor_config=GatewayActorConfig(tokenizer=FakeTokenizer()),
    )

    session = await runtime.create_session("session-owner")
    wait_task = runtime.wait_for_completion("session-owner", timeout=2.0)

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "content": "owner path"}]},
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "OWNER"

        complete = await client.post(
            f"{session.base_url.removesuffix('/v1')}/complete",
            json={"reward_info": {"score": 0.5, "label": "owner"}},
        )
        assert complete.status_code == 200

    await wait_task
    trajectories = await runtime.finalize_session("session-owner")
    await runtime.shutdown()

    assert len(trajectories) == 1
    assert trajectories[0].reward_info == {"score": 0.5, "label": "owner"}


@pytest.mark.asyncio
async def test_gateway_serving_runtime_round_robins_actors_across_alive_nodes(ray_runtime, monkeypatch):
    """gateway_count > 1 should distribute actors across alive CPU nodes round-robin."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.runtime import GatewayServingRuntime

    fake_nodes = [
        {"NodeID": "a" * 56, "Alive": True, "Resources": {"CPU": 8.0}},
        {"NodeID": "b" * 56, "Alive": True, "Resources": {"CPU": 8.0}},
        {"NodeID": "c" * 56, "Alive": False, "Resources": {"CPU": 8.0}},
        {"NodeID": "d" * 56, "Alive": True, "Resources": {"GPU": 1.0}},
    ]
    monkeypatch.setattr("uni_agent.gateway.runtime.ray.nodes", lambda: fake_nodes)

    captured_node_ids = []

    class _StubStartHandle:
        @staticmethod
        def remote():
            return ray.put(None)

    class _StubActorHandle:
        start = _StubStartHandle()

    class _RecordingActor:
        @classmethod
        def options(cls, *, scheduling_strategy):
            captured_node_ids.append(scheduling_strategy.node_id)
            return cls

        @classmethod
        def remote(cls, config, backend):
            return _StubActorHandle()

    monkeypatch.setattr("uni_agent.gateway.gateway.GatewayActor", _RecordingActor)

    runtime = GatewayServingRuntime(
        llm_client=object(),
        gateway_count=5,
        gateway_actor_config=GatewayActorConfig(tokenizer=object()),
    )

    assert captured_node_ids == ["a" * 56, "b" * 56, "a" * 56, "b" * 56, "a" * 56], captured_node_ids
    assert len(runtime.owned_gateway_actors) == 5
    # No shutdown call needed: stub actors have no real Ray state.
