import httpx
import pytest
import ray

from tests.uni_agent.support import FakeTokenizer, RecordingLLMClient


@pytest.fixture(scope="session")
def ray_runtime():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_gateway_manager_rejects_zero_gateway_count():
    """``GatewayManager`` raises ``ValueError`` when ``gateway_count=0``, rather
    than silently spawning a half-initialized manager."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager

    with pytest.raises(ValueError, match="gateway_count must be positive"):
        GatewayManager(
            llm_client=object(),
            gateway_count=0,
            gateway_actor_config=GatewayActorConfig(tokenizer=object()),
        )


@pytest.mark.asyncio
async def test_gateway_manager_round_robins_actors_across_alive_nodes(ray_runtime, monkeypatch):
    """gateway_count > 1 should distribute actors across alive CPU nodes round-robin."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager

    fake_nodes = [
        {"NodeID": "a" * 56, "Alive": True, "Resources": {"CPU": 8.0}},
        {"NodeID": "b" * 56, "Alive": True, "Resources": {"CPU": 8.0}},
        {"NodeID": "c" * 56, "Alive": False, "Resources": {"CPU": 8.0}},
        {"NodeID": "d" * 56, "Alive": True, "Resources": {"GPU": 1.0}},
    ]
    monkeypatch.setattr("uni_agent.gateway.manager.ray.nodes", lambda: fake_nodes)

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

    manager = GatewayManager(
        llm_client=object(),
        gateway_count=5,
        gateway_actor_config=GatewayActorConfig(tokenizer=object()),
    )

    assert captured_node_ids == ["a" * 56, "b" * 56, "a" * 56, "b" * 56, "a" * 56], captured_node_ids
    assert len(manager.gateways) == 5
    # No shutdown call needed: stub actors have no real Ray state.


@pytest.mark.asyncio
async def test_gateway_manager_finalizes_each_session_on_its_owning_gateway(ray_runtime):
    """create/finalize must route every session back to the same owning gateway
    across a multi-gateway pool. Two sessions land on different gateways; each is
    tagged with its own reward_info, and finalize must return that session's own
    trajectory -- a routing-table mix-up would surface as a swapped label.

    Asserts the manager's ownership routing (not the selection policy): any
    placement policy must still finalize a session on the gateway that created it.
    """
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager

    manager = GatewayManager(
        llm_client=RecordingLLMClient("OK"),
        gateway_count=2,
        gateway_actor_config=GatewayActorConfig(tokenizer=FakeTokenizer()),
    )

    session_a = await manager.create_session("session-a")
    session_b = await manager.create_session("session-b")
    # Least-loaded routing places the two sessions on distinct gateways, so a
    # mis-routed finalize would hit the other session's gateway.
    assert session_a.base_url != session_b.base_url

    async with httpx.AsyncClient(timeout=5.0) as client:
        for session, label in ((session_a, "a"), (session_b, "b")):
            chat = await client.post(
                f"{session.base_url}/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": f"hi {label}"}]},
            )
            assert chat.status_code == 200
            reward = await client.post(session.reward_info_url, json={"reward_info": {"label": label}})
            assert reward.status_code == 200

    trajectories_a = await manager.finalize_session("session-a")
    trajectories_b = await manager.finalize_session("session-b")

    assert [t.reward_info["label"] for t in trajectories_a] == ["a"]
    assert [t.reward_info["label"] for t in trajectories_b] == ["b"]

    await manager.shutdown()
