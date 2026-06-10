import httpx
import pytest
import ray

from tests.uni_agent.support import FakeTokenizer, QueuedBackend, TrackingGatewayActor


@pytest.fixture(scope="session")
def ray_runtime():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.mark.asyncio
async def test_gateway_manager_routes_sessions_stickily(ray_runtime):
    """Each session is pinned to a single gateway actor (sticky routing).
    Two consecutive requests for the same session_id land on the same
    gateway, producing one continuous trajectory."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import GatewayActor
    from uni_agent.gateway.manager import GatewayManager

    gateways = [
        GatewayActor.remote(GatewayActorConfig(tokenizer=FakeTokenizer()), QueuedBackend(["A"])),
        GatewayActor.remote(GatewayActorConfig(tokenizer=FakeTokenizer()), QueuedBackend(["B"])),
    ]
    ray.get([gateway.start.remote() for gateway in gateways])

    manager = GatewayManager(gateways)
    session_a = await manager.create_session("session-a")
    session_b = await manager.create_session("session-b")

    assert manager.gateway_count == 2
    assert session_a.base_url != session_b.base_url

    async with httpx.AsyncClient(timeout=5.0) as client:
        first = await client.post(
            f"{session_a.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "content": "route a"}]},
        )
        second = await client.post(
            f"{session_b.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "content": "route b"}]},
        )
        assert first.status_code == 200
        assert second.status_code == 200

    trajectories_a = await manager.finalize_session("session-a")
    trajectories_b = await manager.finalize_session("session-b")

    assert len(trajectories_a) == 1
    assert len(trajectories_b) == 1

    ray.get([gateway.shutdown.remote() for gateway in gateways])


@pytest.mark.asyncio
async def test_gateway_manager_uses_least_active_sessions_routing(ray_runtime):
    """New sessions are routed to the gateway actor with the fewest active
    sessions (least-loaded). When a session is finalized the counter
    decrements, making that gateway available for the next create."""
    from uni_agent.gateway.manager import GatewayManager

    gateways = [
        TrackingGatewayActor.remote("gw-0"),
        TrackingGatewayActor.remote("gw-1"),
    ]
    ray.get([gateway.start.remote() for gateway in gateways])

    manager = GatewayManager(gateways)
    session_a = await manager.create_session("session-a")
    session_b = await manager.create_session("session-b")
    session_c = await manager.create_session("session-c")

    assert manager.active_sessions_per_gateway == [2, 1]
    assert session_a.base_url.startswith("http://gw-0/")
    assert session_b.base_url.startswith("http://gw-1/")
    assert session_c.base_url.startswith("http://gw-0/")

    await manager.finalize_session("session-a")
    assert manager.active_sessions_per_gateway == [1, 1]

    session_d = await manager.create_session("session-d")
    assert session_d.base_url.startswith("http://gw-0/")
    assert manager.active_sessions_per_gateway == [2, 1]

    ray.get([gateway.shutdown.remote() for gateway in gateways])
