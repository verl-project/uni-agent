from __future__ import annotations

import asyncio

import pytest
import ray

from tests.uni_agent.support import FakeTokenizer, SequencedBackend
from uni_agent.agent_aware_router.session_state import GatewaySessionStateProjector
from uni_agent.events import (
    GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
    SESSION_CLOSED,
    SESSION_OPENED,
    DeliveryMode,
    DirectEventEndpoint,
    DirectStateBatch,
    LocalEventBus,
    Scope,
    SnapshotAndCursor,
    SubscriptionSpec,
)

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@ray.remote
class _DirectShadowTarget:
    def __init__(self) -> None:
        self._bus = LocalEventBus()
        self._projector = GatewaySessionStateProjector()
        self._bus.subscribe(
            SubscriptionSpec(
                GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
                event_types=(SESSION_OPENED, SESSION_CLOSED),
                scope=Scope.DIRECT,
                delivery=DeliveryMode.STATE_SYNC,
            ),
            self._projector.apply,
        )
        self._endpoint = DirectEventEndpoint(self._bus, snapshot_installer=self._projector.install_snapshot)

    def install_direct_snapshot(self, request):
        return self._endpoint.install_snapshot(SnapshotAndCursor.from_dict(request)).to_dict()

    def receive_direct_events(self, batch):
        return self._endpoint.receive_events(DirectStateBatch.from_dict(batch)).to_dict()

    def status(self):
        return {**self._projector.status(), "streams": list(self._endpoint.status())}


@pytest.fixture(scope="module")
def direct_ray_runtime():
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    yield
    ray.shutdown()


async def _wait_for_status(target, predicate, timeout_s: float = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        status = await target.status.remote()
        if predicate(status):
            return status
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"Direct shadow state did not converge: {status}")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_gateway_manager_streams_session_lifecycle_to_ray_shadow_target(direct_ray_runtime):
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager

    target = _DirectShadowTarget.remote()
    manager = GatewayManager(
        llm_client=SequencedBackend([]),
        gateway_count=1,
        gateway_actor_config=GatewayActorConfig(
            tokenizer=FakeTokenizer(),
            direct_state_sync_enabled=True,
        ),
        direct_event_target=target,
    )

    try:
        await manager.create_session(
            "direct-session",
            metadata={"_event_context": {"episode_id": "episode-direct"}},
        )
        opened = await _wait_for_status(target, lambda status: len(status["current"]) == 1)
        assert opened["current"][0]["session_id"] == "direct-session"
        assert opened["current"][0]["episode_id"] == "episode-direct"

        assert await manager.finalize_session("direct-session") == []
        closed = await _wait_for_status(target, lambda status: len(status["terminal"]) == 1)
        assert closed["current"] == []
        assert closed["terminal"][0]["status"] == "closed"
        assert closed["streams"][0]["cursor"] == 2

        source_status = await manager.gateways[0].get_direct_sync_status.remote()
        assert source_status["status"] == "healthy"
        assert source_status["contiguous_cursor"] == 2
        assert source_status["pending_events"] == 0
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_two_gateways_keep_independent_direct_streams(direct_ray_runtime):
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager

    target = _DirectShadowTarget.remote()
    manager = GatewayManager(
        llm_client=SequencedBackend([]),
        gateway_count=2,
        gateway_actor_config=GatewayActorConfig(
            tokenizer=FakeTokenizer(),
            direct_state_sync_enabled=True,
        ),
        direct_event_target=target,
    )

    try:
        await manager.create_session("direct-a")
        await manager.create_session("direct-b")
        opened = await _wait_for_status(
            target,
            lambda status: len(status["current"]) == 2 and len(status["streams"]) == 2,
        )
        assert {item["owner"] for item in opened["current"]} == {"gateway-0", "gateway-1"}

        await manager.finalize_session("direct-a")
        await manager.finalize_session("direct-b")
        closed = await _wait_for_status(target, lambda status: len(status["terminal"]) == 2)
        assert closed["current"] == []
        assert {stream["cursor"] for stream in closed["streams"]} == {2}
    finally:
        await manager.shutdown()


def test_gateway_manager_requires_a_direct_target_when_enabled(direct_ray_runtime):
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager

    with pytest.raises(ValueError, match="direct_event_target"):
        GatewayManager(
            llm_client=SequencedBackend([]),
            gateway_count=1,
            gateway_actor_config=GatewayActorConfig(
                tokenizer=FakeTokenizer(),
                direct_state_sync_enabled=True,
            ),
        )
