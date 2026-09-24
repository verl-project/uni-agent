from __future__ import annotations

import asyncio

import httpx
import pytest
import ray

from tests.uni_agent.support import FakeTokenizer, SequencedBackend
from uni_agent.events import (
    GENERATION_FINISHED,
    SESSION_CLOSED,
    SESSION_OPENED,
    DirectAck,
    DirectAckStatus,
    DirectStateBatch,
    SnapshotAndCursor,
)

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@ray.remote
class _DirectAcceptingTarget:
    def install_direct_snapshot(self, data):
        request = SnapshotAndCursor.from_dict(data)
        return DirectAck(
            subscription_id=request.subscription_id,
            stream_epoch=request.stream_epoch,
            contiguous_cursor=request.snapshot.watermark,
            status=DirectAckStatus.OK,
        ).to_dict()

    def receive_direct_events(self, data):
        batch = DirectStateBatch.from_dict(data)
        return DirectAck(
            subscription_id=batch.subscription_id,
            stream_epoch=batch.stream_epoch,
            contiguous_cursor=batch.last_seq,
            status=DirectAckStatus.OK,
        ).to_dict()


@pytest.fixture(scope="module")
def global_ray_runtime():
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    yield
    ray.shutdown()


async def _wait_for_status(runtime, predicate, timeout_s: float = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        status = await runtime.status()
        if predicate(status):
            return status
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"Global telemetry did not converge: {status}")
        await asyncio.sleep(0.01)


async def _wait_for_gateway_status(gateway, method_name: str, predicate, timeout_s: float = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout_s
    method = getattr(gateway, method_name)
    while True:
        status = await method.remote()
        if predicate(status):
            return status
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"Gateway collector state did not converge: {status}")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_gateway_lifecycle_reaches_isolated_global_reporter(global_ray_runtime):
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager
    from uni_agent.telemetry import GlobalTelemetryRuntime, GlobalTelemetryRuntimeConfig

    runtime = GlobalTelemetryRuntime.start(GlobalTelemetryRuntimeConfig())
    manager = GatewayManager(
        llm_client=SequencedBackend(["OK"]),
        gateway_count=1,
        gateway_actor_config=GatewayActorConfig(
            tokenizer=FakeTokenizer(),
            global_telemetry_enabled=True,
            global_telemetry_flush_interval_s=0,
        ),
        global_telemetry_runtime=runtime,
    )

    try:
        session = await manager.create_session(
            "global-session",
            metadata={"_event_context": {"episode_id": "episode-global"}},
        )
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            response = await client.post(
                f"{session.base_url}/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == 200
        assert len(await manager.finalize_session("global-session")) == 1
        status = await _wait_for_status(
            runtime,
            lambda current: current["broker"]["accepted_events"] == 3
            and current["broker"]["subscribers"][0]["accepted_events"] == 3
            and current["reporter"]["event_counts"].get(SESSION_OPENED) == 1
            and current["reporter"]["event_counts"].get(GENERATION_FINISHED) == 1
            and current["reporter"]["event_counts"].get(SESSION_CLOSED) == 1,
        )

        assert status["reporter"]["namespace"] == "shadow"
        assert status["reporter"]["source_counts"] == {"gateway-0": 3}
        assert status["broker"]["accepted_events"] == 3
        assert status["broker"]["subscribers"][0]["accepted_events"] == 3
        source = await manager.gateways[0].get_global_telemetry_status.remote()
        assert source["status"] == "running"
        assert source["accepted_events"] == 3
        assert source["dropped_events"] == 0
        assert source["incomplete_batches"] == 0
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_direct_and_global_paths_keep_independent_progress(global_ray_runtime):
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager
    from uni_agent.telemetry import GlobalTelemetryRuntime, GlobalTelemetryRuntimeConfig

    runtime = GlobalTelemetryRuntime.start(GlobalTelemetryRuntimeConfig())
    direct_target = _DirectAcceptingTarget.remote()
    manager = GatewayManager(
        llm_client=SequencedBackend([]),
        gateway_count=1,
        gateway_actor_config=GatewayActorConfig(
            tokenizer=FakeTokenizer(),
            direct_state_sync_enabled=True,
            global_telemetry_enabled=True,
            global_telemetry_flush_interval_s=0,
        ),
        direct_event_target=direct_target,
        global_telemetry_runtime=runtime,
    )

    try:
        await manager.create_session("dual-path-session")
        assert await manager.finalize_session("dual-path-session") == []
        direct_status = await _wait_for_gateway_status(
            manager.gateways[0],
            "get_direct_sync_status",
            lambda status: status["contiguous_cursor"] == 2,
        )
        global_status = await _wait_for_status(
            runtime,
            lambda current: current["reporter"]["event_counts"].get(SESSION_OPENED) == 1
            and current["reporter"]["event_counts"].get(SESSION_CLOSED) == 1,
        )
        source_status = await manager.gateways[0].get_global_telemetry_status.remote()

        assert direct_status["status"] == "healthy"
        assert direct_status["pending_events"] == 0
        assert source_status["accepted_events"] == 2
        assert source_status["dropped_events"] == 0
        assert global_status["reporter"]["source_counts"] == {"gateway-0": 2}
    finally:
        await manager.shutdown()


def test_gateway_manager_requires_global_runtime_when_enabled(global_ray_runtime):
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager

    with pytest.raises(ValueError, match="global_telemetry_runtime"):
        GatewayManager(
            llm_client=SequencedBackend([]),
            gateway_count=1,
            gateway_actor_config=GatewayActorConfig(
                tokenizer=FakeTokenizer(),
                global_telemetry_enabled=True,
            ),
        )


@pytest.mark.parametrize("value", [None, "false", 0, 1])
def test_gateway_global_telemetry_flag_is_strict_boolean(value):
    from uni_agent.gateway.config import GatewayActorConfig

    with pytest.raises(ValueError, match="global_telemetry_enabled"):
        GatewayActorConfig(tokenizer=object(), global_telemetry_enabled=value)


def test_global_runtime_config_preserves_explicit_event_types():
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.telemetry import GlobalTelemetryRuntimeConfig

    config = GlobalTelemetryRuntimeConfig.from_mapping({"event_types": [SESSION_OPENED]})
    gateway_config = GatewayActorConfig(
        tokenizer=object(),
        global_telemetry_event_types=(SESSION_OPENED,),
    )

    assert config.event_types == (SESSION_OPENED,)
    assert gateway_config.global_telemetry_event_types == config.event_types

    with pytest.raises(ValueError, match="sequence of event names"):
        GatewayActorConfig(
            tokenizer=object(),
            global_telemetry_event_types=SESSION_OPENED,
        )
