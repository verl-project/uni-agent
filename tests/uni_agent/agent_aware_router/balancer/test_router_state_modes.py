from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from uni_agent.agent_aware_router.balancer import KVCAwareBalancer
from uni_agent.agent_aware_router.collectors.parse import StickyUpdate
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.store.per_replica_store import PerReplicaStore
from uni_agent.agent_aware_router.store.per_request_store import PerRequestStore
from uni_agent.events import (
    GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
    GATEWAY_SESSION_STATE_SCOPE,
    REPLICA_CAPACITY_CHANGED,
    ROUTE_COMMITTED,
    SESSION_CLOSED,
    DeliveryMode,
    DirectAckStatus,
    DirectStateBatch,
    Event,
    EventContext,
    Scope,
    SnapshotAndCursor,
    StateSnapshot,
    SubscriptionSpec,
)

from ._helpers import _FakeCollectorManager, _make_balancer, _router_config

pytestmark = [pytest.mark.cpu, pytest.mark.level0]
_CREATED_BALANCERS = []


@pytest.fixture(autouse=True)
def reset_router_stores():
    PerReplicaStore._instance = None
    PerRequestStore._instance = None
    KVCacheStore._instance = None
    yield
    for balancer in _CREATED_BALANCERS:
        balancer._provider.stop()
    _CREATED_BALANCERS.clear()
    PerReplicaStore._instance = None
    PerRequestStore._instance = None
    KVCacheStore._instance = None


def _build_balancer(*, router_state_mode=None):
    balancer = _make_balancer(router_state_mode=router_state_mode)
    _CREATED_BALANCERS.append(balancer)
    return balancer


def _route_to(replica_id: str, monkeypatch) -> None:
    import uni_agent.agent_aware_router.balancer as balancer_module

    monkeypatch.setattr(balancer_module, "route", lambda *_args, **_kwargs: [replica_id])


def test_legacy_mode_allocates_no_router_projection_runtime():
    balancer = _build_balancer()

    assert balancer.get_router_state_status() == {
        "mode": "legacy",
        "enabled": False,
        "sticky": {"commit_owner": "legacy"},
        "inflight": {"commit_owner": "legacy"},
        "admission": {"enabled": False},
        "observation_failures": 0,
    }
    assert balancer._router_event_bus is None
    assert balancer._router_event_publisher is None
    assert balancer._router_sticky_projector is None
    assert balancer._router_inflight_projector is None
    assert balancer._capacity_versions == {}
    assert balancer._replica_epochs == {}


@pytest.mark.parametrize("mode", [True, "primary", "off"])
def test_router_state_mode_rejects_invalid_values(mode):
    with pytest.raises(ValueError, match="collectors.router_state.mode"):
        _build_balancer(router_state_mode=mode)


def test_shadow_mode_keeps_legacy_writer_and_reports_parity(monkeypatch):
    balancer = _build_balancer(router_state_mode="shadow")
    _route_to("s0", monkeypatch)

    balancer.acquire_server("request-1", [1, 2])

    status = balancer.get_router_state_status()
    assert balancer._store.get_sticky_binding("request-1") == "s0"
    assert balancer._router_sticky_projector.binding("request-1") == "s0"
    assert status["sticky"]["commit_owner"] == "legacy"
    assert status["sticky"]["parity_mismatches"] == 0

    assert balancer.clear_sticky_cache()["cleared_entries"] == 1
    assert balancer._router_sticky_projector.binding("request-1") is None


def test_shadow_projection_cannot_write_router_state():
    balancer = _build_balancer(router_state_mode="shadow")

    balancer._router_sticky_projector.observe(StickyUpdate(action="put", request_id="shadow-only", replica_id="s0"))

    assert balancer._router_sticky_projector.binding("shadow-only") == "s0"
    assert balancer._store.get_sticky_binding("shadow-only") is None
    assert balancer.get_router_state_status()["sticky"]["parity_mismatches"] == 1


def test_projector_mode_commits_sticky_update_once(monkeypatch):
    balancer = _build_balancer(router_state_mode="projector")
    _route_to("s0", monkeypatch)
    original = balancer._store.put_sticky_binding
    writes = []

    def record_write(request_id: str, replica_id: str) -> None:
        writes.append((request_id, replica_id))
        original(request_id, replica_id)

    monkeypatch.setattr(balancer._store, "put_sticky_binding", record_write)

    balancer.acquire_server("request-1", [1])

    assert writes == [("request-1", "s0")]
    assert balancer._store.get_sticky_binding("request-1") == "s0"
    assert balancer.get_router_state_status()["sticky"]["commit_owner"] == "projector"

    balancer.remove_servers(["s0"])
    assert balancer._store.get_sticky_binding("request-1") is None
    assert balancer._router_sticky_projector.binding("request-1") is None


def test_projector_failure_blocks_later_routes(monkeypatch):
    balancer = _build_balancer(router_state_mode="projector")
    _route_to("s0", monkeypatch)

    with pytest.raises(ValueError, match="unsupported sticky action"):
        balancer._router_sticky_projector.apply(StickyUpdate(action="invalid"))

    with pytest.raises(RuntimeError, match="Projector is unhealthy"):
        balancer.acquire_server("request-1", [1])
    assert balancer.get_total_inflight() == 0


def test_projector_commit_failure_keeps_route_unreturned_and_blocks_expansion(monkeypatch):
    balancer = _build_balancer(router_state_mode="projector")
    _route_to("s0", monkeypatch)

    def fail_write(_request_id: str, _replica_id: str) -> None:
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(balancer._store, "put_sticky_binding", fail_write)

    with pytest.raises(RuntimeError, match="failed while committing"):
        balancer.acquire_server("request-1", [1])

    assert balancer.get_total_inflight() == 1
    assert balancer.get_router_state_status()["admission"]["route_commits"] == 0
    with pytest.raises(RuntimeError, match="Projector is unhealthy"):
        balancer.acquire_server("request-2", [1])


def test_admission_observes_committed_route_and_capacity(monkeypatch):
    balancer = _build_balancer(router_state_mode="shadow")
    _route_to("s0", monkeypatch)

    server_id, _ = balancer.acquire_server("request-1", [1])
    balancer.release_server(server_id, "request-1")

    admission = balancer.get_router_state_status()["admission"]
    assert admission["authoritative"] is False
    assert admission["route_commits"] == 1
    assert admission["latest_route"] == {
        "request_id": "request-1",
        "replica_id": "s0",
        "ledger_version": 1,
    }
    assert admission["capacity"] == [
        {
            "replica_id": "s0",
            "replica_epoch": admission["capacity"][0]["replica_epoch"],
            "ledger_version": 2,
            "reason": "release",
            "health": "healthy",
            "replica_inflight": 0,
            "total_inflight": 0,
        }
    ]


def test_admission_observer_failure_does_not_block_committed_route(monkeypatch):
    balancer = _build_balancer(router_state_mode="shadow")
    _route_to("s0", monkeypatch)

    def fail_observer(_event: Event) -> None:
        raise RuntimeError("observer unavailable")

    balancer._router_event_bus.subscribe(
        SubscriptionSpec(
            subscription_id="failing-admission-observer",
            event_types=(ROUTE_COMMITTED, REPLICA_CAPACITY_CHANGED),
            scope=Scope.LOCAL,
            delivery=DeliveryMode.INLINE,
        ),
        fail_observer,
    )

    assert balancer.acquire_server("request-1", [1])[0] == "s0"
    status = balancer.get_router_state_status()
    assert status["admission"]["route_commits"] == 1
    assert status["observation_failures"] == 2


def test_admission_observes_direct_gateway_lifecycle():
    balancer = _build_balancer(router_state_mode="shadow")
    snapshot = SnapshotAndCursor(
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        stream_epoch="stream-1",
        snapshot=StateSnapshot(
            owner="gateway-0",
            source_epoch="source-1",
            scope=GATEWAY_SESSION_STATE_SCOPE,
            revision=1,
            watermark=0,
            source_health="healthy",
            current_entities={"session-1": {"episode_id": "episode-1", "revision": 1, "status": "open"}},
            terminal_entities={},
        ),
    )
    assert balancer.install_direct_snapshot(snapshot.to_dict())["status"] == DirectAckStatus.OK.value

    event = Event(
        event_id="close-1",
        event_type=SESSION_CLOSED,
        schema_version=1,
        run_id="run-1",
        producer_id="gateway-0",
        producer_epoch="source-1",
        producer_seq=1,
        context=EventContext(episode_id="episode-1", session_id="session-1"),
        occurred_at_unix_ns=1,
        payload={"revision": 2, "source_revision": 2, "close_reason": "finalized"},
    )
    batch = DirectStateBatch(
        run_id="run-1",
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        source_epoch="source-1",
        stream_epoch="stream-1",
        first_seq=1,
        last_seq=1,
        events=(event,),
    )
    assert balancer.receive_direct_events(batch.to_dict())["status"] == DirectAckStatus.OK.value

    admission = balancer.get_router_state_status()["admission"]
    assert admission["active_sessions"] == 0
    assert admission["terminal_sessions"] == 1
    assert admission["source_revisions"] == {"gateway-0": 2}
    assert admission["source_health"] == {"gateway-0": "healthy"}


class _RemoteMethod:
    def __init__(self, value):
        self._value = value

    def remote(self):
        return self._value


class _RolloutServer:
    def __init__(self, mode):
        self.get_rollout_config = _RemoteMethod(
            OmegaConf.create(
                {
                    "custom": {
                        "agent_framework": {
                            "collectors": {"router_state": {"mode": mode}},
                        }
                    },
                    "max_num_seqs": 64,
                    "max_num_batched_tokens": 1024,
                }
            )
        )
        self.get_server_address = _RemoteMethod(("127.0.0.1", 8000))
        self.get_kv_events_endpoints = _RemoteMethod(None)


def test_router_state_mode_is_read_from_formal_collectors_config(monkeypatch):
    import uni_agent.agent_aware_router.balancer as balancer_module

    monkeypatch.setattr(balancer_module.ray, "get", lambda value: value)
    balancer = KVCAwareBalancer(
        {"s0": _RolloutServer("shadow")},
        _router_config(),
        provider_factory=_FakeCollectorManager,
    )
    _CREATED_BALANCERS.append(balancer)

    assert balancer.get_router_state_status()["mode"] == "shadow"
