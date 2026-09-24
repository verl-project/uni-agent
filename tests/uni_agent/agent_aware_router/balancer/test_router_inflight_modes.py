from __future__ import annotations

import pytest

from uni_agent.agent_aware_router.collectors.collector import Collector
from uni_agent.agent_aware_router.collectors.parse import MetricsUpdate, Parser
from uni_agent.agent_aware_router.collectors.transport.callback import CallbackTransport
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.store.per_replica_store import PerReplicaStore
from uni_agent.agent_aware_router.store.per_request_store import PerRequestStore
from uni_agent.agent_aware_router.types import MetricKey

from ._helpers import _make_balancer

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


def _metric(store, node_id: str, key: str) -> float:
    return float(store.get_metric(node_id, key) or 0)


def test_legacy_mode_keeps_collector_inflight_writer(monkeypatch):
    balancer = _build_balancer()
    _route_to("s0", monkeypatch)

    balancer.acquire_server("request-1", [1, 2, 3])

    assert balancer._router_inflight_projector is None
    assert balancer.get_router_state_status()["inflight"] == {"commit_owner": "legacy"}
    assert _metric(balancer._store, "s0", MetricKey.INFLIGHT_COUNT) == 1
    assert _metric(balancer._store, "s0", MetricKey.INFLIGHT_TOKENS) == 3
    assert _metric(balancer._store, "s0", MetricKey.DISPATCHED_COUNT) == 1
    assert _metric(balancer._store, "s0", MetricKey.PROMPT_LEN_SUM) == 3
    assert _metric(balancer._store, "s0", MetricKey.INFLIGHT_TURN_SUM) == 1

    balancer.release_server("s0", "request-1")

    assert _metric(balancer._store, "s0", MetricKey.INFLIGHT_COUNT) == 0
    assert _metric(balancer._store, "s0", MetricKey.INFLIGHT_TOKENS) == 0
    assert _metric(balancer._store, "s0", MetricKey.COMPLETED_COUNT) == 1
    assert _metric(balancer._store, "s0", MetricKey.INFLIGHT_TURN_SUM) == 0


def test_shadow_mode_reports_inflight_parity(monkeypatch):
    balancer = _build_balancer(router_state_mode="shadow")
    _route_to("s0", monkeypatch)

    server_id, _ = balancer.acquire_server("request-1", [1, 2])
    balancer.acquire_server("request-2", [5, 6, 7, 8, 9])
    balancer.release_server(server_id, "request-1")

    inflight = balancer.get_router_state_status()["inflight"]
    assert inflight["commit_owner"] == "legacy"
    assert inflight["healthy"] is True
    assert inflight["updates"] == 3
    assert inflight["parity_mismatches"] == 0
    assert inflight["tracked_replicas"] == 1
    assert inflight["replica_inflight"] == {"s0": 1}
    assert _metric(balancer._store, "s0", MetricKey.INFLIGHT_COUNT) == 1
    assert _metric(balancer._store, "s0", MetricKey.INFLIGHT_TURN_SUM) == 1
    assert _metric(balancer._store, "s0", MetricKey.INFLIGHT_TOKENS) == 5


def test_shadow_mode_counts_inflight_parity_mismatch(monkeypatch):
    balancer = _build_balancer(router_state_mode="shadow")
    _route_to("s0", monkeypatch)

    server_id, _ = balancer.acquire_server("request-1", [1])
    # External corruption of the family the shadow projector tracks.
    balancer._store.incr_metrics("s0", {MetricKey.INFLIGHT_COUNT: 5})
    balancer.release_server(server_id, "request-1")

    inflight = balancer.get_router_state_status()["inflight"]
    assert inflight["commit_owner"] == "legacy"
    assert inflight["parity_mismatches"] >= 1
    assert inflight["healthy"] is True


def test_shadow_inflight_projection_cannot_write_router_state():
    balancer = _build_balancer(router_state_mode="shadow")
    projector = balancer._router_inflight_projector

    projector.observe(
        MetricsUpdate(
            node_id="s0",
            metrics={MetricKey.INFLIGHT_COUNT: 1, MetricKey.DISPATCHED_COUNT: 1},
            is_delta=True,
            request_id="shadow-only",
        )
    )

    assert _metric(balancer._store, "s0", MetricKey.INFLIGHT_COUNT) == 0
    assert projector.status()["replica_inflight"] == {"s0": 1}
    assert projector.status()["parity_mismatches"] >= 1


def test_projector_mode_commits_inflight_delta_once(monkeypatch):
    balancer = _build_balancer(router_state_mode="projector")
    _route_to("s0", monkeypatch)
    original = balancer._store.incr_metrics
    writes = []

    def record_write(node_id, deltas):
        writes.append((node_id, dict(deltas)))
        return original(node_id, deltas)

    monkeypatch.setattr(balancer._store, "incr_metrics", record_write)

    server_id, _ = balancer.acquire_server("request-1", [1, 2, 3])
    balancer.release_server(server_id, "request-1")

    # Exactly one batched inflight-family write per acquire/release, through the
    # shared commit the projector owns — the Collector's legacy writer is gone.
    assert len(writes) == 2
    assert writes[0][0] == "s0"
    assert writes[0][1][MetricKey.INFLIGHT_COUNT] == 1
    assert writes[0][1][MetricKey.INFLIGHT_TURN_SUM] == 1
    assert writes[1][1][MetricKey.INFLIGHT_COUNT] == -1
    assert writes[1][1][MetricKey.INFLIGHT_TOKENS] == -3

    inflight = balancer.get_router_state_status()["inflight"]
    assert inflight["commit_owner"] == "projector"
    assert inflight["healthy"] is True
    assert inflight["parity_mismatches"] == 0
    assert _metric(balancer._store, "s0", MetricKey.INFLIGHT_COUNT) == 0
    assert _metric(balancer._store, "s0", MetricKey.COMPLETED_COUNT) == 1
    assert _metric(balancer._store, "s0", MetricKey.INFLIGHT_TURN_SUM) == 0
    assert _metric(balancer._store, "s0", MetricKey.PROMPT_LEN_SUM) == 3


def test_projector_inflight_failure_blocks_later_routes(monkeypatch):
    balancer = _build_balancer(router_state_mode="projector")
    _route_to("s0", monkeypatch)

    def fail_write(_node_id, _deltas):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(balancer._store, "incr_metrics", fail_write)

    with pytest.raises(RuntimeError, match="failed while committing"):
        balancer.acquire_server("request-1", [1])

    assert balancer.get_total_inflight() == 1
    inflight = balancer.get_router_state_status()["inflight"]
    assert inflight["healthy"] is False
    assert "RuntimeError" in inflight["last_error"]

    with pytest.raises(RuntimeError, match="Router inflight Projector is unhealthy"):
        balancer.acquire_server("request-2", [1])


def test_inflight_projector_rejects_non_delta_updates():
    from uni_agent.agent_aware_router.router_state import RouterStateMode

    balancer = _build_balancer(router_state_mode="shadow")
    projector = balancer._router_inflight_projector
    assert projector._mode is RouterStateMode.SHADOW

    with pytest.raises(ValueError, match="delta"):
        projector.apply(MetricsUpdate(node_id="s0", metrics={MetricKey.INFLIGHT_COUNT: 1}, is_delta=False))

    assert projector.healthy is False


class _StubBalancer:
    def register_call_back(self, event, fn):
        pass

    def un_register_call_back(self, event, fn):
        pass


class _FixedParser(Parser):
    def __init__(self, update):
        self._update = update

    def parse(self, raw_data, node_id):
        return self._update


def test_collector_routes_delta_updates_to_inflight_handler():
    delta = MetricsUpdate(
        node_id="s0",
        metrics={MetricKey.INFLIGHT_COUNT: -1, MetricKey.COMPLETED_COUNT: 1},
        is_delta=True,
        request_id="request-1",
    )
    seen = []
    collector = Collector(
        CallbackTransport(_StubBalancer()),
        _FixedParser(delta),
        inflight_update_handler=seen.append,
    )
    collector.start()
    try:
        # Drive the registered on_release packing closure: args → StatisticEvent →
        # handler → parser → MetricsUpdate → the inflight handler replaces the
        # Collector's own write, so the store must stay untouched.
        collector._transport._registered[1][1]("s0", "request-1")
    finally:
        collector.stop()

    assert [u.request_id for u in seen] == ["request-1"]
    assert _metric(collector._data_store, "s0", MetricKey.INFLIGHT_COUNT) == 0
    assert _metric(collector._data_store, "s0", MetricKey.COMPLETED_COUNT) == 0


def test_collector_keeps_absolute_updates_on_legacy_write_path():
    absolute = MetricsUpdate(node_id="s0", metrics={MetricKey.KV_CACHE_USAGE_PERC: 42.0}, is_delta=False)
    store_write = []
    collector = Collector(
        CallbackTransport(_StubBalancer()),
        _FixedParser(absolute),
        inflight_update_handler=lambda _update: store_write.append("handler"),
    )
    collector.start()
    try:
        collector._transport._registered[0][1]("request-1", "s0", [1])
    finally:
        collector.stop()

    assert store_write == []
    assert collector._data_store.get_metric("s0", MetricKey.KV_CACHE_USAGE_PERC) == 42.0
