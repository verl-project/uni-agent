from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from uni_agent.llm_router.balancer import KVCAwareBalancer

from ._helpers import (
    _router_config,
)

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


# ============================================================
# Sticky-session end-to-end (real strategy, no route monkeypatch)
# ============================================================


class _MetricsProvider:
    """Metrics-aware fake provider for sticky e2e tests.

    Configured per-replica metrics; returns real KV/load numbers so the real
    KVCacheAwareStrategy can compute s_load and decide overload + stickiness.
    get_gpu_prefix_hit_rate / get_tier_prefix_hit_rate return empty → combined
    scoring degrades to load-only (no cache term), which is fine for sticky
    behavior: the deciding factor is whether the bound replica is overloaded.
    """

    def __init__(self, metrics: dict[str, dict] | None = None):
        self._metrics = metrics or {}

    def start(self):
        pass

    def stop(self):
        pass

    def get_metrics(self, replica_id):
        return dict(self._metrics.get(replica_id, {}))

    def get_metric(self, replica_id, key):
        return self.get_metrics(replica_id).get(key, 0.0)

    def get_gpu_prefix_hit_rate(self, prompt_ids):
        return {}

    def get_tier_prefix_hit_rate(self, replica_id, prompt_ids, tier):
        return 0.0


def _kv_metrics(per_replica: dict[str, dict]) -> dict[str, dict]:
    """Normalize {sid: {kv, running, waiting}} into MetricKey-keyed dicts.

    Defaults: kv=0.3 (→ load=0.12, NOT overloaded under load_threshold 0.9),
    running=0, waiting=0.
    """
    from uni_agent.llm_router.metric_spec import MetricKey

    out = {}
    for sid, m in per_replica.items():
        out[sid] = {
            MetricKey.KV_CACHE_USAGE_PERC: m.get("kv", 0.3),
            MetricKey.NUM_REQUESTS_RUNNING: m.get("running", 0),
            MetricKey.NUM_REQUESTS_WAITING: m.get("waiting", 0),
        }
    return out


class TestStickyEndToEnd:
    """Real KVCacheAwareStrategy + real route(): sticky affinity across turns."""

    def _make_balancer_with_metrics(self, servers, metrics):
        """Build a balancer whose provider returns the given per-replica metrics."""

        provider = _MetricsProvider(metrics)

        def fake_init(self):
            self._provider = provider

        orig = KVCAwareBalancer._init_provider
        KVCAwareBalancer._init_provider = fake_init
        try:
            return KVCAwareBalancer(servers, _router_config())
        finally:
            KVCAwareBalancer._init_provider = orig

    def test_second_turn_same_request_stays_sticky(self):
        """Feature: bound + not overloaded → second turn routes to same server.
        Description: turn1 acquires (cold start, picks some server); turn2 same
        request_id with that server still healthy must return the SAME server.
        Expectation: turn1.sid == turn2.sid (sticky hit)
        """
        # s0/s1 both healthy (kv=0.3 → load=0.12, not overloaded)
        balancer = self._make_balancer_with_metrics(
            {"s0": "h0", "s1": "h1"},
            _kv_metrics({"s0": {}, "s1": {}}),
        )
        sid1, _ = balancer.acquire_server("r1", [1, 2])
        sid2, _ = balancer.acquire_server("r1", [1, 2])
        assert sid1 == sid2

    def test_overloaded_sticky_falls_back_to_healthy(self, monkeypatch):
        """Feature: bound replica becomes saturated (load>0.9) → rebind to a healthy one.
        Description: turn1 binds r1→s0; then s0 saturated (kv=1,r=64,w=1000 → load≈0.98),
        s1 healthy (kv=0.3 → load=0.12).
        Expectation: turn2 routes to s1 (not the saturated s0), and rebinds r1→s1
        """
        monkeypatch.setenv("MAX_NUM_SEQS", "64")  # pin the load formula's running term
        # turn1: both healthy, cold start
        balancer = self._make_balancer_with_metrics(
            {"s0": "h0", "s1": "h1"},
            _kv_metrics({"s0": {}, "s1": {}}),
        )
        sid1, _ = balancer.acquire_server("r1", [1, 2])
        # mutate metrics: s0 now saturated (load>0.9), s1 healthy
        balancer._provider._metrics = _kv_metrics(
            {
                "s0": {"kv": 1.0, "running": 64, "waiting": 1000},
                "s1": {"kv": 0.3},
            }
        )
        sid2, _ = balancer.acquire_server("r1", [1, 2])
        assert sid2 == "s1", f"expected fallback to s1, got {sid2} (turn1 was {sid1})"

    def test_removed_sticky_server_reselects(self):
        """Feature: bound server removed → reselect from remaining pool.
        Description: turn1 binds r1→s0; remove s0; turn2 must pick from {s1,s2}.
        Expectation: turn2 sid in {s1,s2}, no crash, sticky rebound
        """
        balancer = self._make_balancer_with_metrics(
            {"s0": "h0", "s1": "h1", "s2": "h2"},
            _kv_metrics({"s0": {}, "s1": {}, "s2": {}}),
        )
        sid1, _ = balancer.acquire_server("r1", [1, 2])
        balancer.remove_servers(["s0"])
        sid2, _ = balancer.acquire_server("r1", [1, 2])
        assert sid2 in {"s1", "s2"}
        # sticky binding to s0 should have been invalidated
        assert balancer._sticky.get("r1") in {"s1", "s2"}

    def test_get_status_reports_sticky_size(self):
        """Feature: get_status() includes sticky_size.
        Description: acquire two distinct request_ids; check status
        Expectation: sticky_size == 2
        """
        balancer = self._make_balancer_with_metrics(
            {"s0": "h0", "s1": "h1"},
            _kv_metrics({"s0": {}, "s1": {}}),
        )
        balancer.acquire_server("r1", [1])
        balancer.acquire_server("r2", [1])
        status = balancer.get_status()
        assert status["sticky_size"] == 2

    def test_sticky_respects_configured_max_size(self):
        """Feature: KVCAwareConfig.sticky_max_size flows to the sticky table.
        Description: build a balancer with sticky_max_size overridden in config
        Expectation: balancer._sticky.max_size == overridden value
        """
        cfg = OmegaConf.create(
            {
                "router_class": "uni_agent.llm_router.balancer.KVCAwareBalancer",
                "sticky_max_size": 42,
                "strategies": [
                    {
                        "_target_": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    },
                ],
            }
        )
        orig = KVCAwareBalancer._init_provider
        KVCAwareBalancer._init_provider = lambda self: setattr(
            self,
            "_provider",
            _MetricsProvider(_kv_metrics({"s0": {}})),
        )
        try:
            balancer = KVCAwareBalancer({"s0": "h0"}, cfg)
            assert balancer._sticky.max_size == 42
        finally:
            KVCAwareBalancer._init_provider = orig
