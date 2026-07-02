"""Helpers for balancer unit tests.

Defines ``_FakeProvider`` and patch helpers.  The provider patch
(``_collectors_mod.RouteDataProvider = _FakeProvider``) is applied via a
session-scoped autouse fixture — NOT at import time — so it only takes
effect while balancer unit tests are running and never leaks to Ray
workers in other test modules.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

# ── Fake provider (no module-level patch) ───────────────────────────────


class _FakeProvider:
    """Stand-in for RouteDataProvider — no real collectors run."""

    def __init__(self, collectors_config, collection_names, server_addresses=None, kv_event_endpoints=None):
        self.collectors_config = collectors_config
        self.collection_names = collection_names
        self.server_addresses = server_addresses
        self.kv_event_endpoints = kv_event_endpoints
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def get_metric(self, replica_id, key):
        return 0.0

    def get_metrics(self, replica_id):
        return {}

    def get_gpu_prefix_hit_rate(self, prompt_ids):
        return {}

    def get_tier_prefix_hit_rate(self, replica_id, prompt_ids, tier):
        return 0.0


# ── Helpers used by test classes ─────────────────────────────────────────


def _router_config(weight: float = 1.0):
    """Build a minimal router_config (OmegaConf) the Balancer accepts."""
    return OmegaConf.create(
        {
            "router_class": "uni_agent.llm_router.balancer.KVCAwareBalancer",
            "strategies": [
                {
                    "_target_": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
                    "weight": weight,
                    "collector_names": ["vllm_zmq"],
                },
            ],
        }
    )


def _fake_init_provider(self):
    """Replacement for KVCAwareBalancer._init_provider in unit tests."""
    collection_names = sorted({name for cfg in self._config.strategies for name in cfg.collector_names})
    self._provider = _FakeProvider(
        self._config.collector,
        collection_names,
    )
    self._provider.start()


def _make_balancer(servers=None):
    """Build a balancer over the given servers (default two)."""
    from uni_agent.llm_router.balancer import KVCAwareBalancer

    if servers is None:
        servers = {"s0": "h0", "s1": "h1"}
    return KVCAwareBalancer(servers, _router_config())


# ── Patch fixture (session-scoped, only for ut balancer tests) ──────────
# Replaces the old module-level patch + restore pattern.  The fixture is
# autouse so every test in this directory gets the patch, and it is properly
# torn down so Ray workers in other test directories never see _FakeProvider.


@pytest.fixture(autouse=True, scope="session")
def _patch_provider():
    """Patch RouteDataProvider + _init_provider for the entire balancer/ut session.

    Applied AFTER import (via fixture), so no import-time side effects.
    Restored on teardown so st-cpu/e2e Ray workers get the real provider.
    """
    import uni_agent.llm_router.collectors as _collectors_mod
    from uni_agent.llm_router.balancer import KVCAwareBalancer

    # Save originals
    _orig_provider = _collectors_mod.RouteDataProvider
    _orig_init = KVCAwareBalancer._init_provider

    # Patch RouteDataProvider so balancer.py (which imports it at top level)
    # gets _FakeProvider when it's first imported by this test session.
    # But balancer.py was already imported by other tests (strategies/route).
    # The patch only affects KVCAwareBalancer._init_provider's construction
    # path, which calls RouteDataProvider(...) — so patching the class
    # attribute on the module is sufficient even if balancer.py is already
    # loaded, because it looks up RouteDataProvider from the module at
    # call time (not import time).
    _collectors_mod.RouteDataProvider = _FakeProvider
    KVCAwareBalancer._init_provider = _fake_init_provider

    yield

    # Restore — Ray workers forked AFTER this will get the real provider
    _collectors_mod.RouteDataProvider = _orig_provider
    KVCAwareBalancer._init_provider = _orig_init
