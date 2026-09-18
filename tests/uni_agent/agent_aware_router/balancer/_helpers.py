# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for balancer unit tests.

Defines ``_FakeCollectorManager`` (real statistic collectors + stubbed network
collectors), injected into the Balancer via its ``provider_factory`` seam —
no class-attribute monkey-patching (ray.remote serializes the class by value,
so a patched class leaks into Ray actors sharing the session).
"""

from __future__ import annotations

from omegaconf import OmegaConf


class _FakeCollectorManager:
    """Stand-in for ``CollectorManager`` — stubs NETWORK collectors but builds
    REAL statistic collectors (anything with ``is_async=False``).

    The Balancer-callback → CallbackTransport → StickyParser/InflightParser →
    DataStore chain is the heart of the Phase-1 refactor, so it must run
    end-to-end (NOT mocked). Network collectors (vllm_metrics / vllm_zmq) need
    live endpoints and are stubbed; tests inject metrics directly via
    ``balancer._store.refresh_metrics(...)``.
    """

    def __init__(
        self, collectors_config, collection_names, server_addresses=None, kv_event_endpoints=None, balancer_handler=None
    ):
        self.collectors_config = collectors_config
        self.collection_names = collection_names
        self.server_addresses = server_addresses
        self.kv_event_endpoints = kv_event_endpoints
        self.balancer_handler = balancer_handler
        self.started = False
        self.stopped = False
        # Build every collector through the real factory; keep only the statistic
        # ones (is_async=False). Network collectors are built but not started.
        from uni_agent.agent_aware_router.collectors.collector import get_collector

        self._statistic_collectors = [
            c
            for c in (
                get_collector(name, collectors_config, balancer_handler=balancer_handler) for name in collection_names
            )
            if not getattr(c._transport, "is_async", True)
        ]

    def start(self):
        self.started = True
        for c in self._statistic_collectors:
            c.start()

    def stop(self):
        self.stopped = True
        for c in self._statistic_collectors:
            c.stop()


def _router_config():
    """Build a minimal router_config (OmegaConf) the Balancer accepts."""
    return OmegaConf.create(
        {
            "strategy": {
                "_target_": "uni_agent.agent_aware_router.config.strategy.KVCAwareStrategyConfig",
            },
        }
    )


def _make_balancer(servers=None, max_num_seqs=None):
    """Build a balancer over the given servers (default two).

    ``_FakeCollectorManager`` is injected through the Balancer's
    ``provider_factory`` seam — real statistic collectors run (the
    Balancer-callback chain), network collectors are stubbed, and the real
    singleton-backed ``DataStore`` is shared with strategy reads.

    ``max_num_seqs`` overrides the capacity the Balancer resolved at construction
    (tests pass plain-string servers with no ``get_rollout_config``, so the
    Balancer's RPC resolution falls back to its default). Applied the same way
    the Balancer applies it in ``__init__``: via ``strategy.set_capacity(...)``.
    """
    from uni_agent.agent_aware_router.balancer import KVCAwareBalancer

    if servers is None:
        servers = {"s0": "h0", "s1": "h1"}
    balancer = KVCAwareBalancer(servers, _router_config(), provider_factory=_FakeCollectorManager)
    if max_num_seqs is not None:
        if hasattr(balancer._strategy, "set_capacity"):
            balancer._strategy.set_capacity(max_num_seqs)
    return balancer
