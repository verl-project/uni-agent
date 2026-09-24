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

"""Balancer test helpers."""

from __future__ import annotations

from omegaconf import OmegaConf


class _FakeCollectorManager:
    """Run real callback collectors without starting network collectors."""

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
    return OmegaConf.create(
        {
            "strategy": {
                "_target_": "uni_agent.agent_aware_router.config.strategy.KVCAwareStrategyConfig",
            },
        }
    )


def _make_balancer(servers=None, max_num_seqs=None, router_state_mode=None):
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
    balancer = KVCAwareBalancer(
        servers,
        _router_config(),
        provider_factory=_FakeCollectorManager,
        router_state_mode=router_state_mode,
    )
    if max_num_seqs is not None:
        if hasattr(balancer._strategy, "set_capacity"):
            balancer._strategy.set_capacity(max_num_seqs, 2048)
    return balancer
