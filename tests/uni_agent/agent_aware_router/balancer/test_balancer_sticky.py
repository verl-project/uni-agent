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

"""Sticky routing through the real strategy, callback collector, and store."""

from __future__ import annotations

import pytest

from uni_agent.agent_aware_router.types import MetricKey

from ._helpers import _make_balancer

pytestmark = [pytest.mark.level0, pytest.mark.cpu]


def _kv_metrics(per_replica: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for sid, m in per_replica.items():
        out[sid] = {
            MetricKey.KV_CACHE_USAGE_PERC: m.get("kv", 0.3),
            MetricKey.NUM_REQUESTS_RUNNING: m.get("running", 0),
            MetricKey.NUM_REQUESTS_WAITING: m.get("waiting", 0),
            MetricKey.INFLIGHT_COUNT: m.get("inflight", 0),
        }
    return out


class TestStickyEndToEnd:
    def _make_balancer(self, servers, metrics):
        balancer = _make_balancer(servers)
        balancer._strategy.set_capacity(64, 1024)
        balancer._store.refresh_metrics(metrics)
        return balancer

    def test_second_turn_same_request_stays_sticky(self):
        balancer = self._make_balancer(
            {"s0": "h0", "s1": "h1"},
            _kv_metrics({"s0": {}, "s1": {}}),
        )
        sid1, _ = balancer.acquire_server("r1", [1, 2])
        sid2, _ = balancer.acquire_server("r1", [1, 2])
        assert sid1 == sid2

    def test_overloaded_sticky_falls_back_to_healthy(self):
        balancer = self._make_balancer(
            {"s0": "h0", "s1": "h1"},
            _kv_metrics({"s0": {}, "s1": {}}),
        )
        balancer.acquire_server("r1", [1, 2])
        balancer._store.refresh_metrics(
            _kv_metrics({"s0": {"kv": 1.0, "running": 64, "waiting": 1000, "inflight": 64}, "s1": {"kv": 0.3}})
        )
        balancer._store.add_kv_blocks("s0", [f"b{i}" for i in range(10)])
        balancer._store.refresh_metrics({"s0": {MetricKey.NUM_GPU_BLOCKS: 10}})
        sid2, _ = balancer.acquire_server("r1", [1, 2])
        assert sid2 == "s1"

    def test_removed_sticky_server_reselects(self):
        balancer = self._make_balancer(
            {"s0": "h0", "s1": "h1", "s2": "h2"},
            _kv_metrics({"s0": {}, "s1": {}, "s2": {}}),
        )
        balancer.acquire_server("r1", [1, 2])
        balancer.remove_servers(["s0"])
        sid2, _ = balancer.acquire_server("r1", [1, 2])
        assert sid2 in {"s1", "s2"}
        assert balancer._store.get_sticky_binding("r1") in {"s1", "s2"}

    def test_get_status_reports_sticky_size(self):
        balancer = self._make_balancer(
            {"s0": "h0", "s1": "h1"},
            _kv_metrics({"s0": {}, "s1": {}}),
        )
        balancer.acquire_server("r1", [1])
        balancer.acquire_server("r2", [1])
        assert balancer.get_status()["sticky_size"] == 2
