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

"""Inflight accounting through the real callback collector and store."""

from __future__ import annotations

import pytest

from uni_agent.agent_aware_router.types import MetricKey

from ._helpers import _make_balancer

pytestmark = [pytest.mark.level0, pytest.mark.cpu]


def test_acquire_release_updates_gauge_and_counters():
    balancer = _make_balancer({"s0": "h0"})

    assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_COUNT) == 0
    server_id, _ = balancer.acquire_server("request-1", [1, 2, 3])
    assert balancer._store.get_metric(server_id, MetricKey.INFLIGHT_COUNT) == 1
    assert balancer._store.get_metric(server_id, MetricKey.DISPATCHED_COUNT) == 1

    balancer.release_server(server_id, request_id="request-1")
    assert balancer._store.get_metric(server_id, MetricKey.INFLIGHT_COUNT) == 0
    assert balancer._store.get_metric(server_id, MetricKey.COMPLETED_COUNT) == 1


def test_turn_values_are_attributed_to_the_receiving_replica():
    balancer = _make_balancer({"s0": "h0"})

    for _ in range(3):
        balancer.acquire_server("request-1", [1])
    balancer.acquire_server("request-2", [1])

    assert balancer._store.get_per_request("request-1", "turn", 0) == 3
    assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_TURN_SUM) == 7
    assert balancer._store.get_metric("s0", MetricKey.DISPATCHED_COUNT) == 4
