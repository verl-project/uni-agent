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

"""Balancer conformance with the verl #7115 router protocol."""

from __future__ import annotations

import pytest

from uni_agent.agent_aware_router.types import MetricKey

from ._helpers import _make_balancer

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_declares_forwarded_fields():
    balancer = _make_balancer()

    assert balancer.require_acquire_fields() == ["prompt_ids"]
    assert balancer.require_release_fields() == ["request_id"]


def test_release_balances_latest_prompt_tokens():
    balancer = _make_balancer({"s0": "h0"})

    for prompt_ids in ([1, 2, 3], [1, 2, 3, 4, 5, 6]):
        server_id, _ = balancer.acquire_server("request-1", prompt_ids)
        balancer.release_server(server_id, request_id="request-1")
        assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_COUNT) == 0
        assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 0


def test_release_without_request_id_cannot_balance_prompt_tokens():
    balancer = _make_balancer({"s0": "h0"})
    server_id, _ = balancer.acquire_server("request-1", [1, 2, 3])

    balancer.release_server(server_id)

    assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_COUNT) == 0
    assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 3


def test_clear_sticky_cache_preserves_prefix_checkpoint_and_reports_load(layout):
    balancer = _make_balancer({"s0": "h0"})
    balancer._store.configure_cache_registry(layout((("full_attention", 16),), unit=None))
    balancer.acquire_server("request-1", list(range(32)))

    result = balancer.clear_sticky_cache()

    assert result == {"cleared_entries": 1, "server_loads": {"s0": 1}}
    assert balancer._store.get_sticky_binding("request-1") is None
    assert balancer._store.get_per_request("request-1", "prefix_hashes") is not None


def test_inflight_counter_tolerates_unknown_and_late_releases():
    balancer = _make_balancer({"s0": "h0"})
    balancer.acquire_server("request-1", [1])
    balancer.acquire_server("request-2", [2])

    balancer.release_server("unknown", request_id="request-1")
    assert balancer.get_status()["total_inflight"] == 2

    balancer.remove_servers(["s0"])
    assert balancer.get_total_inflight() == 2
    balancer.release_server("s0", request_id="request-1")
    assert balancer.get_total_inflight() == 1
    balancer.release_server("s0", request_id="request-2")
    assert balancer.get_total_inflight() == 0
    assert "s0" not in balancer._inflight
    balancer.release_server("s0", request_id="request-2")
    assert balancer.get_total_inflight() == 0


def test_readding_server_preserves_outstanding_requests():
    balancer = _make_balancer({"s0": "h0"})
    balancer.acquire_server("old-request")
    balancer.remove_servers(["s0"])
    balancer.add_servers({"s0": "replacement"})
    balancer.acquire_server("new-request")
    assert balancer.get_total_inflight() == 2
    balancer.release_server("s0", "old-request")
    assert balancer.get_total_inflight() == 1
    balancer.release_server("s0", "new-request")
    assert balancer.get_total_inflight() == 0
