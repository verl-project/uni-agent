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
    # Re-acquires replace earlier active turns: request-1 turn 3 + request-2 turn 1.
    assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_TURN_SUM) == 4
    assert balancer._store.get_metric("s0", MetricKey.DISPATCHED_COUNT) == 4
    balancer.release_server("s0", request_id="request-1")
    balancer.release_server("s0", request_id="request-2")
    assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_TURN_SUM) == 0
    assert balancer.get_total_inflight() == 0
    assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_COUNT) == 0
    assert balancer._store.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 0


def _assert_active(balancer, expected):
    """Check the drain count and collector gauges together, including empty replicas."""
    assert balancer.get_total_inflight() == sum(count for count, _, _ in expected.values())
    assert balancer.get_status()["total_inflight"] == balancer.get_total_inflight()
    for replica, (count, tokens, turns) in expected.items():
        assert balancer._inflight.get(replica, 0) == count
        for key, value in zip(
            (MetricKey.INFLIGHT_COUNT, MetricKey.INFLIGHT_TOKENS, MetricKey.INFLIGHT_TURN_SUM),
            (count, tokens, turns),
            strict=True,
        ):
            assert balancer._store.get_metric(replica, key) == value


@pytest.mark.parametrize("request_id", ["request-1", ""])
@pytest.mark.parametrize("targets", [("s0", "s0", "s0"), ("s0", "s1", "s0")])
def test_replacements_and_releases_keep_drain_and_store_counts_in_sync(monkeypatch, request_id, targets):
    from uni_agent.agent_aware_router import balancer as balancer_module

    choices = iter((*targets, "s1"))
    monkeypatch.setattr(balancer_module, "route", lambda *_args: [next(choices)])
    balancer = _make_balancer({"s0": "h0", "s1": "h1"})
    balancer._store.set_block_size(2)
    for turn, replica in enumerate(targets, 1):
        balancer.acquire_server(request_id, list(range(turn * 2)))
        expected = {"s0": (0, 0, 0), "s1": (0, 0, 0)}
        expected[replica] = (1, turn * 2, turn)
        _assert_active(balancer, expected)
    balancer.acquire_server("other", [1, 2])
    _assert_active(balancer, {"s0": (1, 6, 3), "s1": (1, 2, 1)})

    # A late release on the old/wrong replica must not debit its unrelated request.
    balancer.release_server("s1", request_id)
    balancer.release_server("s1", "unknown")
    _assert_active(balancer, {"s0": (1, 6, 3), "s1": (1, 2, 1)})
    balancer.release_server("s0", request_id)
    balancer.release_server("s0", request_id)
    _assert_active(balancer, {"s0": (0, 0, 0), "s1": (1, 2, 1)})
    balancer.release_server("s1", "other")
    _assert_active(balancer, {"s0": (0, 0, 0), "s1": (0, 0, 0)})
    assert sum(balancer._store.get_metric(s, MetricKey.DISPATCHED_COUNT) for s in ("s0", "s1")) == 4
    assert sum(balancer._store.get_metric(s, MetricKey.COMPLETED_COUNT) for s in ("s0", "s1")) == 2
    assert all(balancer._store.inflight_block_count(s) == 0 for s in ("s0", "s1"))


def test_duplicate_release_does_not_debit_another_request_on_same_replica():
    balancer = _make_balancer({"s0": "h0"})
    balancer.acquire_server("first", [1, 2])
    balancer.acquire_server("second", [1])
    _assert_active(balancer, {"s0": (2, 3, 2)})
    balancer.release_server("s0", "first")
    balancer.release_server("s0", "first")
    _assert_active(balancer, {"s0": (1, 1, 1)})
    balancer.release_server("s0", "second")
    _assert_active(balancer, {"s0": (0, 0, 0)})
    # Reuse after a completed dispatch is a new active turn.
    balancer.acquire_server("first", [1, 2, 3])
    _assert_active(balancer, {"s0": (1, 3, 2)})
    balancer.release_server("s0", "first")
    _assert_active(balancer, {"s0": (0, 0, 0)})


def test_removed_replica_release_does_not_debit_readded_replica_work():
    balancer = _make_balancer({"s0": "h0"})
    balancer.acquire_server("removed", [1, 2])
    balancer.remove_servers(["s0"])
    assert balancer.get_total_inflight() == 0  # drain covers only the current pool
    balancer.add_servers({"s0": "new-h0"})
    balancer.acquire_server("new", [1])
    balancer.release_server("s0", "removed")
    _assert_active(balancer, {"s0": (1, 1, 1)})
    balancer.release_server("s0", "new")
    _assert_active(balancer, {"s0": (0, 0, 0)})
