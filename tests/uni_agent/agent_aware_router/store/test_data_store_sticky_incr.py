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

"""Tests for PerReplicaStore.incr and DataStore sticky-session delegation.

Covers incremental writes (inflight ±1, keeping the parser stateless) and the
DataStore facade storing sticky bindings in PerRequestStore.
"""

from __future__ import annotations

import pytest

from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.store.per_replica_store import PerReplicaStore
from uni_agent.agent_aware_router.store.per_request_store import PerRequestStore
from uni_agent.agent_aware_router.types import MetricKey

pytestmark = [pytest.mark.level0, pytest.mark.cpu]


# ── PerReplicaStore.incr (plain instances — isolated, not the singleton) ──


class TestPerReplicaStoreIncr:
    def test_incr_isolates_nodes(self):
        s = PerReplicaStore()
        s.incr("n0", MetricKey.INFLIGHT_COUNT)
        s.incr("n1", MetricKey.INFLIGHT_COUNT, 3)
        assert s.get("n0", MetricKey.INFLIGHT_COUNT) == 1
        assert s.get("n1", MetricKey.INFLIGHT_COUNT) == 3

    def test_incr_does_not_clobber_other_keys(self):
        s = PerReplicaStore()
        s.refresh({"n0": {MetricKey.NUM_REQUESTS_RUNNING: 7}})
        s.incr("n0", MetricKey.INFLIGHT_COUNT)
        assert s.get("n0", MetricKey.NUM_REQUESTS_RUNNING) == 7
        assert s.get("n0", MetricKey.INFLIGHT_COUNT) == 1

    def test_incr_unknown_key_raises(self):
        s = PerReplicaStore()
        with pytest.raises(KeyError):
            s.incr("n0", "not_a_real_key")

    def test_incr_many_applies_all_deltas(self):
        """Batched write matches per-key incr semantics (defaults, sums, isolation)."""
        s = PerReplicaStore()
        s.incr_many(
            "n0",
            {
                MetricKey.INFLIGHT_COUNT: 1,
                MetricKey.DISPATCHED_COUNT: 1,
                MetricKey.PROMPT_LEN_SUM: 2163,
            },
        )
        assert s.get("n0", MetricKey.INFLIGHT_COUNT) == 1
        assert s.get("n0", MetricKey.DISPATCHED_COUNT) == 1
        assert s.get("n0", MetricKey.PROMPT_LEN_SUM) == 2163

    def test_incr_many_unknown_key_raises(self):
        s = PerReplicaStore()
        with pytest.raises(KeyError):
            s.incr_many("n0", {MetricKey.INFLIGHT_COUNT: 1, "not_a_real_key": 1})


# ── DataStore sticky-session delegation + inflight (singleton-backed) ──


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Isolate each test from the global PerReplicaStore / PerRequestStore singletons."""
    PerRequestStore._instance = None
    PerReplicaStore._instance = None
    yield
    PerRequestStore._instance = None
    PerReplicaStore._instance = None


class TestDataStoreStickyDelegation:
    def test_invalidate_binding(self):
        ds = DataStore()
        ds.put_sticky_binding("r1", "s0")
        ds.invalidate_sticky_binding("r1")
        assert ds.get_sticky_binding("r1") is None

    def test_invalidate_replica_clears_all_bound(self):
        ds = DataStore()
        ds.put_sticky_binding("r1", "s0")
        ds.put_sticky_binding("r2", "s1")
        ds.put_sticky_binding("r3", "s0")
        ds.invalidate_sticky_replica("s0")
        assert ds.get_sticky_binding("r1") is None
        assert ds.get_sticky_binding("r3") is None
        assert ds.get_sticky_binding("r2") == "s1"

    def test_sticky_status_reports_size(self):
        ds = DataStore()
        ds.put_sticky_binding("r1", "s0")
        ds.put_sticky_binding("r2", "s1")
        assert ds.sticky_status()["size"] == 2

    def test_put_get_overwrite_and_missing(self):
        # put → get; overwrite same request_id → latest wins; missing → None
        ds = DataStore()
        ds.put_sticky_binding("r1", "s0")
        assert ds.get_sticky_binding("r1") == "s0"  # put then get
        ds.put_sticky_binding("r1", "s1")  # overload-fallback re-routes
        assert ds.get_sticky_binding("r1") == "s1"  # overwrite: latest wins
        assert ds.get_sticky_binding("ghost") is None  # missing → None

    def test_invalidate_missing_target_is_noop(self):
        # invalidate by request_id and by replica, both for non-existent targets
        ds = DataStore()
        ds.put_sticky_binding("r1", "s0")
        ds.invalidate_sticky_binding("ghost")  # unknown request_id — must not raise
        assert ds.get_sticky_binding("ghost") is None
        ds.invalidate_sticky_replica("sX")  # no binding points at sX
        assert ds.get_sticky_binding("r1") == "s0"  # untouched binding survives


# ── Write methods return post-write values (§23 modified-X foundation) ──


class TestPerReplicaStoreWriteReturns:
    """incr / incr_many / refresh return the values they just wrote."""

    def test_incr_returns_new_value(self):
        s = PerReplicaStore()
        assert s.incr("n0", MetricKey.INFLIGHT_COUNT) == 1
        assert s.incr("n0", MetricKey.INFLIGHT_COUNT, 2) == 3

    def test_incr_negative_delta_returned(self):
        s = PerReplicaStore()
        s.incr("n0", MetricKey.INFLIGHT_COUNT, 5)
        assert s.incr("n0", MetricKey.INFLIGHT_COUNT, -2) == 3

    def test_incr_many_returns_all_new_values(self):
        s = PerReplicaStore()
        out = s.incr_many("n0", {MetricKey.INFLIGHT_COUNT: 1, MetricKey.DISPATCHED_COUNT: 1})
        assert out == {MetricKey.INFLIGHT_COUNT: 1, MetricKey.DISPATCHED_COUNT: 1}
        out2 = s.incr_many("n0", {MetricKey.INFLIGHT_COUNT: 1, MetricKey.PROMPT_LEN_SUM: 100})
        assert out2 == {MetricKey.INFLIGHT_COUNT: 2, MetricKey.PROMPT_LEN_SUM: 100}

    def test_incr_many_empty_returns_empty_dict(self):
        s = PerReplicaStore()
        assert s.incr_many("n0", {}) == {}

    def test_refresh_returns_full_merged_snapshot_including_preexisting(self):
        s = PerReplicaStore()
        s.incr("n0", MetricKey.INFLIGHT_COUNT, 7)  # pre-existing key not in the refresh
        snap = s.refresh({"n0": {MetricKey.NUM_REQUESTS_RUNNING: 3}})
        assert snap == {"n0": {MetricKey.INFLIGHT_COUNT: 7, MetricKey.NUM_REQUESTS_RUNNING: 3}}

    def test_refresh_snapshot_is_a_copy(self):
        s = PerReplicaStore()
        snap = s.refresh({"n0": {MetricKey.NUM_REQUESTS_RUNNING: 3}})
        snap["n0"][MetricKey.NUM_REQUESTS_RUNNING] = 999  # mutate must not leak into the store
        assert s.get("n0", MetricKey.NUM_REQUESTS_RUNNING) == 3


class TestDataStoreWriteReturns:
    """DataStore facade passes the post-write returns through."""

    def test_incr_metric_returns_new_value(self):
        ds = DataStore()
        assert ds.incr_metric("n0", MetricKey.INFLIGHT_COUNT) == 1

    def test_incr_metrics_returns_new_values(self):
        ds = DataStore()
        out = ds.incr_metrics("n0", {MetricKey.DISPATCHED_COUNT: 1, MetricKey.PROMPT_LEN_SUM: 50})
        assert out == {MetricKey.DISPATCHED_COUNT: 1, MetricKey.PROMPT_LEN_SUM: 50}

    def test_refresh_metrics_returns_snapshot(self):
        ds = DataStore()
        snap = ds.refresh_metrics({"n0": {MetricKey.NUM_REQUESTS_RUNNING: 2}})
        assert snap == {"n0": {MetricKey.NUM_REQUESTS_RUNNING: 2}}
