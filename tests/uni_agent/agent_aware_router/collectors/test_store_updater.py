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

"""Application/observation ordering and shared capability compatibility."""

import pytest

from uni_agent.agent_aware_router.collectors.parse import (
    KVCacheUpdate,
    MetricsUpdate,
    StickyUpdate,
)
from uni_agent.agent_aware_router.collectors.updater import StoreUpdater
from uni_agent.agent_aware_router.store.capabilities import RequestState
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.store.per_replica_store import PerReplicaStore
from uni_agent.agent_aware_router.store.per_request_store import PerRequestStore
from uni_agent.agent_aware_router.types import InflightUpdate, Layer, MetricKey

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


class Observer:
    def __init__(self, store):
        self.store = store
        self.calls = []

    def on_metrics_applied(self, update, values):
        assert all(self.store.metrics.get_metric(update.node_id, key) == value for key, value in values.items())
        self.calls.append(("metrics", values))

    def on_kv_applied(self, update):
        self.calls.append(("kv", self.store.kv.per_replica_block_counts()))

    def on_blocks_recorded(self, count):
        self.calls.append(("blocks", count))


@pytest.fixture
def application(monkeypatch):
    for cls in (KVCacheStore, PerReplicaStore, PerRequestStore):
        monkeypatch.setattr(cls, "_instance", None)
    store = DataStore()
    observer = Observer(store)
    return store, observer, StoreUpdater(store, observer)


def test_metrics_observer_receives_merged_and_incremented_post_write_values(application):
    store, observer, updater = application
    store.refresh_metrics({"s0": {MetricKey.NUM_REQUESTS_WAITING: 7}, "s1": {MetricKey.INFLIGHT_COUNT: 9}})
    updater.apply_metrics(MetricsUpdate("s0", {MetricKey.NUM_REQUESTS_RUNNING: 2}))
    assert observer.calls[-1][1] == {MetricKey.NUM_REQUESTS_WAITING: 7, MetricKey.NUM_REQUESTS_RUNNING: 2}
    updater.apply_metrics(MetricsUpdate("s0", {MetricKey.INFLIGHT_COUNT: 1}, is_delta=True))
    updater.apply_metrics(MetricsUpdate("s0", {MetricKey.INFLIGHT_COUNT: -1}, is_delta=True))
    assert observer.calls[-1] == ("metrics", {MetricKey.INFLIGHT_COUNT: 0})
    assert store.get_metric("s1", MetricKey.INFLIGHT_COUNT) == 9


def test_failed_store_write_is_not_observed(application, monkeypatch):
    store, observer, updater = application

    def fail(*args):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(store.metrics, "refresh_metrics", fail)
    with pytest.raises(RuntimeError, match="store unavailable"):
        updater.apply_metrics(MetricsUpdate("s0", {}))
    assert observer.calls == []


def test_pin_is_observed_only_after_its_marker_is_written(application, monkeypatch):
    store, observer, updater = application

    def record(count):
        assert store.get_marker("r")["replica"] == "s0"
        assert store.requests.get_turn("r") == 1
        observer.calls.append(("blocks", count))

    monkeypatch.setattr(observer, "on_blocks_recorded", record)
    store.set_block_size(2)
    updater.apply_inflight(InflightUpdate("acquire", "s0", "r", (1, 2)))
    assert observer.calls[0] == ("blocks", 1)
    assert observer.calls[1][0] == "metrics"


def test_kv_clear_remove_add_order_and_first_block_size_wins(application):
    store, observer, updater = application
    store.add_kv_blocks("s0", ["old", "shared"])
    store.add_kv_blocks("s1", ["other"])
    updater.apply_kv(
        KVCacheUpdate(
            "s0",
            add_blocks={Layer.GPU: ["shared", "new"]},
            remove_blocks={Layer.GPU: ["shared"]},
            clear_all=True,
            block_size=16,
        )
    )
    assert observer.calls == [("kv", {"s1": 1, "s0": 2})]
    assert not store.has_kv_block("old")
    updater.apply_kv(KVCacheUpdate("s0", block_size=32))
    assert store.get_block_size() == 16


def test_inflight_updater_delegates_without_interpreting_callback_state(application, monkeypatch):
    from uni_agent.agent_aware_router.store.inflight import InflightMetricsApplied, InflightResult

    store, observer, updater = application
    fact = InflightUpdate("release", "s0", "r")
    deltas = {MetricKey.INFLIGHT_COUNT: -1}
    values = {MetricKey.INFLIGHT_COUNT: 0}
    seen = []

    def apply(update):
        assert update is fact
        seen.append(update)
        store.metrics.refresh_metrics({"s0": values})
        return InflightResult([InflightMetricsApplied("s0", deltas, values)], blocks_recorded=2)

    monkeypatch.setattr(store.inflight, "apply", apply)
    updater.apply_inflight(fact)
    assert seen == [fact]
    assert observer.calls == [("blocks", 2), ("metrics", values)]


def test_failed_inflight_transition_is_not_observed(application, monkeypatch):
    store, observer, updater = application

    def fail(update):
        raise RuntimeError("transition failed")

    monkeypatch.setattr(store.inflight, "apply", fail)
    with pytest.raises(RuntimeError, match="transition failed"):
        updater.apply_inflight(InflightUpdate("acquire", "s0", "r"))
    assert observer.calls == []


def test_observer_failure_cannot_leave_an_applied_dispatch_unbooked(application, monkeypatch):
    store, observer, updater = application
    store.set_block_size(2)
    original = observer.on_metrics_applied

    def fail(*args):
        raise RuntimeError("observer unavailable")

    monkeypatch.setattr(observer, "on_metrics_applied", fail)
    with pytest.raises(RuntimeError, match="observer unavailable"):
        updater.apply_inflight(InflightUpdate("acquire", "s0", "r", (1, 2)))
    # DataStore finishes the transition before the updater starts observations.
    assert store.requests.get_turn("r") == 1
    assert store.get_metric("s0", MetricKey.INFLIGHT_COUNT) == 1
    monkeypatch.setattr(observer, "on_metrics_applied", original)
    updater.apply_inflight(InflightUpdate("release", "s0", "r"))
    for key in (MetricKey.INFLIGHT_COUNT, MetricKey.INFLIGHT_TOKENS, MetricKey.INFLIGHT_TURN_SUM):
        assert store.get_metric("s0", key) == 0
    assert store.get_marker("r") is None
    assert store.inflight_block_count("s0") == 0


def test_sticky_updates_keep_other_request_state(application):
    store, _, updater = application
    for rid, replica in [("r1", "s0"), ("r2", "s0"), ("r3", "s1")]:
        store.requests.next_turn(rid)
        updater.apply_sticky(StickyUpdate("put", rid, replica))
    updater.apply_sticky(StickyUpdate("invalidate", "r1"))
    updater.apply_sticky(StickyUpdate("invalidate_replica", replica_ids=("s0",)))
    assert [store.get_sticky_binding(r) for r in ["r1", "r2", "r3"]] == [None, None, "s1"]
    assert store.requests.clear_sticky_bindings() == 1
    assert store.requests.get_turn("r1") == store.requests.get_turn("r2") == store.requests.get_turn("r3") == 1


def test_named_request_operations_share_the_original_lru_lifetime():
    requests = RequestState(PerRequestStore(max_size=2))
    requests.next_turn("old")
    requests.book_prompt_length("old", 17)
    requests.put_sticky_binding("old", "s0")
    requests.set_prefix_cache("old", {"hash_strs": ["h"], "parent_hash": 1})
    requests.next_turn("middle")
    requests.next_turn("new")
    assert requests.get_turn("old") == requests.get_prompt_length("old") == 0
    assert requests.get_sticky_binding("old") is None
    assert requests.get_prefix_cache("old") is None
