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

"""Resident-view rebuilding stays separate from live request accounting."""

import pytest

from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.types import Layer

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


class _ObservedBlockIndex(dict):
    """Keep real ownership data while recording which hashes cleanup inspects."""

    def __init__(self, blocks):
        super().__init__(blocks)
        self.inspected = []

    def items(self):
        for block_hash, replicas in super().items():
            self.inspected.append(block_hash)
            yield block_hash, replicas

    def get(self, block_hash, default=None):
        self.inspected.append(block_hash)
        return super().get(block_hash, default)

    def __getitem__(self, block_hash):
        self.inspected.append(block_hash)
        return super().__getitem__(block_hash)


def test_invalidate_only_inspects_the_target_replica_resident_blocks():
    store = KVCacheStore()
    other_hashes = [f"other-{i}" for i in range(20)]
    store.add_blocks("b", ["shared", *other_hashes])
    assert store.record_dispatch_blocks("a", ["shared", "dispatch-only"]) == 2
    store.add_blocks("a", ["shared", "event-only", "event-only"])
    assert store.per_replica_block_counts() == {"a": 3, "b": 21}
    for replica_id in ("a", "b"):
        store.incr_block_refs(replica_id, ["shared"])
        store.put_marker(f"request-{replica_id}", replica_id, ["shared"])
    index = _ObservedBlockIndex(store.replicas_by_block)
    store.replicas_by_block = index

    store.invalidate_replica("a")

    assert set(index.inspected) == {"shared", "dispatch-only", "event-only"}
    assert index == {"shared": {"b"}, **{block_hash: {"b"} for block_hash in other_hashes}}
    assert store.per_replica_block_counts() == {"b": 21}
    for replica_id in ("a", "b"):
        assert store.inflight_block_count(replica_id) == 1
        assert store.get_marker(f"request-{replica_id}") == {"replica": replica_id, "hashes": ["shared"]}

    index.inspected.clear()
    store.invalidate_replica("a")
    assert index.inspected == []

    # A failed partial rebuild must clear its newly recorded ownership too.
    store.add_blocks("a", ["shared", "partial"])
    index.inspected.clear()
    store.invalidate_replica("a")
    assert set(index.inspected) == {"shared", "partial"}
    assert "partial" not in index
    assert index["shared"] == {"b"}
    assert store.per_replica_block_counts() == {"b": 21}


def test_removed_resident_blocks_are_not_revisited_by_invalidation():
    store = KVCacheStore()
    store.add_blocks("b", ["shared"])
    store.record_dispatch_blocks("a", ["shared", "removed"])
    store.add_blocks("a", ["shared", "removed"])
    store.remove_blocks("a", ["removed", "removed", "unknown", "shared"])
    index = _ObservedBlockIndex(store.replicas_by_block)
    store.replicas_by_block = index

    store.invalidate_replica("a")

    assert index.inspected == []
    assert index == {"shared": {"b"}}
    assert store.per_replica_block_counts() == {"b": 1}


@pytest.mark.parametrize("clear_method", ["invalidate_replica", "clear_replica"])
def test_clear_without_gpu_ownership_clears_other_layer_counts_without_scanning(clear_method):
    store = KVCacheStore()
    store.add_blocks("b", ["shared"])
    store.add_blocks("a", ["cpu-1", "cpu-2"], Layer.CPU)
    store.add_blocks("a", ["ssd"], Layer.SSD)
    store.add_blocks("b", ["other-cpu"], Layer.CPU)
    index = _ObservedBlockIndex(store.replicas_by_block)
    store.replicas_by_block = index

    getattr(store, clear_method)("a")

    assert index.inspected == []
    assert index == {"shared": {"b"}}
    assert store._replica_layer_counts == {Layer.GPU: {"b": 1}, Layer.CPU: {"b": 1}, Layer.SSD: {}}


def test_invalidate_gates_rebuilt_hits_and_dispatch_without_losing_request_state():
    store = KVCacheStore()
    store.block_size = 2
    for replica_id in ("a", "b"):
        store.add_blocks(replica_id, ["h1", "h2"])
        store.add_blocks(replica_id, ["cpu"], Layer.CPU)
        store.add_blocks(replica_id, ["ssd"], Layer.SSD)
        store.incr_block_refs(replica_id, ["h1", "held"])
        store.put_marker(f"request-{replica_id}", replica_id, ["h1", "held"])
    store.add_blocks("a", ["old-only"])

    store.invalidate_replica("a")

    assert store.replicas_by_block == {"h1": {"b"}, "h2": {"b"}}
    assert all(counts == {"b": 1 if layer != Layer.GPU else 2} for layer, counts in store._replica_layer_counts.items())
    for replica_id in ("a", "b"):
        assert store.inflight_block_count(replica_id) == 2
        assert store.get_marker(f"request-{replica_id}") == {"replica": replica_id, "hashes": ["h1", "held"]}
    assert store.get_layer_prefix_hit_rate("b", ["h1", "h2"]) == 1.0

    # Events can build the resident view while router acquire/release accounting
    # remains active. A dispatched request cannot inject optimistic residency.
    store.add_blocks("a", ["h1", "h2", "removed-during-replay"])
    store.remove_blocks("a", ["removed-during-replay"])
    assert store.get_layer_prefix_hit_rate("a", ["h1", "h2"]) == 0.0
    assert store.record_dispatch_blocks("a", ["dispatch-only"]) == 0
    assert "dispatch-only" not in store.replicas_by_block
    store.incr_block_refs("a", ["during-replay"])
    store.put_marker("request-during-replay", "a", ["during-replay"])
    store.decr_block_refs("a", ["held"])
    assert store.inflight_block_count("a") == 2
    assert store.get_marker("request-during-replay") == {"replica": "a", "hashes": ["during-replay"]}

    store.mark_replica_ready("a")
    assert store.get_layer_prefix_hit_rate("a", ["h1", "h2"]) == 1.0
    assert store.record_dispatch_blocks("a", ["dispatch-only"]) == 1
    assert store.get_layer_prefix_hit_rate("b", ["h1", "h2"]) == 1.0
    assert store.inflight_block_count("b") == 2


@pytest.mark.parametrize("gated", [False, True], ids=["ready", "rebuilding"])
def test_clear_resident_blocks_preserves_request_state_and_rebuilding(gated):
    store = KVCacheStore()
    store.block_size = 2
    if gated:
        store.invalidate_replica("a")
    for replica_id in ("a", "b"):
        store.add_blocks(replica_id, ["shared"])
        store.add_blocks(replica_id, ["cpu"], Layer.CPU)
        store.add_blocks(replica_id, ["ssd"], Layer.SSD)
        store.incr_block_refs(replica_id, ["shared", "held"])
        store.put_marker(f"request-{replica_id}", replica_id, ["shared", "held"])

    store.clear_resident_blocks("a")

    assert store.replicas_by_block == {"shared": {"b"}}
    assert store._replica_layer_counts == {Layer.GPU: {"b": 1}, Layer.CPU: {"b": 1}, Layer.SSD: {"b": 1}}
    assert store._rebuilding_replicas == ({"a"} if gated else set())
    assert store.block_size == 2
    for replica_id in ("a", "b"):
        assert store.inflight_block_count(replica_id) == 2
        assert store.get_marker(f"request-{replica_id}") == {"replica": replica_id, "hashes": ["shared", "held"]}
    assert store.get_layer_prefix_hit_rate("b", ["shared"]) == 1.0
    store.add_blocks("a", ["rebuilt"])
    assert store.get_layer_prefix_hit_rate("a", ["rebuilt"]) == (0.0 if gated else 1.0)
    assert store.record_dispatch_blocks("a", ["dispatch-only"]) == (0 if gated else 1)
    if gated:
        store.mark_replica_ready("a")
        assert store.get_layer_prefix_hit_rate("a", ["rebuilt"]) == 1.0
        assert store.record_dispatch_blocks("a", ["dispatch-only"]) == 1


def test_clear_replica_cleans_request_state_but_does_not_end_rebuilding():
    store = KVCacheStore()
    store.block_size = 2
    for replica_id in ("a", "b"):
        store.add_blocks(replica_id, ["h1"])
        store.incr_block_refs(replica_id, ["h1"])
        store.put_marker(f"request-{replica_id}", replica_id, ["h1"])
    store.invalidate_replica("a")
    store.add_blocks("a", ["before-clear"])

    store.clear_replica("a")

    assert store.inflight_block_count("a") == 0
    assert store.get_marker("request-a") is None
    assert "before-clear" not in store.replicas_by_block
    assert store.inflight_block_count("b") == 1
    assert store.get_marker("request-b") == {"replica": "b", "hashes": ["h1"]}
    store.add_blocks("a", ["h1"])
    assert store.get_layer_prefix_hit_rate("a", ["h1"]) == 0.0
    assert store.record_dispatch_blocks("a", ["dispatch-only"]) == 0
    store.mark_replica_ready("a")
    assert store.get_layer_prefix_hit_rate("a", ["h1"]) == 1.0


def test_data_store_repeated_invalidation_discards_partial_replay_until_ready(monkeypatch):
    kv = KVCacheStore()
    monkeypatch.setattr(KVCacheStore, "_instance", kv)
    store = DataStore()
    store.set_block_size(2)
    store.add_kv_blocks("a", ["old"])
    store.incr_block_refs("a", ["held"])
    store.put_marker("request", "a", ["held"])

    store.invalidate_kv_replica("a")
    store.add_kv_blocks("a", ["partial"])
    store.invalidate_kv_replica("a")
    assert not store.has_kv_block("partial")
    store.add_kv_blocks("a", ["rebuilt"])
    assert store.get_layer_prefix_hit_rate("a", ["rebuilt"]) == 0.0
    assert store.record_dispatch_blocks("a", ["optimistic"]) == 0
    assert store.inflight_block_count("a") == 1
    assert store.get_marker("request") == {"replica": "a", "hashes": ["held"]}

    store.mark_kv_replica_ready("a")
    assert store.get_layer_prefix_hit_rate("a", ["rebuilt"]) == 1.0
    store.decr_block_refs("a", ["held"])
    assert store.inflight_block_count("a") == 0
    assert store.pop_marker("request") == {"replica": "a", "hashes": ["held"]}


def test_dispatch_estimates_discard_old_residency_and_preserve_request_accounting():
    store = KVCacheStore()
    store.block_size = 2
    for replica_id in ("a", "b"):
        store.add_blocks(replica_id, ["shared", "old"])
        store.add_blocks(replica_id, ["cpu"], Layer.CPU)
        store.add_blocks(replica_id, ["ssd"], Layer.SSD)
        store.incr_block_refs(replica_id, ["shared", "held"])
        store.put_marker(f"request-{replica_id}", replica_id, ["shared", "held"])

    store.enable_dispatch_estimates("a")

    assert store.replicas_by_block == {"shared": {"b"}, "old": {"b"}}
    assert store._replica_layer_counts == {Layer.GPU: {"b": 2}, Layer.CPU: {"b": 1}, Layer.SSD: {"b": 1}}
    assert store.block_size == 2
    for replica_id in ("a", "b"):
        assert store.inflight_block_count(replica_id) == 2
        assert store.get_marker(f"request-{replica_id}") == {"replica": replica_id, "hashes": ["shared", "held"]}
    assert store.record_dispatch_blocks("a", ["shared", "estimate"]) == 2
    assert store.record_dispatch_blocks("a", ["shared", "estimate"]) == 0
    assert store.get_layer_prefix_hit_rate("a", ["shared", "estimate"]) == 1.0
    assert store.get_layer_prefix_hit_rate("a", ["old"]) == 0.0
    assert store.get_layer_prefix_hit_rate("b", ["shared", "old"]) == 1.0
    store.remove_blocks("a", ["shared", "estimate"])
    assert store.replicas_by_block == {"shared": {"b"}, "old": {"b"}}


def test_replay_gates_estimates_and_failed_replay_starts_a_fresh_dispatch_view(monkeypatch):
    kv = KVCacheStore()
    monkeypatch.setattr(KVCacheStore, "_instance", kv)
    store = DataStore()
    store.set_block_size(2)
    store.add_kv_blocks("b", ["shared", "other"])
    store.incr_block_refs("a", ["held"])
    store.put_marker("request", "a", ["held"])
    store.add_kv_blocks("a", ["old"])
    store.enable_kv_dispatch_estimates("a")
    assert store.record_dispatch_blocks("a", ["fallback"]) == 1
    assert store.get_layer_prefix_hit_rate("a", ["fallback"]) == 1.0

    store.invalidate_kv_replica("a")
    assert not store.has_kv_block("fallback")
    assert store.record_dispatch_blocks("a", ["blocked"]) == 0
    store.add_kv_blocks("a", ["shared", "partial"])
    assert store.get_layer_prefix_hit_rate("a", ["shared", "partial"]) == 0.0

    store.enable_kv_dispatch_estimates("a")
    assert not store.has_kv_block("partial")
    assert store.get_layer_prefix_hit_rate("a", ["shared"]) == 0.0
    assert store.record_dispatch_blocks("a", ["shared", "new-estimate"]) == 2
    assert store.get_layer_prefix_hit_rate("a", ["shared", "new-estimate"]) == 1.0
    assert store.per_replica_block_counts() == {"a": 2, "b": 2}

    store.invalidate_kv_replica("a")
    assert not store.has_kv_block("new-estimate")
    store.add_kv_blocks("a", ["rebuilt"])
    assert store.get_layer_prefix_hit_rate("a", ["rebuilt"]) == 0.0
    store.mark_kv_replica_ready("a")
    assert store.get_layer_prefix_hit_rate("a", ["rebuilt"]) == 1.0
    assert store.get_layer_prefix_hit_rate("b", ["shared", "other"]) == 1.0
    assert store.inflight_block_count("a") == 1
    assert store.get_marker("request") == {"replica": "a", "hashes": ["held"]}
    assert store.get_block_size() == 2
