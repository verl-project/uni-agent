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

"""Hybrid resident rebuilding is separate from active request reservations."""

import pytest

from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.types.kv_cache import KVCacheUpdate
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental


@pytest.fixture
def cache(layout, install_metadata):
    value = KVCacheStore()
    install_metadata(value, "a", layout())
    install_metadata(value, "b", layout())
    return value


def populate(cache, parse, stored, replica="a"):
    events = parse([stored(), stored(1, "mamba", 8, hashes=["r16"])], node=replica)
    cache.apply_update(KVCacheUpdate(replica, events))


def estimate(cache, replica="a"):
    hashes = list(map(str, get_prefix_hashes_incremental(list(range(17)), cache.hash_schema()[1], 0, 0)[0]))
    return cache.estimate_prefixes([replica], hashes, 17)[replica]


def pin(cache, request, replica="a"):
    cache.incr_block_refs(replica, ["held"])
    cache.put_marker(request, replica, ["held"])


def release(cache, request):
    marker = cache.pop_marker(request)
    cache.decr_block_refs(marker["replica"], marker["hashes"])


@pytest.mark.cpu
@pytest.mark.level0
def test_invalidate_gates_rebuilt_hits_and_dispatch_without_losing_request_state(cache, parse, stored):
    for replica in ("a", "b"):
        populate(cache, parse, stored, replica)
        pin(cache, "request-" + replica, replica)
    cache.invalidate_replica("a")
    assert estimate(cache).estimated_hit_tokens == 0
    assert estimate(cache, "b").estimated_hit_tokens == 16
    assert cache.inflight_block_count("a") == 1
    assert cache.get_marker("request-a") is not None
    populate(cache, parse, stored)
    assert estimate(cache).estimated_hit_tokens == 0
    hashes = list(map(str, get_prefix_hashes_incremental(list(range(17)), 2, 0, 0)[0]))
    assert cache.record_dispatch_blocks("a", hashes) == 0
    cache.mark_replica_ready("a")
    assert estimate(cache).estimated_hit_tokens == 16
    assert cache.inflight_block_count("a") == 1
    release(cache, "request-a")
    assert cache.inflight_block_count("a") == 0
    assert cache.inflight_block_count("b") == 1


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("gated", [False, True], ids=["ready", "rebuilding"])
def test_clear_resident_blocks_preserves_request_state_and_rebuilding(cache, parse, stored, gated):
    for replica in ("a", "b"):
        populate(cache, parse, stored, replica)
        pin(cache, "request-" + replica, replica)
    if gated:
        cache.invalidate_replica("a")
        populate(cache, parse, stored)
    marker = cache.get_marker("request-a")
    cache.clear_resident_blocks("a")
    assert cache._gated_replicas == ({"a"} if gated else set())
    assert estimate(cache).estimated_hit_tokens == 0
    assert estimate(cache, "b").estimated_hit_tokens == 16
    assert cache.inflight_block_count("a") == 1
    assert cache.get_marker("request-a") == marker
    populate(cache, parse, stored)
    assert estimate(cache).estimated_hit_tokens == (0 if gated else 16)
    if gated:
        cache.mark_replica_ready("a")
        assert estimate(cache).estimated_hit_tokens == 16


@pytest.mark.cpu
@pytest.mark.level0
def test_clear_replica_cleans_request_state_but_does_not_end_rebuilding(cache, parse, stored):
    for replica in ("a", "b"):
        populate(cache, parse, stored, replica)
        pin(cache, "request-" + replica, replica)
    cache.invalidate_replica("a")
    cache.clear_replica("a")
    assert cache.get_marker("request-a") is None
    assert cache.inflight_block_count("a") == 0
    assert cache.inflight_block_count("b") == 1
    populate(cache, parse, stored)
    assert estimate(cache).estimated_hit_tokens == 0
    cache.mark_replica_ready("a")
    assert estimate(cache).estimated_hit_tokens == 16


@pytest.mark.cpu
@pytest.mark.level0
def test_data_store_repeated_invalidation_discards_partial_replay_until_ready(cache, parse, stored, monkeypatch):
    monkeypatch.setattr(KVCacheStore, "_instance", cache)
    store = DataStore()
    pin(cache, "request")
    store.invalidate_kv_replica("a")
    populate(cache, parse, stored)
    store.invalidate_kv_replica("a")
    assert store.per_replica_block_counts() == {}
    populate(cache, parse, stored)
    assert estimate(cache).estimated_hit_tokens == 0
    store.mark_kv_replica_ready("a")
    assert estimate(cache).estimated_hit_tokens == 16
    assert store.inflight_block_count("a") == 1
    release(store, "request")
    assert store.inflight_block_count("a") == 0


@pytest.mark.cpu
@pytest.mark.level0
def test_dispatch_estimates_discard_old_residency_and_preserve_request_accounting(cache, parse, stored):
    populate(cache, parse, stored)
    populate(cache, parse, stored, "b")
    pin(cache, "request")
    cache.invalidate_replica("a")
    populate(cache, parse, stored)
    cache.enable_dispatch_estimates("a")
    assert cache.per_replica_block_counts() == {"b": 5}
    assert estimate(cache).estimated_hit_tokens == 0
    hashes = list(map(str, get_prefix_hashes_incremental(list(range(17)), 2, 0, 0)[0]))
    assert cache.record_dispatch_blocks("a", hashes) == 6
    assert estimate(cache).estimated_hit_tokens == 16
    assert cache.inflight_block_count("a") == 1
    cache.invalidate_replica("a")
    assert estimate(cache).estimated_hit_tokens == 0
    assert cache.inflight_block_count("a") == 1
    assert estimate(cache, "b").estimated_hit_tokens == 16
