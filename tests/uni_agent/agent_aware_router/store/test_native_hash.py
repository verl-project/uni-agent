"""Full events resolve sparse group identities without request preprocessing."""

import pytest

from uni_agent.agent_aware_router.collectors.parse.base import KVCacheUpdate
from uni_agent.agent_aware_router.store.group_cache import GroupCache
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental

pytestmark = [pytest.mark.cpu, pytest.mark.level0]

# vLLM 0.29.0 SHA256, seed="42", unit=4, integer event hashes.
HASHES = ["14690410320281468560", "18067663692984908069", "11820907415473743539", "2464756095330721020"]


@pytest.fixture
def store(layout):
    value = KVCacheStore()
    metadata = layout((("sliding_window", 4), ("full_attention", 8)), unit=None, window=9)
    metadata["native_hash_config"] = {
        "algorithm": "sha256",
        "seed": "42",
        "unit": 4,
        "pickle_protocol": 5,
        "int_hashes": True,
    }
    value.configure_registry(metadata)
    return value, metadata


@pytest.fixture
def events(stored, parse):
    full = parse([stored(1, "full_attention", 8, hashes=HASHES[1::2])])
    sparse = parse([stored(0, "sliding_window", 4, hashes=HASHES[1:], kv_cache_spec_sliding_window=9)])
    return full, sparse


def hit(cache):
    hashes, _ = get_prefix_hashes_incremental(list(range(17)), 4, 0, 0)
    return cache.estimate([str(h) for h in hashes], 17).estimated_hit_tokens


@pytest.mark.parametrize("full_node,version", [("a", "0.23.0"), ("b", "0.31.0")])
def test_full_event_identity_is_shared_across_replica_caches(full_node, version, store, events):
    value, metadata = store
    metadata["vllm_version"] = version
    value.configure_registry(metadata)
    full, sparse = events
    value.apply_update(KVCacheUpdate(node_id="a", group_events=sparse))
    a = value.group_caches["a"]
    assert hit(a) == 0 and a.pending
    value.apply_update(KVCacheUpdate(node_id=full_node, group_events=full))
    assert not a.pending and len(a.by_group[0]) == 3
    assert hit(a) == (16 if full_node == "a" else 0)
    value.apply_update(KVCacheUpdate(node_id="a", group_events=full))
    assert hit(a) == 16


def test_restored_identity_preserves_copy_counts_and_clear(store, events, parse, removed):
    value, _ = store
    cache = GroupCache(value.registry_state.registry, "a", native_hashes=value.native_hashes)
    full, sparse = events
    cache.apply(full + sparse + sparse)
    assert hit(cache) == 16
    for expected in (16, 0):
        cache.apply(parse([removed(0, [HASHES[2]])]))
        assert hit(cache) == expected
    cache.apply(parse([{"type": "AllBlocksCleared"}]))
    cache.apply(full)
    assert hit(cache) == 0
    cache.apply(sparse)
    assert hit(cache) == 16


@pytest.mark.parametrize("seed", [None, "random", "43"])
def test_unavailable_or_wrong_hash_seed_keeps_conservative_estimates(seed, store, events):
    value, metadata = store
    metadata["native_hash_config"]["seed"] = seed
    value.configure_registry(metadata)
    full, sparse = events
    cache = GroupCache(value.registry_state.registry, "a", native_hashes=value.native_hashes)
    cache.apply(full + sparse)
    assert value.native_hashes is None or not value.native_hashes.enabled
    assert hit(cache) == 0 and cache.pending


@pytest.mark.parametrize(
    "groups", [(("full_attention", 4),), (("full_attention", 4), ("mamba", 8)), (("sliding_window", 4),)]
)
def test_layouts_without_missing_full_endpoints_skip_native_hashing(groups, store, layout):
    value, metadata = store
    raw = layout(groups, unit=None)
    raw["native_hash_config"] = metadata["native_hash_config"]
    value.configure_registry(raw)
    assert value.native_hashes is None
