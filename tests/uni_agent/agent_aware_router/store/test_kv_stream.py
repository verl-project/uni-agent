"""Ordered cache boundaries with authoritative engine layouts."""

import msgpack
import pytest

from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental


def _apply(cache, events, node="a"):
    frame = msgpack.packb([0, events])
    cache.apply_update(VLLMKVParser().parse(frame, node))


def _old_stored(remote="cached", tokens=None, block=4):
    return ["BlockStored", [remote], None, list(range(block)) if tokens is None else tokens, block, None, None]


def _estimate(cache, node="a", tokens=None, unit=4):
    tokens = list(range(5)) if tokens is None else tokens
    hashes, _ = get_prefix_hashes_incremental(tokens, unit, 0, 0)
    return cache.estimate_prefixes([node], [str(value) for value in hashes], len(tokens))[node]


@pytest.mark.cpu
@pytest.mark.level0
def test_clear_discards_stores_before_the_boundary_and_isolates_nodes(layout, install_metadata):
    cache = KVCacheStore()
    install_metadata(cache, "a", layout((("full_attention", 4),), unit=4))
    install_metadata(cache, "b", layout((("full_attention", 4),), unit=4))
    before = _old_stored("old")
    after = _old_stored("new", list(range(4, 8)))
    _apply(cache, [before], "b")
    _apply(cache, [before, ["AllBlocksCleared"], after])
    assert _estimate(cache).estimated_hit_tokens == 0
    assert _estimate(cache, tokens=list(range(4, 9))).estimated_hit_tokens == 4
    assert _estimate(cache, "b").estimated_hit_tokens == 4
    assert cache.per_replica_block_counts() == {"a": 1, "b": 1}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("status", [None, "not_ready", "unsupported", "error"])
def test_events_require_a_layout_before_they_can_match(status, layout, install_metadata):
    cache = KVCacheStore()
    initial = cache.hash_schema()
    if status is not None:
        install_metadata(cache, "a", {"status": status})
    _apply(cache, [_old_stored()])
    assert _estimate(cache).estimated_hit_tokens == 0
    assert cache.hash_schema() == initial

    install_metadata(cache, "a", layout((("full_attention", 4),), unit=4))
    generation, unit, _ = cache.hash_schema()
    assert generation > initial[0] and unit == 4
    assert _estimate(cache).estimated_hit_tokens == 0
    _apply(cache, [_old_stored()])
    assert _estimate(cache).estimated_hit_tokens == 4


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("metadata_state", ["invalid", "disabled", "multiple-groups"])
def test_explicit_layout_never_silently_falls_back_to_single_full(metadata_state, layout, install_metadata):
    cache = KVCacheStore()
    metadata = layout()
    if metadata_state == "invalid":
        metadata["groups"] = []
    elif metadata_state == "disabled":
        metadata["cache_config"]["enable_prefix_caching"] = False
    install_metadata(cache, "a", metadata)
    _apply(cache, [_old_stored()])
    assert _estimate(cache).estimated_hit_tokens == 0
    assert not any(cache.per_replica_block_counts().values())


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("evidence", ["tagged-group", "different-size", "partial-size"])
def test_unknown_layout_events_do_not_prevent_metadata_recovery(evidence, stored, layout):
    cache = KVCacheStore()
    _apply(cache, [_old_stored()])
    assert _estimate(cache).estimated_hit_tokens == 0
    conflicting = (
        stored(tokens=list(range(4)))
        if evidence == "tagged-group"
        else _old_stored(block=2 if evidence == "partial-size" else 8)
    )
    _apply(cache, [conflicting])
    assert _estimate(cache).estimated_hit_tokens == 0
    assert not cache.per_replica_block_counts()
    _apply(cache, [_old_stored()])
    assert _estimate(cache).estimated_hit_tokens == 0
    assert cache.refresh_replica_metadata("a", layout((("full_attention", 4),), unit=4))
    cache.mark_replica_ready("a")
    _apply(cache, [_old_stored()])
    assert _estimate(cache).estimated_hit_tokens == 4


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("sparse", [False, True], ids=["unindexed-tail", "sparse-range"])
def test_old_full_tail_adapter_does_not_trim_sparse_token_ranges(sparse, layout, install_metadata):
    cache = KVCacheStore()
    install_metadata(cache, "a", layout((("full_attention", 4),), unit=4))
    tokens = list(range(16 if sparse else 9))
    event = ["BlockStored", ["r8", "r12", "r16"] if sparse else ["r4", "r8"], None, tokens, 4, None, None]
    update = VLLMKVParser().parse(msgpack.packb([0, [event]]), "a")
    assert update.events[0].raw_token_ids == tuple(tokens)
    cache.apply_update(update)
    assert _estimate(cache, tokens=list(range(17))).estimated_hit_tokens == (0 if sparse else 8)
    if sparse:
        assert cache.group_caches["a"].pending[0][1].raw_token_ids == tuple(tokens)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("medium", ["CPU", "SSD"])
def test_untagged_external_events_do_not_invalidate_known_gpu_groups(medium, layout, stored, install_metadata):
    cache = KVCacheStore()
    install_metadata(cache, "a", layout())
    events = [stored(), stored(1, "mamba", 8, hashes=["r16"])]
    _apply(cache, events)
    assert _estimate(cache, tokens=list(range(17)), unit=2).estimated_hit_tokens == 16
    counts = cache.per_replica_block_counts()
    external = [
        ["BlockStored", ["r16"], None, list(range(16)), 16, None, medium],
        ["BlockRemoved", ["r16"], medium],
    ]
    _apply(cache, external)
    assert _estimate(cache, tokens=list(range(17)), unit=2).estimated_hit_tokens == 16
    assert cache.per_replica_block_counts() == counts
