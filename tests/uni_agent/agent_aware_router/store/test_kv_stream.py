"""Native stream ordering, cache boundaries, and conservative Full inference."""

import msgpack
import pytest

from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.collectors.transport.base import TransportMessage
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def _apply(cache, events, node="a", sequence=None):
    frame = TransportMessage(msgpack.packb([0, events]), sequence)
    return cache.apply_update(VLLMKVParser().parse(frame, node))


def _old_stored(remote="cached", tokens=None, block=4):
    return ["BlockStored", [remote], None, list(range(block)) if tokens is None else tokens, block, None, None]


def _estimate(cache, node="a", tokens=None, unit=4):
    tokens = list(range(5)) if tokens is None else tokens
    hashes, _ = get_prefix_hashes_incremental(tokens, unit, 0, 0)
    return cache.estimate_prefixes([node], [str(value) for value in hashes], len(tokens))[node]


def test_stream_gap_requires_clear_before_stored_can_be_applied():
    cache = KVCacheStore()
    event = _old_stored()
    assert _apply(cache, [event], sequence=0)
    assert not _apply(cache, [event], sequence=2)
    assert cache._stream_errors == {"a": "kv_sequence_gap:1:2"}
    assert not cache.per_replica_block_counts()
    assert not _apply(cache, [event], sequence=3)
    assert _apply(cache, [["AllBlocksCleared"], event], sequence=4)
    assert cache.per_replica_block_counts() == {"a": 1}
    assert _estimate(cache).estimated_hit_tokens == 4


def test_clear_discards_stores_before_the_boundary_and_isolates_nodes():
    cache = KVCacheStore()
    before = _old_stored("old")
    after = _old_stored("new", list(range(4, 8)))
    assert _apply(cache, [before], "b")
    assert _apply(cache, [before, ["AllBlocksCleared"], after])
    assert _estimate(cache).estimated_hit_tokens == 0
    assert _estimate(cache, tokens=list(range(4, 9))).estimated_hit_tokens == 4
    assert _estimate(cache, "b").estimated_hit_tokens == 4
    assert cache.per_replica_block_counts() == {"a": 1, "b": 1}


def test_old_clear_cannot_remove_a_stream_gap_barrier():
    cache = KVCacheStore()
    assert _apply(cache, [], sequence=5)
    assert not _apply(cache, [], sequence=7)
    assert not _apply(cache, [["AllBlocksCleared"]], sequence=4)
    assert "a" in cache._stream_errors
    assert _apply(cache, [["AllBlocksCleared"]], sequence=8)
    assert "a" not in cache._stream_errors


def test_duplicate_malformed_payload_cannot_invalidate_healthy_residency():
    cache = KVCacheStore()
    assert _apply(cache, [_old_stored()], sequence=5)
    broken = VLLMKVParser().parse(TransportMessage(b"\xc1", sequence=5), "a")
    assert broken.stream_error == "invalid_kv_event_payload"
    assert not cache.apply_update(broken)
    assert _estimate(cache).estimated_hit_tokens == 4
    assert _apply(cache, [["BlockRemoved", ["cached"], None]], sequence=6)
    assert _estimate(cache).estimated_hit_tokens == 0
    assert not cache._stream_errors


def test_clear_in_first_batch_after_a_gap_restores_only_new_residency():
    cache = KVCacheStore()
    assert _apply(cache, [_old_stored()], sequence=5)
    after = _old_stored("new", list(range(4, 8)))
    assert _apply(cache, [["AllBlocksCleared"], after], sequence=7)
    assert not cache._stream_errors
    assert _estimate(cache).estimated_hit_tokens == 0
    assert _estimate(cache, tokens=list(range(4, 9))).estimated_hit_tokens == 4
    assert cache.per_replica_block_counts() == {"a": 1}


def test_new_stream_resets_only_its_replica_and_accepts_nonzero_baseline():
    cache = KVCacheStore()
    assert _apply(cache, [_old_stored()], "a", sequence=100)
    assert _apply(cache, [_old_stored()], "b", sequence=110)
    reset = VLLMKVParser().parse(TransportMessage(b"", status="new_stream"), "a")
    assert cache.apply_update(reset)
    assert cache.per_replica_block_counts() == {"b": 1}
    assert _apply(cache, [_old_stored("new", list(range(4, 8)))], sequence=5)
    assert _estimate(cache).estimated_hit_tokens == 0
    assert _estimate(cache, tokens=list(range(4, 9))).estimated_hit_tokens == 4
    assert _estimate(cache, "b").estimated_hit_tokens == 4
    assert not cache._stream_errors


@pytest.mark.parametrize("status", ["no_continuity", "unreachable"])
def test_protocol_failure_requires_clear_before_unsequenced_events_can_restore_cache(status):
    cache = KVCacheStore()
    event = _old_stored()
    assert _apply(cache, [event])
    failure = VLLMKVParser().parse(TransportMessage(b"", status=status), "a")
    assert not cache.apply_update(failure)
    assert not cache.per_replica_block_counts()
    assert not _apply(cache, [event])
    assert _apply(cache, [["AllBlocksCleared"], event])
    assert _estimate(cache).estimated_hit_tokens == 4


@pytest.mark.parametrize("metadata_available", [False, True], ids=["inferred-full", "native-full"])
def test_duplicate_replay_does_not_leave_residency_after_removal(metadata_available, layout):
    cache = KVCacheStore()
    if metadata_available:
        cache.configure_registry(layout((("full_attention", 4),), unit=None))
    assert _apply(cache, [_old_stored()], sequence=5)
    assert not _apply(cache, [_old_stored()], sequence=5)
    assert cache.per_replica_block_counts() == {"a": 1}
    assert _apply(cache, [["BlockRemoved", ["cached"], None]], sequence=6)
    assert cache.per_replica_block_counts().get("a", 0) == 0
    assert _apply(cache, [_old_stored()], sequence=7)
    # A queued live removal already consumed during replay must not erase a
    # later store of the same identity.
    assert not _apply(cache, [["BlockRemoved", ["cached"], None]], sequence=6)
    assert cache.per_replica_block_counts() == {"a": 1}
    assert _estimate(cache).estimated_hit_tokens == 4


@pytest.mark.parametrize("metadata_available", [False, True], ids=["inferred-full", "native-full"])
def test_malformed_sequenced_batch_invalidates_only_its_replica(metadata_available, layout):
    cache = KVCacheStore()
    if metadata_available:
        cache.configure_registry(layout((("full_attention", 4),), unit=None))
    for node in ("a", "b"):
        assert _apply(cache, [_old_stored()], node, sequence=0)
    broken = VLLMKVParser().parse(TransportMessage(b"\xc1", 1), "a")
    assert broken.stream_error == "invalid_kv_event_payload"
    assert not cache.apply_update(broken)
    assert cache.per_replica_block_counts() == {"b": 1}
    assert not _apply(cache, [_old_stored()], sequence=2)
    assert _apply(cache, [["AllBlocksCleared"], _old_stored()], sequence=3)
    assert cache.per_replica_block_counts() == {"a": 1, "b": 1}


@pytest.mark.parametrize("status", ["not_ready", "unsupported", "error"])
def test_unavailable_metadata_can_infer_only_untagged_single_full(status):
    cache = KVCacheStore()
    cache.configure_registry({"status": status})
    assert _apply(cache, [_old_stored()])
    estimate = _estimate(cache)
    assert estimate.estimated_hit_tokens == 4
    assert cache.registry_state.reason == "legacy_single_full_assumption"


@pytest.mark.parametrize("metadata_state", ["invalid", "disabled", "multiple-groups"])
def test_explicit_layout_never_silently_falls_back_to_single_full(metadata_state, layout):
    cache = KVCacheStore()
    metadata = layout()
    if metadata_state == "invalid":
        metadata["groups"] = []
    elif metadata_state == "disabled":
        metadata["cache_config"]["enable_prefix_caching"] = False
    cache.configure_registry(metadata)
    _apply(cache, [_old_stored()])
    assert _estimate(cache).estimated_hit_tokens == 0
    assert not any(cache.per_replica_block_counts().values())


@pytest.mark.parametrize("evidence", ["tagged-group", "different-size", "partial-size"])
def test_ambiguous_event_layout_revokes_inferred_full_cache(evidence, stored):
    cache = KVCacheStore()
    assert _apply(cache, [_old_stored()])
    assert _estimate(cache).estimated_hit_tokens == 4
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


@pytest.mark.parametrize("sparse", [False, True], ids=["unindexed-tail", "sparse-range"])
def test_old_full_tail_adapter_does_not_trim_sparse_token_ranges(sparse, layout):
    cache = KVCacheStore()
    cache.configure_registry(layout((("full_attention", 4),), unit=None))
    tokens = list(range(16 if sparse else 9))
    event = ["BlockStored", ["r8", "r12", "r16"] if sparse else ["r4", "r8"], None, tokens, 4, None, None]
    update = VLLMKVParser().parse(TransportMessage(msgpack.packb([0, [event]])), "a")
    assert update.events[0].raw_token_ids == tuple(tokens)
    assert cache.apply_update(update)
    assert _estimate(cache, tokens=list(range(17))).estimated_hit_tokens == (0 if sparse else 8)
    if sparse:
        assert cache.group_caches["a"].pending[0][1].raw_token_ids == tuple(tokens)


def test_inferred_hash_schema_changes_when_native_layout_arrives(layout):
    cache = KVCacheStore()
    initial = cache.hash_schema()
    _apply(cache, [_old_stored()])
    inferred = cache.hash_schema()
    assert inferred[0] > initial[0] and inferred[1] == 4
    cache.configure_registry(layout((("full_attention", 8),), unit=None))
    native = cache.hash_schema()
    assert native[0] > inferred[0] and native[1] == 8
    assert cache.per_replica_block_counts() == {}


@pytest.mark.parametrize("medium", ["CPU", "SSD"])
def test_untagged_external_events_do_not_invalidate_known_gpu_groups(medium, layout, stored):
    cache = KVCacheStore()
    cache.configure_registry(layout())
    events = [stored(), stored(1, "mamba", 8, hashes=["r16"])]
    assert _apply(cache, events)
    assert _estimate(cache, tokens=list(range(17)), unit=2).estimated_hit_tokens == 16
    counts = cache.per_replica_block_counts()
    external = [
        ["BlockStored", ["r16"], None, list(range(16)), 16, None, medium],
        ["BlockRemoved", ["r16"], medium],
    ]
    assert _apply(cache, external)
    assert _estimate(cache, tokens=list(range(17)), unit=2).estimated_hit_tokens == 16
    assert cache.per_replica_block_counts() == counts
