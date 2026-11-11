"""Behavioral tests for event-driven hybrid cache residency and routing."""

from unittest.mock import patch

import pytest

from uni_agent.agent_aware_router.store.cache_registry import build_cache_registry
from uni_agent.agent_aware_router.store.group_cache import ReplicaCacheIndex
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.fixture
def index(layout):
    def make_index(raw=None, **kwargs):
        registry = build_cache_registry(raw or layout()).registry
        assert registry is not None
        return ReplicaCacheIndex(registry, "a", **kwargs)

    return make_index


def estimate(cache, n=17, tokens=None):
    tokens = list(range(n)) if tokens is None else tokens
    hashes, _ = get_prefix_hashes_incremental(tokens, cache.registry.local_hash_unit, 0, 0)
    return cache.estimate([str(h) for h in hashes], len(tokens))


def dispatch(cache, n=17):
    hashes, _ = get_prefix_hashes_incremental(list(range(n)), cache.registry.local_hash_unit, 0, 0)
    return cache.record_dispatch([str(h) for h in hashes])


@pytest.fixture
def seed(parse, stored):
    def make_seed(cache, endpoints=("r8", "r16")):
        cache.apply(parse([stored(), stored(1, "mamba", 8, hashes=list(endpoints))]))

    return make_seed


def test_sparse_state_waits_for_cross_group_identity_without_inventing_endpoints(index, parse, stored):
    cache = index()
    with patch("uni_agent.agent_aware_router.store.group_cache.compute_hash") as compute:
        cache.apply(parse([stored(1, "mamba", 8, hashes=["r16"])]))
        compute.assert_not_called()
    with patch.object(cache, "_store", wraps=cache._store) as store_event:
        cache.apply([])
        store_event.assert_not_called()
    assert estimate(cache).status == "ready"
    assert estimate(cache).estimated_hit_tokens == 0
    assert cache.pending and not cache.entries
    assert len(cache.retained_hashes) == 1
    cache.apply(parse([stored()]))
    assert not cache.pending
    assert len(cache.retained_hashes) == 5
    assert len(cache.anchors) == 4  # r4/r8/r12/r16; not every local unit.
    assert len(cache.by_group[1]) == 1
    result = estimate(cache)
    assert result.status == "ready"
    assert result.estimated_hit_tokens == 16
    assert result.hit_rate == 16 / 17
    assert estimate(cache, 13).estimated_hit_tokens == 0


def test_pending_parent_chain_recovers_in_dependency_order(index, layout, parse, stored):
    cache = index(layout((("full_attention", 4),)))
    cache.apply(
        parse(
            [
                stored(parent="r8", hashes=["r12"], tokens=list(range(8, 12))),
                stored(parent="r4", hashes=["r8"], tokens=list(range(4, 8))),
            ]
        )
    )
    assert len(cache.pending) == 2
    cache.apply(parse([stored(hashes=["r4"], tokens=list(range(4)))]))
    assert not cache.pending
    assert estimate(cache, 13).estimated_hit_tokens == 12


def test_removing_unknown_sparse_hash_recovers_remaining_checkpoint(index, parse, removed, stored):
    cache = index()
    cache.apply(parse([stored()]))
    cache.apply(parse([stored(1, "mamba", 4, hashes=["unknown", "r16"])]))
    assert cache.pending
    cache.apply(parse([removed(1, ["unknown"])]))
    assert not cache.pending
    assert estimate(cache).estimated_hit_tokens == 16


def test_group_remove_keeps_full_chain_and_later_mamba_checkpoint(index, parse, removed, seed):
    cache = index()
    seed(cache)
    cache.apply(parse([removed(1, ["r8", "r8"])]))
    assert estimate(cache).estimated_hit_tokens == 16
    assert cache.references["r8"] == 1
    assert (0, "r8") in cache.retained_hashes and (1, "r8") not in cache.retained_hashes
    cache.apply(parse([removed(0, ["r12"])]))
    assert estimate(cache).estimated_hit_tokens == 0
    assert "r16" in cache.anchors


def test_mamba_groups_need_a_common_checkpoint_not_minimum_of_maxima(index, layout, parse, stored):
    cache = index(layout((("full_attention", 4), ("mamba", 4), ("mamba", 4))))
    cache.apply(
        parse([stored(), stored(1, "mamba", 4, hashes=["r8", "r16"]), stored(2, "mamba", 4, hashes=["r8", "r12"])])
    )
    assert estimate(cache).estimated_hit_tokens == 8


def test_partial_mamba_uses_actual_event_size(index, parse, stored, removed):
    cache = index()
    cache.apply(parse([stored()]))
    partial = stored(1, "mamba", 2, tokens=[4, 5], hashes=["r6"], parent="r4")
    cache.apply(parse([partial, partial]))
    assert estimate(cache, 7).estimated_hit_tokens == 0
    cache.apply(parse([stored(block=2, tokens=[4, 5], hashes=["r6"], parent="r4")]))
    assert estimate(cache, 7).estimated_hit_tokens == 6
    cache.apply(parse([removed(0, ["r4"])]))
    assert estimate(cache, 7).estimated_hit_tokens == 0


def test_sliding_requires_only_window_minus_one_tokens(index, layout, parse, removed, stored):
    cache = index(layout((("full_attention", 4), ("sliding_window", 4)), window=9))
    cache.apply(
        parse([stored(), stored(1, "sliding_window", 4, hashes=["r12", "r16"], kv_cache_spec_sliding_window=9)])
    )
    assert estimate(cache).estimated_hit_tokens == 16
    cache.apply(parse([removed(1, ["r12"])]))
    assert estimate(cache).estimated_hit_tokens == 0


def test_sliding_can_use_short_contiguous_prefix_at_start(index, layout, parse, stored):
    cache = index(layout((("sliding_window", 4),), window=17))
    cache.apply(parse([stored(kind="sliding_window", tokens=list(range(8)), kv_cache_spec_sliding_window=17)]))
    assert estimate(cache).estimated_hit_tokens == 8


@pytest.mark.parametrize("prefix_policy", [False, True], ids=["original-rpc", "prefix-policy-rpc"])
def test_original_and_extended_metadata_keep_ordinary_cache_hits(prefix_policy, index, layout, parse, removed, seed):
    raw = layout()
    if prefix_policy:
        for group in raw["groups"]:
            group.update(prefix_cacheable=True, prefix_replay_tokens=0)
    cache = index(raw)
    seed(cache)
    result = estimate(cache)
    assert result.estimated_hit_tokens == result.estimated_saved_tokens == 16
    assert result.replayed_tokens == 0
    assert result.hit_rate == result.saved_token_rate == 16 / 17
    cache.apply(parse([removed(1, ["r16"])]))
    assert estimate(cache).estimated_hit_tokens == 8


@pytest.mark.parametrize("kind,policy", [("sliding_window", True), ("unknown", True), ("unknown", False)])
def test_non_cacheable_auxiliary_group_does_not_require_events_or_add_residency(
    kind, policy, index, layout, parse, removed, stored
):
    raw = layout((("full_attention", 4), (kind, 4)))
    if policy:
        raw["groups"][1].update(prefix_cacheable=False, prefix_replay_tokens=0)
    cache = index(raw)
    cache.apply(parse([stored()]))
    assert estimate(cache).estimated_hit_tokens == 16
    before = set(cache.entries)
    cache.apply(parse([stored(1, kind, 4, kv_cache_spec_sliding_window=5), removed(1, ["r16"])]))
    assert set(cache.entries) == before
    assert not any(group == 1 for group, _ in cache.retained_hashes)
    assert estimate(cache).estimated_hit_tokens == 16


@pytest.mark.parametrize("hit,replay,expected", [(4, 8, 0), (8, 8, 0), (12, 8, 12), (16, 7, 16)])
@pytest.mark.parametrize("source", ["native", "dispatch"])
def test_bounded_replay_distinguishes_native_hits_from_saved_prefill(
    hit, replay, expected, source, index, layout, parse, stored
):
    raw = layout((("full_attention", 4), ("sliding_window_mla", 4)), window=replay)
    for group in raw["groups"]:
        group.update(prefix_cacheable=True, prefix_replay_tokens=0)
    raw["groups"][1].update(prefix_cacheable=False, prefix_replay_tokens=replay)
    cache = index(raw)
    if source == "native":
        cache.apply(parse([stored(tokens=list(range(hit)))]))
    else:
        dispatch(cache, hit)
    result = estimate(cache)
    assert result.estimated_hit_tokens == expected
    assert result.replayed_tokens == (replay if expected else 0)
    assert result.estimated_saved_tokens == max(0, hit - replay)
    assert result.hit_rate == expected / 17
    assert result.saved_token_rate == max(0, hit - replay) / 17
    assert result.provisional == bool(expected and source == "dispatch")
    cache.apply(parse([dict(type="AllBlocksCleared")]))
    cleared = estimate(cache)
    assert cleared.estimated_hit_tokens == cleared.replayed_tokens == cleared.estimated_saved_tokens == 0


def test_chunked_rechecks_chunk_when_full_reduces_candidate(index, layout, parse, removed, stored):
    cache = index(layout((("full_attention", 4), ("chunked_local_attention", 4)), unit=4, chunk=8))
    cache.apply(parse([stored(), removed(0, ["r16"]), stored(1, "chunked_local_attention", 4, hashes=["r12"])]))
    assert estimate(cache).estimated_hit_tokens == 12
    cache.apply(parse([removed(1, ["r12"])]))
    assert estimate(cache).estimated_hit_tokens == 8  # New chunk needs no older KV.
    cache.apply(parse([removed(0, ["r8"])]))
    assert estimate(cache).estimated_hit_tokens == 0  # At 4, chunk 0's block is needed.


def test_exact_prompt_boundary_retains_one_token_for_recompute(index, seed):
    cache = index()
    seed(cache)
    result = estimate(cache, 16)
    assert result.lookup_limit_tokens == 14
    assert result.estimated_hit_tokens == 8
    assert result.hit_rate == 0.5
    assert estimate(cache, 0).status == "empty"


@pytest.mark.parametrize("copies", [1, 2])
def test_unknown_parent_is_not_rebased_to_root_and_remove_cancels_pending(copies, index, parse, removed, stored):
    cache = index()
    tail = stored(1, "mamba", 8, tokens=list(range(8, 16)), hashes=["r16"], parent="r8")
    cache.apply(parse([tail] * copies + [removed(1, ["r16"])]))
    assert bool(cache.pending) == (copies == 2)
    cache.apply(parse([stored()]))
    assert not cache.pending
    assert estimate(cache).estimated_hit_tokens == (16 if copies == 2 else 0)
    cache.apply(parse([removed(1, ["r16"])]))
    assert estimate(cache).estimated_hit_tokens == 0
    cache.apply(parse([tail]))
    assert estimate(cache).estimated_hit_tokens == 16


def test_duplicate_copies_survive_one_removal_and_clear_drops_counts(index, parse, removed, seed, stored):
    cache = index()
    seed(cache)
    seed(cache)
    assert len(cache.entries) == 6
    assert len(cache.retained_hashes) == 6
    assert cache.references["r16"] == 2
    cache.apply(parse([removed(0, ["r16"]), removed(1, ["r16"])]))
    assert estimate(cache).estimated_hit_tokens == 16
    cache.apply(parse([removed(1, ["r16"])]))
    assert estimate(cache).estimated_hit_tokens == 8
    assert (0, "r16") in cache.retained_hashes and (1, "r16") not in cache.retained_hashes
    cache.apply(parse([removed(1, ["r8", "r8"])]))
    assert estimate(cache).estimated_hit_tokens == 0
    cache.apply(parse([stored(parent="missing", tokens=list(range(4)), hashes=["pending4"])]))
    assert cache.pending
    cache.apply(parse([dict(type="AllBlocksCleared")]))
    assert not cache.anchors and not cache.entries and not cache.pending
    assert not cache.retained_hashes
    assert estimate(cache).estimated_hit_tokens == 0
    cache.apply(parse([stored(parent="r16", tokens=list(range(16, 20)), hashes=["r20"])]))
    assert estimate(cache).status == "ready"
    assert estimate(cache).estimated_hit_tokens == 0
    assert cache.pending and not cache.entries
    seed(cache)
    assert estimate(cache).estimated_hit_tokens == 16
    cache.apply(parse([removed(1, ["r16"])]))
    assert estimate(cache).estimated_hit_tokens == 8


@pytest.mark.parametrize("modifier", [dict(medium="CPU"), dict(locality="REMOTE"), dict(ownership="offload")])
def test_other_storage_domains_cannot_delete_local_gpu_entries(modifier, index, parse, removed, seed, stored):
    cache = index()
    seed(cache)
    event = removed(1, ["r16"])
    event.update(modifier)
    cache.apply(parse([stored(hashes=["excluded"], tokens=list(range(4)), **modifier), event]))
    assert estimate(cache).estimated_hit_tokens == 16
    assert len(cache.retained_hashes) == 6


@pytest.mark.parametrize("identity", [dict(lora_id=1), dict(lora_name="adapter"), dict(extra_keys=[["salt"]])])
def test_non_text_cache_identity_is_not_aliased_to_plain_tokens(identity, index, layout, parse, stored):
    cache = index(layout((("full_attention", 4),)))
    cache.apply(parse([stored(**identity)]))
    assert not cache.entries and not cache.anchors and not cache.retained_hashes


def test_shared_prefix_does_not_cross_replica_residency(index, parse, removed, seed, stored):
    a = index()
    b = ReplicaCacheIndex(a.registry, "b")
    seed(a)
    b.apply(parse([stored()]))
    assert estimate(a).estimated_hit_tokens == 16
    assert estimate(b).estimated_hit_tokens == 0
    b.apply(parse([stored(1, "mamba", 8, hashes=["r16"])]))
    a.apply(parse([removed(1, ["r8", "r16"])]))
    assert estimate(b).estimated_hit_tokens == 16
    assert estimate(a).estimated_hit_tokens == 0


@pytest.mark.parametrize("unknown", ["group", "kind"])
def test_unknown_required_group_disables_estimates(unknown, index, parse, removed, seed, stored):
    cache = index()
    seed(cache)
    event = stored(group=77) if unknown == "group" else stored(kind="unknown")
    cache.apply(parse([event]))
    assert estimate(cache).status == "unsupported"
    assert estimate(cache).estimated_hit_tokens == 0
    cache.apply(parse([removed(1, ["r16"])]))
    assert estimate(cache).status == "unsupported"


def test_uninterpretable_events_preserve_hits_and_later_updates(index, parse, removed, seed, stored):
    cache = index()
    seed(cache)
    for event in (
        stored(medium=None),
        {**stored(), "token_ids": None},
        stored(kind="mla_attention"),
        stored(1, "mamba", 8, tokens=list(range(100, 116)), hashes=["r16"]),
    ):
        cache.apply(parse([event]))
        assert estimate(cache).estimated_hit_tokens == 16
    cache.apply(parse([removed(1, ["r16"])]))
    assert estimate(cache).estimated_hit_tokens == 8


def test_pending_limits_preserve_hits_without_reviving_expired_events(index, parse, seed, stored):
    cache = index(pending_limit=1, pending_ttl=60)
    seed(cache)
    with patch("uni_agent.agent_aware_router.store.group_cache.time.monotonic", return_value=0):
        cache.apply(
            parse(
                [
                    stored(parent="r20", block=2, hashes=["r22"], tokens=[20, 21]),
                    stored(parent="r24", block=2, hashes=["r26"], tokens=[24, 25]),
                ]
            )
        )
        assert len(cache.pending) == 1
        assert estimate(cache).estimated_hit_tokens == 16
    with patch("uni_agent.agent_aware_router.store.group_cache.time.monotonic", return_value=61):
        assert estimate(cache).estimated_hit_tokens == 16
        assert not cache.pending
        cache.apply(
            parse(
                [
                    stored(tokens=list(range(16, 24)), hashes=["r20", "r24"], parent="r16"),
                    stored(1, "mamba", 2, hashes=["r26"], tokens=[24, 25], parent="r24"),
                ]
            )
        )
        assert estimate(cache, 27).estimated_hit_tokens == 16


def test_mapping_gc_preserves_live_group_references(index, parse, removed, seed):
    cache = index(history_limit=0)
    seed(cache)
    cache.apply(parse([removed(1, ["r8", "r16"])]))
    assert set(cache.anchors) == {"r4", "r8", "r12", "r16"}
    cache.apply(parse([removed(0, ["r4", "r8", "r12", "r16"])]))
    assert not cache.anchors


def test_mamba_all_uses_scheduler_unit_and_only_actual_states(index, layout, parse, removed, stored):
    cache = index(layout(groups=(("full_attention", 8), ("mamba", 8)), unit=None, mode="all"))
    assert cache.registry.local_hash_unit == 8
    assert cache.registry.mamba_cache_mode == "all"
    cache.apply(parse([stored(block=8), stored(1, "mamba", 8)]))
    assert estimate(cache, 9).estimated_hit_tokens == 8
    cache.apply(parse([removed(1, ["r8"])]))
    assert estimate(cache).estimated_hit_tokens == 16
    assert estimate(cache, 9).estimated_hit_tokens == 0
    cache.apply(parse([removed(1, ["r16"])]))
    assert estimate(cache).estimated_hit_tokens == 0
    assert len(cache.by_group[0]) == 2


def test_empty_mamba_hashes_do_not_claim_root_checkpoint(index, parse, stored):
    cache = index()
    cache.apply(parse([stored(1, "mamba", 8, tokens=list(range(8)), hashes=[]), stored(tokens=list(range(8)))]))
    assert estimate(cache, 9).estimated_hit_tokens == 0
    cache.apply(
        parse(
            [
                stored(1, "mamba", 8, tokens=list(range(8, 16)), hashes=["r16"], parent="r8"),
                stored(tokens=list(range(8, 16)), hashes=["r12", "r16"], parent="r8"),
            ]
        )
    )
    assert estimate(cache).estimated_hit_tokens == 16


def test_clear_inside_received_batch_discards_previous_epoch(index, parse, seed, stored):
    cache = index()
    seed(cache)
    cache.apply(
        parse(
            [
                stored(tokens=list(range(16, 20)), hashes=["r20"], parent="r16"),
                {"type": "AllBlocksCleared"},
                stored(tokens=list(range(4))),
            ]
        )
    )
    assert cache.epoch == 1
    assert set(cache.anchors) == {"r4"}
    assert not cache.pending
    assert estimate(cache).estimated_hit_tokens == 0


@pytest.mark.parametrize("kind,block", [("full_attention", 4), ("mamba", 8), ("sliding_window", 4)])
def test_dispatch_predictions_are_confirmed_independently_per_group(kind, block, index, layout, parse, stored):
    cache = index(layout((("full_attention", 4), (kind, block)), window=9))
    assert dispatch(cache) == 4 + 16 // block
    assert dispatch(cache) == 0
    assert not cache.anchors and not cache.entries and not cache.retained_hashes
    predicted = estimate(cache)
    assert predicted.estimated_hit_tokens == 16 and predicted.provisional

    cache.apply(parse([stored()]))
    assert not cache.dispatch_hints[0]
    assert len(cache.dispatch_hints[1]) == 16 // block
    assert len(cache.retained_hashes) == 4
    assert estimate(cache) == predicted  # Native Full must not confirm the other group.

    cache.apply(parse([stored(1, kind, block, kv_cache_spec_sliding_window=9)]))
    assert not cache.dispatch_hints[1]
    confirmed = estimate(cache)
    assert confirmed.estimated_hit_tokens == 16 and not confirmed.provisional
    assert len(cache.retained_hashes) == 4 + 16 // block


def test_dispatch_removal_and_clear_preserve_group_boundaries(index, layout, parse, stored, removed):
    cache = index(layout((("full_attention", 4), ("full_attention", 4))))
    dispatch(cache)
    cache.apply(parse([stored(tokens=list(range(4)))]))
    other_hints = dict(cache.dispatch_hints[0])
    cache.apply(parse([removed(1, ["r4"])]))
    assert len(cache.dispatch_hints[1]) == 3  # Known identities remove only that endpoint.
    assert dict(cache.dispatch_hints[0]) == other_hints
    assert len(cache.retained_hashes) == 1

    remaining_hints = dict(cache.dispatch_hints[1])
    cache.apply(parse([removed(0, ["unknown-identity"])]))
    assert not cache.dispatch_hints[0]  # Unknown identities conservatively invalidate this group.
    assert dict(cache.dispatch_hints[1]) == remaining_hints
    assert len(cache.retained_hashes) == 1
    cache.apply(parse([{"type": "AllBlocksCleared"}]))
    assert not cache.dispatch_hints and not cache.entries and not cache.retained_hashes
    assert estimate(cache).estimated_hit_tokens == 0


def test_confirmed_sliding_window_does_not_depend_on_older_dispatch_predictions(index, layout, parse, stored):
    cache = index(layout((("full_attention", 4), ("sliding_window", 4)), window=9))
    dispatch(cache)
    cache.apply(
        parse([stored(), stored(1, "sliding_window", 4, hashes=["r12", "r16"], kv_cache_spec_sliding_window=9)])
    )
    assert len(cache.dispatch_hints[1]) == 2  # Earlier speculative blocks remain unused by this window.
    result = estimate(cache)
    assert result.estimated_hit_tokens == 16 and not result.provisional


def test_dispatch_predictions_use_only_expressible_boundaries_and_bounded_history(index, layout):
    cache = index(layout((("full_attention", 4), ("mamba", 6)), unit=None, mode="all"), history_limit=1)
    assert cache.registry.local_hash_unit == 12
    assert dispatch(cache, 25) == 4
    assert [list(hints.values()) for hints in cache.dispatch_hints.values()] == [[24], [24]]
    assert not cache.anchors and not cache.entries and not cache.retained_hashes


def test_lcm_full_matches_expressible_endpoints_without_skipping_a_gap(index, layout, parse, stored, removed):
    cache = index(layout((("full_attention", 4), ("mamba", 6)), unit=None, mode="all"))
    assert dispatch(cache, 25) == 4
    predicted = estimate(cache, 25)
    assert predicted.estimated_hit_tokens == 24 and predicted.provisional
    cache.apply(parse([stored(block=12, tokens=list(range(24))), stored(1, "mamba", 12, tokens=list(range(24)))]))
    confirmed = estimate(cache, 25)
    assert confirmed.estimated_hit_tokens == 24 and not confirmed.provisional
    cache.apply(parse([removed(0, ["r12"])]))
    assert estimate(cache, 25).estimated_hit_tokens == 0  # r24 cannot bridge the missing canonical endpoint.
