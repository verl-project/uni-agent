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

"""Dispatch-time group references, private reservations and acquire-time release snapshots."""

from __future__ import annotations

import pytest

from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.types.kv_cache import GroupReservation

pytestmark = [pytest.mark.level0, pytest.mark.cpu]

PREFIX = ["h1", "h2", "h3", "h4", "h5"]


def _store() -> KVCacheStore:
    """A fresh non-singleton store."""
    return KVCacheStore()


def _pin(store, request, node, hashes):
    store.pin_request(
        request,
        node,
        hashes,
        store.dispatch_schema(),
        reservations=(GroupReservation(0, tuple(hashes)),),
        predict=False,
    )


def test_held_set_ref_counts_shared_blocks():
    store = _store()
    _pin(store, "r1", "s0", PREFIX)
    _pin(store, "r2", "s0", PREFIX[:3])
    assert store.inflight_block_count("s0") == 5
    store.unpin_request("r1")
    assert store.inflight_block_count("s0") == 3
    store.unpin_request("r2")
    assert store.inflight_block_count("s0") == 0


def test_ref_table_clear_isolates_replicas_and_late_release():
    store = _store()
    _pin(store, "old", "s0", PREFIX)
    _pin(store, "other", "s1", PREFIX)
    store.clear_replica("s0")
    assert store.get_marker("old") is None
    assert store.inflight_block_count("s0") == 0
    assert store.inflight_block_count("s1") == 5
    _pin(store, "new", "s0", PREFIX)
    store.unpin_request("old")
    assert store.inflight_block_count("s0") == 5
    store.unpin_request("new")
    store.unpin_request("new")
    store.unpin_request("never-pinned")
    assert store.inflight_block_count("s0") == 0
    assert store.inflight_block_count("s1") == 5


def test_reacquire_moves_exact_reservation_between_replicas():
    store = _store()
    _pin(store, "r1", "s0", PREFIX)
    _pin(store, "r1", "s1", PREFIX[:2])
    assert store.inflight_block_count("s0") == 0
    assert store.inflight_block_count("s1") == 2
    store.unpin_request("r1")
    assert store.inflight_block_count("s1") == 0


def test_marker_peek_does_not_mutate_release_snapshot():
    store = _store()
    assert store.get_marker("r1") is None
    _pin(store, "r1", "s0", PREFIX)
    peeked = store.get_marker("r1")
    peeked["hashes"].append("mutated")
    assert store.get_marker("r1")["hashes"] == PREFIX
    store.unpin_request("r1")
    assert store.get_marker("r1") is None
    assert store.inflight_block_count("s0") == 0


def _dispatch(store, prompt, request="r1", node="s0"):
    from uni_agent.agent_aware_router.collectors.parse.basic.inflight import InflightParser
    from uni_agent.agent_aware_router.collectors.transport.callback import StatisticEvent

    updates = InflightParser().parse(
        StatisticEvent("on_acquire", request_id=request, replica_id=node, prompt_ids=tuple(prompt)),
        node,
        schema=store.dispatch_schema(node),
    )
    pin = updates[0]
    return store.pin_request(request, node, pin.block_hashes, pin.schema, reservations=pin.reservations)


def _estimate(store, prompt, node="s0"):
    from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental

    hashes, _ = get_prefix_hashes_incremental(prompt, store.hash_schema()[1], 0, 0)
    return store.estimate_prefixes([node], list(map(str, hashes)), len(prompt))[node]


def test_full_dispatch_physical_stride_and_native_takeover(layout, parse, stored, removed):
    from uni_agent.agent_aware_router.types.kv_cache import KVCacheUpdate

    store = _store()
    store.configure_registry(layout((("full_attention", 4),), unit=2))
    prompt = list(range(17))
    assert _dispatch(store, prompt) == 4
    assert store.inflight_block_count("s0") == 4  # physical blocks, not eight hash units
    assert store.inflight_reserved_blocks("s0") == 4
    prediction = _estimate(store, prompt)
    assert prediction.estimated_hit_tokens == 16 and prediction.provisional
    assert store.per_replica_block_counts() == {"s0": 0}
    store.unpin_request("r1")
    assert store.inflight_reserved_blocks("s0") == 0
    assert _estimate(store, prompt) == prediction

    store.apply_update(KVCacheUpdate("s0", events=parse([stored()], "s0")))
    confirmed = _estimate(store, prompt)
    assert confirmed.estimated_hit_tokens == 16 and not confirmed.provisional
    assert store.per_replica_block_counts() == {"s0": 4}
    store.apply_update(KVCacheUpdate("s0", events=parse([removed(0, ["r8"])], "s0")))
    assert _estimate(store, prompt).estimated_hit_tokens == 4  # no stale dispatch hint resurrects r8


def test_hybrid_dispatch_uses_group_refs_private_budget_and_native_takeover(layout, parse, stored):
    from uni_agent.agent_aware_router.types.kv_cache import KVCacheUpdate

    store = _store()
    store.configure_registry(layout())
    prompt = list(range(17))
    assert _dispatch(store, prompt) == 6  # four Full endpoints and two Mamba checkpoints
    assert store.inflight_reserved_blocks("s0") == 6  # four shared Full refs + two private state blocks
    assert _estimate(store, prompt).estimated_hit_tokens == 16
    assert _estimate(store, prompt).provisional
    assert store.per_replica_block_counts() == {"s0": 0}
    _dispatch(store, prompt, "r2")
    assert store.inflight_reserved_blocks("s0") == 8  # Full shared; mutable state private per request
    store.unpin_request("r1")
    assert store.inflight_reserved_blocks("s0") == 6
    store.apply_update(KVCacheUpdate("s0", events=parse([stored()], "s0")))
    assert _estimate(store, prompt).provisional  # only Full has been confirmed
    store.apply_update(KVCacheUpdate("s0", events=parse([stored(1, "mamba", 8)], "s0")))
    assert not _estimate(store, prompt).provisional
    store.unpin_request("r2")
    assert store.inflight_block_count("s0") == 0
    assert _estimate(store, prompt).estimated_hit_tokens == 16


def test_release_preserves_new_schema_reservations(layout):
    store = _store()
    metadata = layout((("full_attention", 4),), unit=2)
    store.configure_registry(metadata)
    prompt = list(range(17))
    _dispatch(store, prompt, "old")
    old_marker = store.get_marker("old")
    store.configure_registry(metadata)
    _dispatch(store, prompt, "new")
    assert store.get_marker("new")["schema"] != old_marker["schema"]
    assert store.inflight_reserved_blocks("s0") is None  # old generation still outstanding
    store.unpin_request("old")
    store.unpin_request("old")  # duplicate release cannot decrement the new generation
    assert store.inflight_block_count("s0") == 4
    assert store.inflight_reserved_blocks("s0") == 4
    store.unpin_request("new")
    assert store.inflight_reserved_blocks("s0") == 0


@pytest.mark.parametrize("boundary", ["removed", "gap", "reset", "clear"])
def test_dispatch_predictions_respect_native_stream_boundaries(boundary, layout, parse, removed):
    from uni_agent.agent_aware_router.types.kv_cache import KVCacheUpdate

    store = _store()
    store.configure_registry(layout((("full_attention", 4),), unit=2))
    prompt = list(range(17))
    store.apply_update(KVCacheUpdate("s0", sequence=1))
    _dispatch(store, prompt)
    _dispatch(store, prompt, "other", "s1")
    if boundary == "removed":
        update = KVCacheUpdate("s0", events=parse([removed(0, ["unseen-native-identity"])], "s0"))
    elif boundary == "gap":
        update = KVCacheUpdate("s0", sequence=3)
    elif boundary == "reset":
        update = KVCacheUpdate("s0", reset_stream=True)
    else:
        update = KVCacheUpdate("s0", events=parse([{"type": "AllBlocksCleared"}], "s0"))
    store.apply_update(update)
    assert _estimate(store, prompt).estimated_hit_tokens == 0
    assert _estimate(store, prompt, "s1").provisional
    assert store.inflight_block_count("s0") == (0 if boundary == "clear" else 4)
    store.unpin_request("r1")
    assert store.inflight_block_count("s0") == 0
    assert store.inflight_block_count("s1") == 4


@pytest.mark.parametrize(
    "groups,mode,prompt_tokens,expected",
    [
        ((("full_attention", 4), ("sliding_window", 8)), "align", 17, 6),
        ((("full_attention", 4), ("chunked_local_attention", 8)), "align", 17, 6),
        ((("full_attention", 4), ("mamba", 6)), "all", 13, 5),
    ],
)
def test_group_reservations_are_independent_of_matching_hash_granularity(layout, groups, mode, prompt_tokens, expected):
    store = _store()
    store.configure_registry(layout(groups, mode=mode))
    # Mamba all matches at LCM=12, but Full4 and Mamba6 still reserve 3+2 blocks.
    prompt = list(range(prompt_tokens))
    _dispatch(store, prompt)
    assert store.inflight_reserved_blocks("s0") == expected
    _dispatch(store, prompt, "r2")
    assert store.inflight_reserved_blocks("s0") == expected
    store.unpin_request("r1")
    assert store.inflight_reserved_blocks("s0") == expected
    store.unpin_request("r2")
    assert store.inflight_reserved_blocks("s0") == 0


def test_same_hash_is_shared_within_group_but_not_across_groups(layout):
    store = _store()
    store.configure_registry(layout((("full_attention", 4), ("full_attention", 4))))
    _dispatch(store, list(range(9)))
    assert store.inflight_reserved_blocks("s0") == 4  # two blocks in each group
    _dispatch(store, list(range(9)), "r2")
    assert store.inflight_reserved_blocks("s0") == 4
    store.unpin_request("r1")
    assert store.inflight_reserved_blocks("s0") == 4


def test_non_prefix_group_private_budget_survives_complete_cache_hit(layout):
    store = _store()
    raw = layout((("full_attention", 4), ("sliding_window_mla", 4)), window=8)
    raw["groups"][1].update(prefix_cacheable=False, prefix_replay_tokens=8)
    raw["cache_reservation_profile"]["groups"][1]["max_blocks_per_request"] = 3
    store.configure_registry(raw)
    _dispatch(store, list(range(17)))
    assert store.inflight_reserved_blocks("s0") == 7
    assert store.request_block_need(17, 17) == 3  # saved prefill cannot remove private state
    _dispatch(store, list(range(17)), "r2")
    assert store.inflight_reserved_blocks("s0") == 10
    store.unpin_request("r1")
    assert store.inflight_reserved_blocks("s0") == 7
    store.unpin_request("r2")
    assert store.inflight_reserved_blocks("s0") == 0


def test_mamba_all_extra_native_budget_is_private_per_request(layout):
    store = _store()
    raw = layout((("full_attention", 4), ("mamba", 8)), mode="all")
    raw["cache_reservation_profile"]["groups"][1]["max_blocks_per_request"] += 2
    store.configure_registry(raw)
    _dispatch(store, list(range(17)))
    assert store.inflight_reserved_blocks("s0") == 8  # shared 4+2 checkpoints; private extra 2
    _dispatch(store, list(range(17)), "r2")
    assert store.inflight_reserved_blocks("s0") == 10
    store.unpin_request("r1")
    assert store.inflight_reserved_blocks("s0") == 8
    store.unpin_request("r2")
    assert store.inflight_reserved_blocks("s0") == 0
