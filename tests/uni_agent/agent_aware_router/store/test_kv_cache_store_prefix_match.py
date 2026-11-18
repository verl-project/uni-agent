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

"""Single-Full layouts use the same event index and token estimates as hybrid."""

import msgpack
import pytest

from uni_agent.agent_aware_router.collectors.parse.base import KVCacheUpdate
from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental

pytestmark = [pytest.mark.level0, pytest.mark.cpu]


def _hashes(tokens, unit=2):
    hashes, _ = get_prefix_hashes_incremental(tokens, unit, 0, 0)
    return [str(value) for value in hashes]


def test_chained_hashes_are_cumulative() -> None:
    """Perturbing block i leaves hash[0..i-1] stable, changes hash[i..].

    This is the invariant that makes the chain-break scan correct: each
    hash embeds all prior blocks, so caching block i implies caching 1..i-1.
    """
    base = [1, 2, 3, 4, 5, 6]  # three full blocks of size 2
    base_h = _hashes(base)

    # Perturb block index 1 (tokens [3,4] -> [3,99]).
    perturbed = [1, 2, 3, 99, 5, 6]
    pert_h = _hashes(perturbed)

    # Blocks before the perturbation are unchanged.
    assert pert_h[0] == base_h[0]
    # The perturbed block and every block after it change.
    assert pert_h[1] != base_h[1]
    assert pert_h[2] != base_h[2]


def test_max_contiguous_match_per_replica(layout, stored, parse, install_metadata):
    store = KVCacheStore()
    install_metadata(store, "a", layout((("full_attention", 2),), unit=None))
    install_metadata(store, "b", layout((("full_attention", 2),), unit=None))
    install_metadata(store, "c", layout((("full_attention", 2),), unit=None))
    for node, count in [("a", 8), ("b", 4), ("c", 6)]:
        store.apply_update(KVCacheUpdate(node, events=parse([stored(block=2, tokens=list(range(count)))])))
    result = store.estimate_prefixes(["a", "b", "c"], _hashes(list(range(9))), 9)
    assert {node: estimate.estimated_hit_tokens for node, estimate in result.items()} == {"a": 8, "b": 4, "c": 6}
    assert result["a"].hit_rate == 8 / 9


def test_chain_break_stops_at_first_missing_block(layout, stored, removed, parse, install_metadata):
    store = KVCacheStore()
    install_metadata(store, "a", layout((("full_attention", 2),), unit=None))
    install_metadata(store, "b", layout((("full_attention", 2),), unit=None))
    for node in ("a", "b"):
        store.apply_update(KVCacheUpdate(node, events=parse([stored(block=2, tokens=list(range(10)))])))
    store.apply_update(KVCacheUpdate("a", events=parse([removed(0, ["r8"])])))
    result = store.estimate_prefixes(["a", "b"], _hashes(list(range(11))), 11)
    assert result["a"].estimated_hit_tokens == 6
    assert result["b"].estimated_hit_tokens == 10


def test_unrelated_prefix_is_not_credited(layout, stored, parse, install_metadata):
    store = KVCacheStore()
    install_metadata(store, "a", layout((("full_attention", 2),), unit=None))
    events = parse([stored(block=2, tokens=[11, 22, 33, 44])])
    store.apply_update(KVCacheUpdate("a", events=events))
    assert store.estimate_prefixes(["a"], _hashes(list(range(9))), 9)["a"].estimated_hit_tokens == 0


@pytest.mark.parametrize("query_tokens,expected", [(8, 4), (9, 8)])
def test_old_untagged_and_modern_group_events_share_full_estimates(query_tokens, expected, layout, install_metadata):
    metadata = layout((("full_attention", 4),), unit=None)
    metadata["groups"][0]["group_idx"] = 3
    old = ["BlockStored", [11, 12], None, list(range(8)), 4, None, None]
    modern = dict(
        type="BlockStored",
        block_hashes=[11, 12],
        parent_block_hash=None,
        token_ids=list(range(8)),
        block_size=4,
        medium="GPU",
        group_idx=3,
        kv_cache_spec_kind="full_attention",
    )
    stores = [KVCacheStore(), KVCacheStore()]
    parser = VLLMKVParser()
    for store in stores:
        install_metadata(store, "a", metadata)
    hashes = _hashes(list(range(query_tokens)), 4)
    batches = [
        ([old], [modern], expected),
        ([["BlockRemoved", [12], None]], [dict(type="BlockRemoved", block_hashes=[12], medium="GPU", group_idx=3)], 4),
        ([["AllBlocksCleared"]], [dict(type="AllBlocksCleared")], 0),
    ]
    for old_events, modern_events, hit in batches:
        for store, events in zip(stores, [old_events, modern_events], strict=True):
            store.apply_update(parser.parse(msgpack.packb([0, events]), "a"))
        estimates = [store.estimate_prefixes(["a"], hashes, query_tokens)["a"] for store in stores]
        assert estimates[0] == estimates[1]
        assert estimates[0].estimated_hit_tokens == hit
        assert stores[0].per_replica_block_counts() == stores[1].per_replica_block_counts()
