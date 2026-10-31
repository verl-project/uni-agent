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

"""Cache generations prevent replay from restoring obsolete residency."""

from dataclasses import replace

import msgpack
import pytest

from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.types.cache_lifecycle import BRIDGE_SCHEMA_KEY, CacheLifecycleState, KVEventFrame
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def update(parser, state, sequence, events, node="a"):
    payload = msgpack.packb(
        {
            BRIDGE_SCHEMA_KEY: 1,
            "state": state.to_dict(),
            "sequence": sequence,
            "payload": msgpack.packb([0, events], use_bin_type=True),
        },
        use_bin_type=True,
    )
    return parser.parse(payload, node)


def test_reset_clears_only_target_and_old_replay_cannot_restore_it(layout, stored):
    cache = KVCacheStore()
    cache.configure_registry(layout())
    parser = VLLMKVParser(group_aware=True)
    state = CacheLifecycleState("engine")
    events = [stored(), stored(group=1, kind="mamba", block=8)]
    for node in ("a", "b"):
        cache.update_lifecycle(node, state)
        assert cache.apply_update(update(parser, state, 0, events, node))
        assert not cache.apply_update(update(parser, state, 0, events, node))
        assert cache.group_caches[node].retained_hashes[(1, "r16")] == 1
    hashes = [str(value) for value in get_prefix_hashes_incremental(list(range(17)), 2, 0, 0)[0]]
    assert cache.estimate_prefixes(["a", "b"], hashes, 17)["a"].estimated_hit_tokens == 16
    pending = replace(state, epoch=1, revision=1, index_state="awaiting_clear")
    cache.update_lifecycle("a", pending)
    assert not cache.apply_update(update(parser, state, 1, events))
    estimates = cache.estimate_prefixes(["a", "b"], hashes, 17)
    assert estimates["a"].estimated_hit_tokens == 0 and estimates["b"].estimated_hit_tokens == 16
    ready = replace(pending, revision=2, index_state="ready")
    cache.update_lifecycle("a", ready)
    assert cache.apply_update(update(parser, ready, 2, [{"type": "AllBlocksCleared"}, *events]))
    assert cache.estimate_prefixes(["a"], hashes, 17)["a"].estimated_hit_tokens == 16


def test_retired_instance_and_late_ready_notification_are_rejected():
    cache = KVCacheStore()
    first = CacheLifecycleState("old", revision=10)
    cache.update_lifecycle("a", first)
    second = CacheLifecycleState("new")
    assert cache.update_lifecycle("a", second)
    assert not cache.update_lifecycle("a", replace(first, revision=100))
    blocked = replace(second, revision=3, serving=False)
    cache.update_lifecycle("a", blocked)
    assert not cache.update_lifecycle("a", replace(second, revision=2))
    assert not cache.lifecycle_states["a"].serving


def test_stream_gap_requires_clear_before_stored_can_be_applied():
    cache = KVCacheStore()
    parser = VLLMKVParser()
    event = ["BlockStored", ["h"], None, list(range(4)), 4, None, "GPU"]
    assert cache.apply_update(parser.parse(msgpack.packb([0, [event]]), "a"))
    assert not cache.apply_update(parser.parse(KVEventFrame(b"", error="gap"), "a"))
    assert not cache.apply_update(parser.parse(msgpack.packb([0, [event]]), "a"))
    assert cache.apply_update(parser.parse(msgpack.packb([0, [["AllBlocksCleared"], event]]), "a"))
    assert cache.per_replica_block_counts()["a"] == 1


def test_legacy_clear_discards_stores_before_the_boundary_and_isolates_nodes():
    parser = VLLMKVParser()
    before = ["BlockStored", ["old"], None, [1, 2, 3, 4], 4, None, "GPU"]
    after = ["BlockStored", ["new"], None, [5, 6, 7, 8], 4, None, "GPU"]
    parser.parse(msgpack.packb([0, [before]]), "b")
    result = parser.parse(msgpack.packb([0, [before, ["AllBlocksCleared"], after]]), "a")
    assert sum(map(len, result.add_blocks.values())) == 1
    assert set(parser.remote_to_local_block_hash) == {"new"}
    parser.parse(msgpack.packb([0, []]), "b")
    assert set(parser.remote_to_local_block_hash) == {"old"}


def test_old_clear_cannot_remove_a_stream_gap_barrier():
    cache = KVCacheStore()
    parser = VLLMKVParser()
    first = parser.parse(KVEventFrame(msgpack.packb([0, []]), sequence=5), "a")
    assert cache.apply_update(first)
    assert not cache.apply_update(parser.parse(KVEventFrame(b"", error="missing_batch"), "a"))
    duplicate = parser.parse(KVEventFrame(msgpack.packb([0, [["AllBlocksCleared"]]]), sequence=4), "a")
    assert not cache.apply_update(duplicate)
    assert "a" in cache._stream_errors
    fresh = parser.parse(KVEventFrame(msgpack.packb([0, [["AllBlocksCleared"]]]), sequence=6), "a")
    assert cache.apply_update(fresh)
    assert "a" not in cache._stream_errors


def test_obsolete_envelope_does_not_replace_current_legacy_context():
    cache = KVCacheStore()
    state = CacheLifecycleState("engine", epoch=2)
    cache.update_lifecycle("a", state)
    parser = VLLMKVParser(lifecycle_reader=lambda node: cache.lifecycle_states.get(node))
    event = ["BlockStored", ["current"], None, [1, 2, 3, 4], 4, None, "GPU"]
    assert cache.apply_update(update(parser, state, 2, [event]))
    assert update(parser, replace(state, epoch=1), 1, [event]) is None
    assert set(parser.remote_to_local_block_hash) == {"current"}


def test_malformed_bridge_is_an_explicit_stream_failure():
    parser = VLLMKVParser()
    broken = msgpack.packb({BRIDGE_SCHEMA_KEY: 1, "state": {}})
    assert parser.parse(broken, "a").stream_error == "invalid_bridge_envelope"
