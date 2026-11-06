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

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import msgpack
import pytest

from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.collectors.transport.base import TransportMessage
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.types.cache_lifecycle import BRIDGE_SCHEMA_KEY, CacheLifecycleState
from uni_agent.agent_aware_router.types.kv_cache import GroupReservation, KVCacheUpdate
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
    return parser.parse(TransportMessage(payload), node)


def test_reset_clears_only_target_and_old_replay_cannot_restore_it(layout, stored):
    cache = KVCacheStore()
    cache.configure_registry(layout())
    parser = VLLMKVParser()
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
    second = CacheLifecycleState("new", revision=11)
    assert cache.update_lifecycle("a", second)
    assert not cache.update_lifecycle("a", replace(first, revision=100))
    blocked = replace(second, revision=13, serving=False)
    cache.update_lifecycle("a", blocked)
    assert not cache.update_lifecycle("a", replace(second, revision=12))
    assert not cache.lifecycle_states["a"].serving


@pytest.mark.parametrize("identity", ["old-epoch", "old-instance"])
def test_obsolete_envelope_cannot_clear_residency_or_change_inferred_layout(identity):
    cache = KVCacheStore()
    state = CacheLifecycleState("engine", epoch=2)
    cache.update_lifecycle("a", state)
    parser = VLLMKVParser()
    event = ["BlockStored", ["current"], None, [1, 2, 3, 4], 4, None, "GPU"]
    assert cache.apply_update(update(parser, state, 2, [event]))
    schema = cache.hash_schema()
    hashes = [str(value) for value in get_prefix_hashes_incremental([1, 2, 3, 4, 5], 4, 0, 0)[0]]
    obsolete = replace(state, epoch=1) if identity == "old-epoch" else replace(state, instance_id="old")
    conflicting = ["BlockStored", ["obsolete"], None, list(range(8)), 8, None, "GPU"]
    assert not cache.apply_update(update(parser, obsolete, 100, [["AllBlocksCleared"], conflicting]))
    broken = msgpack.packb(
        {BRIDGE_SCHEMA_KEY: 1, "state": obsolete.to_dict(), "sequence": 101, "payload": b"\xc1"},
        use_bin_type=True,
    )
    assert not cache.apply_update(parser.parse(TransportMessage(broken), "a"))
    assert cache.hash_schema() == schema
    assert cache.per_replica_block_counts() == {"a": 1}
    assert cache.estimate_prefixes(["a"], hashes, 5)["a"].estimated_hit_tokens == 4
    assert cache.apply_update(update(parser, state, 3, [["BlockRemoved", ["current"], "GPU"]]))
    assert cache.estimate_prefixes(["a"], hashes, 5)["a"].estimated_hit_tokens == 0


def test_bridge_requires_control_handshake_before_layout_inference():
    cache = KVCacheStore()
    state = CacheLifecycleState("engine")
    event = ["BlockStored", ["current"], None, [1, 2, 3, 4], 4, None, "GPU"]
    schema = cache.hash_schema()
    result = update(VLLMKVParser(), state, 0, [event])
    assert not cache.apply_update(result)
    assert cache.hash_schema() == schema
    assert cache.per_replica_block_counts() == {}
    assert cache.update_lifecycle("a", state)
    assert cache.apply_update(result)
    assert cache.per_replica_block_counts() == {"a": 1}


def test_removing_replica_retires_identity_and_discards_stream_progress(layout, stored):
    cache = KVCacheStore()
    cache.configure_registry(layout((("full_attention", 4),)))
    parser = VLLMKVParser()
    first = CacheLifecycleState("old")
    for node in ("a", "b"):
        assert cache.update_lifecycle(node, first)
        assert cache.apply_update(update(parser, first, 100, [stored()], node))
    failed = replace(update(parser, first, 101, []), stream_error="kv_sequence_gap")
    assert not cache.apply_update(failed)
    cache.remove_replica("a")
    assert cache.lifecycle_state("a") is None
    assert "a" not in cache.group_caches
    assert "a" not in cache._stream_errors and "a" not in cache._event_sequences
    assert cache.per_replica_block_counts() == {"b": 4}
    assert not cache.update_lifecycle("a", replace(first, revision=100))
    assert not cache.apply_update(update(parser, first, 102, [stored()]))
    replacement = CacheLifecycleState("replacement")
    assert cache.update_lifecycle("a", replacement)
    assert cache.apply_update(update(parser, replacement, 0, [stored()]))
    assert cache.per_replica_block_counts() == {"a": 4, "b": 4}
    assert not cache.update_lifecycle("a", replace(first, revision=100))
    cache.remove_replica("a")
    assert cache.update_lifecycle("a", first, registration=True)
    assert cache.lifecycle_state("a") == first
    assert cache.per_replica_block_counts() == {"b": 4}
    assert cache.apply_update(update(parser, first, 0, [stored()]))
    newer = replace(first, epoch=1, revision=1)
    assert cache.update_lifecycle("a", newer)
    assert not cache.update_lifecycle("a", first, registration=True)
    assert cache.lifecycle_state("a") == newer


@pytest.mark.parametrize("change", ["instance", "awaiting_clear", "removed"])
def test_lifecycle_invalidation_clears_only_target_reservations(change, layout):
    cache = KVCacheStore()
    cache.configure_registry(layout((("full_attention", 4),), unit=None))
    state = CacheLifecycleState("engine")
    for node in ("a", "b"):
        assert cache.update_lifecycle(node, state)
        cache.pin_request(
            f"old-{node}",
            node,
            ["prefix"],
            cache.dispatch_schema(),
            reservations=(GroupReservation(0, ("prefix",), private_blocks=2),),
        )
        assert cache.inflight_block_count(node) == 3
    if change == "instance":
        assert cache.update_lifecycle("a", replace(state, instance_id="replacement", revision=1))
    elif change == "awaiting_clear":
        assert cache.update_lifecycle("a", replace(state, epoch=1, revision=1, index_state="awaiting_clear"))
    else:
        cache.remove_replica("a")
    assert cache.get_marker("old-a") is None
    assert cache.inflight_block_count("a") == 0
    assert "a" not in cache.group_caches
    assert cache.get_marker("old-b") is not None
    assert cache.inflight_block_count("b") == 3
    assert "b" in cache.group_caches
    cache.pin_request(
        "new-a",
        "a",
        ["new-prefix"],
        cache.dispatch_schema(),
        reservations=(GroupReservation(0, ("new-prefix",), private_blocks=1),),
        predict=False,
    )
    cache.unpin_request("old-a")
    assert cache.inflight_block_count("a") == 2
    cache.unpin_request("new-a")
    assert cache.inflight_block_count("a") == 0


def test_paused_engine_preserves_reservations_and_awaiting_clear_prevents_predictions(layout):
    cache = KVCacheStore()
    cache.configure_registry(layout((("full_attention", 4),), unit=None))
    state = CacheLifecycleState("engine")
    assert cache.update_lifecycle("a", state)
    cache.pin_request(
        "running",
        "a",
        ["prefix"],
        cache.dispatch_schema(),
        reservations=(GroupReservation(0, ("prefix",)),),
    )
    assert cache.update_lifecycle("a", replace(state, revision=1, serving=False))
    assert cache.inflight_block_count("a") == 1
    assert cache.get_marker("running") is not None
    assert cache.update_lifecycle("a", replace(state, epoch=1, revision=2, index_state="awaiting_clear"))
    assert cache.inflight_block_count("a") == 0
    assert (
        cache.pin_request(
            "late",
            "a",
            ["prefix"],
            cache.dispatch_schema(),
            reservations=(GroupReservation(0, ("prefix",)),),
        )
        == 0
    )
    assert "a" not in cache.group_caches


@pytest.mark.parametrize("notification", ["delayed-clear", "stream-gap"])
def test_current_epoch_reservations_survive_delayed_clear_and_stream_errors(layout, notification):
    from uni_agent.agent_aware_router.server.cache_lifecycle import CacheLifecycle

    async def run():
        cache = KVCacheStore()
        cache.configure_registry(layout((("full_attention", 4),), unit=None))

        async def notify(sink, state):
            cache.update_lifecycle("a", CacheLifecycleState.from_dict(state))

        control = CacheLifecycle(notifier=notify)
        control.bind({"token": "router"})
        cache.update_lifecycle("a", control.state)
        schema = cache.dispatch_schema()
        cache.pin_request("before-reset", "a", ["old"], schema, reservations=(GroupReservation(0, ("old",)),))
        await control.begin("clear_kv_cache", 1)
        assert cache.get_marker("before-reset") is None
        await control.finish(True)

        async def deliver_clear():
            payload = await control.accept(msgpack.packb([0, [["AllBlocksCleared"]]]))
            parsed = VLLMKVParser().parse(TransportMessage(payload), "a")
            assert cache.apply_update(KVCacheUpdate("a", parsed.events, lifecycle=control.state))

        if notification == "stream-gap":
            await deliver_clear()
        cache.pin_request(
            "after-reset",
            "a",
            ["new"],
            schema,
            reservations=(GroupReservation(0, ("new",), private_blocks=2),),
        )
        assert cache.inflight_block_count("a") == 3
        if notification == "delayed-clear":
            await deliver_clear()
        else:
            await control.stream_gap("native_events_lost")
            assert cache.estimate_prefixes(["a"], ["new"], 5)["a"].estimated_hit_tokens == 0
        assert cache.get_marker("after-reset") is not None
        assert cache.inflight_block_count("a") == 3
        cache.unpin_request("before-reset")
        assert cache.inflight_block_count("a") == 3
        cache.unpin_request("after-reset")
        assert cache.inflight_block_count("a") == 0

    asyncio.run(run())


@pytest.mark.parametrize("new_profile", ["replacement", None])
def test_replica_metadata_refresh_updates_only_its_native_profile(new_profile, layout):
    cache = KVCacheStore()
    metadata = layout((("sliding_window", 4), ("full_attention", 8)), unit=None)
    metadata["profile"] = "original"

    def factory(raw, unit):
        profile = raw.get("profile")
        return SimpleNamespace(unit=unit, root=profile.encode(), profile_key=profile) if profile else None

    cache.set_native_hash_codec_factory(factory)
    cache.configure_registry(metadata, replica_metadata={"a": metadata, "b": metadata})
    state = CacheLifecycleState("engine")
    for node in ("a", "b"):
        assert cache.update_lifecycle(node, state)
        assert cache.apply_update(KVCacheUpdate(node, lifecycle=state))
    previous = cache._native_by_node["b"]
    untouched = cache.group_caches["b"]
    generation = cache.hash_schema()
    assert cache._native_by_node["a"] is previous
    replacement = {**metadata, "profile": new_profile}
    cache.configure_replica_metadata("a", replacement)
    assert cache.hash_schema() == generation
    assert cache.lifecycle_state("a") == state
    assert cache.lifecycle_state("b") == state
    assert cache._native_by_node["b"] is previous
    assert cache.group_caches["b"] is untouched
    assert "a" not in cache.group_caches
    if new_profile is None:
        assert "a" not in cache._native_by_node
    else:
        current = cache._native_by_node["a"]
        assert current is not previous and current.codec.profile_key == new_profile
        cache.configure_replica_metadata("c", replacement)
        assert cache._native_by_node["c"] is current
    cache.remove_replica("b")
    assert previous not in cache._native_indexes.values()
    assert "b" not in cache._replica_metadata


def test_removed_native_source_cannot_recreate_residency_until_registered(layout, stored, parse):
    cache = KVCacheStore()
    metadata = layout((("full_attention", 4),), unit=None)
    cache.configure_registry(metadata)
    event = KVCacheUpdate("a", parse([stored()]), sequence=0)
    assert cache.apply_update(event)
    cache.remove_replica("a")
    assert not cache.apply_update(replace(event, sequence=1))
    assert cache.per_replica_block_counts() == {}
    cache.configure_replica_metadata("a", metadata)
    assert cache.apply_update(event)
    assert cache.per_replica_block_counts() == {"a": 4}


@pytest.mark.parametrize("error", ["invalid_bridge_envelope", "event_stream_no_continuity"])
def test_stream_failure_without_identity_invalidates_acknowledged_engine(error, layout, stored):
    cache = KVCacheStore()
    cache.configure_registry(layout((("full_attention", 4),), unit=None))
    parser = VLLMKVParser()
    state = CacheLifecycleState("engine")
    assert cache.update_lifecycle("a", state)
    assert cache.apply_update(update(parser, state, 10, [stored()]))
    assert cache.per_replica_block_counts() == {"a": 4}
    assert not cache.apply_update(KVCacheUpdate("a", stream_error=error))
    assert cache.lifecycle_state("a") == state
    assert cache.per_replica_block_counts() == {}
    assert cache._stream_errors == {"a": error}
    assert not cache.apply_update(update(parser, state, 11, [stored()]))
    assert cache.apply_update(update(parser, state, 12, [{"type": "AllBlocksCleared"}, stored()]))
    assert cache.per_replica_block_counts() == {"a": 4}
