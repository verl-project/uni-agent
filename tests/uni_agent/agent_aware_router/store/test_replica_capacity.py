"""Per-replica native capacity, common-layout validation and safe metadata fallback."""

from copy import deepcopy

import pytest

from uni_agent.agent_aware_router.collectors.parse.basic.inflight import InflightParser
from uni_agent.agent_aware_router.collectors.parse.vllm.metrics import VLLMMetricsParser
from uni_agent.agent_aware_router.collectors.transport.callback import StatisticEvent
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.store.per_replica_store import PerReplicaStore
from uni_agent.agent_aware_router.store.per_request_store import PerRequestStore
from uni_agent.agent_aware_router.strategies.base import ReplicaInfo
from uni_agent.agent_aware_router.strategies.kvc_aware import STICKY_TOP_SCORE, KVCacheAwareStrategy
from uni_agent.agent_aware_router.types import MetricKey, OverloadMode
from uni_agent.agent_aware_router.types.kv_cache import KVCacheUpdate

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.fixture
def store():
    for cls in (KVCacheStore, PerReplicaStore, PerRequestStore):
        cls._instance = None
    yield DataStore()
    for cls in (KVCacheStore, PerReplicaStore, PerRequestStore):
        cls._instance = None


def profile(layout, capacity):
    raw = layout((("full_attention", 4),), unit=4)
    raw["cache_reservation_profile"].update(num_blocks=capacity + 1, capacity_blocks=capacity)
    return raw


def dispatch(store, node, request, tokens):
    pin = InflightParser().parse(
        StatisticEvent("on_acquire", request_id=request, replica_id=node, prompt_ids=tuple(tokens)),
        node,
        schema=store.dispatch_schema(node),
    )[0]
    store.pin_request(request, node, pin.block_hashes, pin.schema, reservations=pin.reservations)


def strategy():
    return KVCacheAwareStrategy(
        load_threshold=0.9,
        layer_weights={"gpu": 1.0, "cpu": 0.0, "ssd": 0.0},
        overload_mode=OverloadMode.REMAINING_CAP,
        do_shortcut=True,
        tie_tolerance=0,
    )


@pytest.mark.parametrize("source", ["metadata", "metrics"])
def test_capacity_gate_and_sticky_use_the_current_replica(store, layout, source):
    a, b = profile(layout, 99), profile(layout, 9)
    if source == "metrics":
        for raw in (a, b):
            raw["cache_reservation_profile"].pop("capacity_blocks")
    store.configure_cache_registry(a, replica_metadata={"a": a, "b": b})
    # Exercise the actual scrape boundary; metadata has priority over these samples.
    for node, configured in (("a", 100), ("b", 10)):
        update = VLLMMetricsParser().parse(f'vllm:cache_config_info{{num_gpu_blocks="{configured}"}} 1', node)
        store.refresh_metrics({node: update.metrics})
    assert (store.capacity_blocks("a"), store.capacity_blocks("b")) == (99, 9)
    dispatch(store, "b", "busy", range(36))
    assert store.inflight_reserved_blocks("b") == 9
    strat = strategy()
    assert strat._available_blocks(store, "b", store.capacity_blocks("b")) == 0
    assert strat.is_overloaded(store, ReplicaInfo("b"))
    store.put_sticky_binding("session", "b")
    assert strat.score([100, 101, 102, 103, 104], store, [ReplicaInfo("a"), ReplicaInfo("b")], "session") == [
        STICKY_TOP_SCORE,
        0,
    ]


def test_missing_replica_capacity_does_not_borrow_sibling_metadata_or_metrics(store, layout):
    a, b = profile(layout, 99), profile(layout, 9)
    b["cache_reservation_profile"].pop("capacity_blocks")
    store.configure_cache_registry(a, replica_metadata={"a": a, "b": b})
    store.refresh_metrics({"a": {MetricKey.KV_CACHE_CAPACITY_BLOCKS: 999}})
    assert store.capacity_blocks("a") == 99  # own metadata beats sampled metric
    assert store.capacity_blocks("b") is None
    assert store.capacity_blocks("new-node") is None
    assert not strategy().is_overloaded(store, ReplicaInfo("b"))


@pytest.mark.parametrize("change", ["block_size", "hash_unit", "prefix_policy", "model"])
def test_explicit_layout_conflict_is_rejected_before_mutating_store(store, layout, change):
    a = layout((("full_attention", 4), ("sliding_window_mla", 4)), unit=4, window=8)
    a["cache_reservation_profile"].update(num_blocks=100, capacity_blocks=99)
    store.configure_cache_registry(a, replica_metadata={"a": a})
    dispatch(store, "a", "old-request", range(8))
    before = store.cache_registry_state
    held_before = store.inflight_reserved_blocks("a")
    b = deepcopy(a)
    if change == "block_size":
        b["groups"][0]["block_size"] = 8
    elif change == "hash_unit":
        b["cache_config"]["prefix_match_unit"] = 2
    elif change == "prefix_policy":
        b["groups"][1].update(prefix_cacheable=False, prefix_replay_tokens=4)
    else:
        b["model_config"]["model"] = "different-model"
    with pytest.raises(ValueError, match="Incompatible KV cache layout"):
        store.configure_cache_registry(a, replica_metadata={"a": a, "b": b})
    assert store.cache_registry_state is before
    assert store.capacity_blocks("a") == 99
    assert store.inflight_reserved_blocks("a") == held_before
    store.unpin_request("old-request")
    assert store.inflight_reserved_blocks("a") == 0


def test_optional_metadata_details_and_capacities_do_not_make_layouts_conflict(store, layout):
    a, b = profile(layout, 99), profile(layout, 9)
    b["groups"][0].update(prefix_cacheable=True, prefix_replay_tokens=0)
    a["model_config"]["model"] = None
    b["native_hash_config"] = {"root_hash_hex": "22" * 32}
    store.configure_cache_registry(a, replica_metadata={"a": a, "b": b})
    assert store.capacity_blocks("b") == 9


def test_first_unavailable_metadata_uses_valid_peer_and_keeps_untagged_legacy_inputs(store, layout, parse, stored):
    missing = {"status": "unsupported", "reason": "metadata_method_unavailable"}
    b = profile(layout, 9)
    state = store.configure_cache_registry(missing, replica_metadata={"a": missing, "b": b})
    assert state.status == "registered"
    assert store.capacity_blocks("a") is None
    assert store.capacity_blocks("b") == 9
    event = stored(tokens=list(range(8)))
    event.pop("group_idx")
    event.pop("kv_cache_spec_kind")
    store.apply_kv_update(KVCacheUpdate("a", events=parse([event], "a")))
    assert store.per_replica_block_counts()["a"] == 2


@pytest.mark.parametrize("invalid", ["disabled", "malformed"])
def test_explicitly_unusable_metadata_cannot_use_peer_layout_or_own_capacity_metric(store, layout, invalid):
    a, b = profile(layout, 99), profile(layout, 9)
    if invalid == "disabled":
        b["cache_config"]["enable_prefix_caching"] = False
    else:
        b["groups"][0]["block_size"] = 0
    store.configure_cache_registry(a, replica_metadata={"a": a, "b": b})
    store.refresh_metrics({"b": {MetricKey.KV_CACHE_CAPACITY_BLOCKS: 9}})
    assert store.capacity_blocks("b") is None
    assert store.dispatch_schema("b") is None
    assert store.estimate_prefixes(["b"], [], 5)["b"].estimated_hit_tokens == 0


def test_private_reservations_and_need_come_from_each_replica_profile(store, layout):
    a = layout()
    b = deepcopy(a)
    a["cache_reservation_profile"]["groups"][1]["max_blocks_per_request"] = 2
    b["cache_reservation_profile"]["groups"][1]["max_blocks_per_request"] = 4
    store.configure_cache_registry(a, replica_metadata={"a": a, "b": b})
    dispatch(store, "a", "a-request", range(17))
    dispatch(store, "b", "b-request", range(17))
    assert store.inflight_reserved_blocks("a") == 6
    assert store.inflight_reserved_blocks("b") == 8
    assert store.request_block_need(17, 16, node_id="a") == 2.25
    assert store.request_block_need(17, 16, node_id="b") == 4.25


def test_invalid_first_metadata_does_not_block_missing_peer_legacy_inference(store, layout, parse, stored):
    invalid = profile(layout, 99)
    invalid["groups"][0]["block_size"] = 0
    missing = {"status": "unsupported", "reason": "metadata_method_unavailable"}
    store.configure_cache_registry(invalid, replica_metadata={"a": invalid, "b": missing})
    event = stored(tokens=list(range(8)))
    event.pop("group_idx")
    event.pop("kv_cache_spec_kind")
    store.apply_kv_update(KVCacheUpdate("b", events=parse([event], "b")))
    assert store.cache_registry_state.inferred
    assert store.per_replica_block_counts() == {"b": 2}
    assert store.dispatch_schema("a") is None
    assert store.dispatch_schema("b") is not None


def test_zero_capacity_breaks_sticky_even_when_private_budget_is_unknown(store, layout):
    raw = layout()
    raw["cache_reservation_profile"] = None
    store.configure_cache_registry(raw, replica_metadata={"a": raw})
    store.refresh_metrics({"a": {MetricKey.KV_CACHE_CAPACITY_BLOCKS: 0}})
    assert store.inflight_reserved_blocks("a") is None
    assert strategy().is_overloaded(store, ReplicaInfo("a"))
