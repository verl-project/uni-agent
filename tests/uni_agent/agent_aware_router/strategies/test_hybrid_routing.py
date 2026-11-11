"""Hybrid cache estimates and retention feed the existing routing policies."""

import msgpack
import pytest

from uni_agent.agent_aware_router.collectors.collector import Collector
from uni_agent.agent_aware_router.collectors.parse.base import KVCacheUpdate
from uni_agent.agent_aware_router.collectors.parse.basic.inflight import InflightParser
from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.collectors.transport.base import TransportMessage
from uni_agent.agent_aware_router.collectors.transport.callback import StatisticEvent
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.store.per_replica_store import PerReplicaStore
from uni_agent.agent_aware_router.store.per_request_store import PerRequestStore
from uni_agent.agent_aware_router.strategies.base import ReplicaInfo
from uni_agent.agent_aware_router.strategies.kvc_aware import STICKY_TOP_SCORE, KVCacheAwareStrategy
from uni_agent.agent_aware_router.types import MetricKey, OverloadMode, SlowCut
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental
from uni_agent.agent_aware_router.utils.prefix_cache import resolve_prefix_hashes

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.fixture
def store(layout):
    for cls in (KVCacheStore, PerReplicaStore, PerRequestStore):
        cls._instance = None
    value = DataStore()
    raw = layout()
    raw["cache_reservation_profile"]["num_blocks"] = 100
    value.configure_cache_registry(raw)
    yield value
    for cls in (KVCacheStore, PerReplicaStore, PerRequestStore):
        cls._instance = None


def strategy(mode=SlowCut.CAPACITY_TOKEN_AWARE):
    value = KVCacheAwareStrategy(
        load_threshold=0.9,
        layer_weights={"gpu": 1.0, "cpu": 0.0, "ssd": 0.0},
        do_shortcut=False,
        slow_cut=mode,
        tie_tolerance=0,
    )
    value.set_capacity(64, 1024)
    return value


def dispatch(store, node, prompt, request):
    updates = InflightParser().parse(
        StatisticEvent("on_acquire", request_id=request, replica_id=node, prompt_ids=tuple(prompt)),
        node,
        schema=store.dispatch_schema(),
    )
    pin = updates[0]
    store.pin_request(request, node, pin.block_hashes, pin.schema, reservations=pin.reservations)


@pytest.fixture
def populate(parse, stored):
    def make_populate(store):
        for node, checkpoint in (("a", "r8"), ("b", "r16")):
            events = parse([stored(), stored(1, "mamba", 8, hashes=[checkpoint])], node)
            store.apply_kv_update(KVCacheUpdate(node_id=node, events=events))

    return make_populate


def test_collected_events_route_by_common_hit_and_group_removal(store, parse, removed, stored):
    collector = Collector(object(), VLLMKVParser())
    try:
        for node, checkpoint in (("a", "r8"), ("b", "r16")):
            payload = msgpack.packb({"events": [stored(1, "mamba", 8, hashes=[checkpoint]), stored()]})
            collector._write_kv_update(collector._parser.parse(TransportMessage(payload), node))
    finally:
        collector._loop.close()
    assert store.cache_registry_state.registry.local_hash_unit == 2
    store.refresh_metrics({node: {MetricKey.NUM_GPU_BLOCKS: 100} for node in ("a", "b")})
    replicas = [ReplicaInfo("a"), ReplicaInfo("b")]
    scores = strategy().score(list(range(17)), store, replicas)
    assert scores[1] > scores[0]
    hashes = resolve_prefix_hashes(list(range(17)), None, store)
    results = store.estimate_prefixes(["a", "b"], hashes, 17)
    assert results["a"].estimated_hit_tokens == 8
    assert results["b"].hit_rate == 16 / 17
    # A partial Stored whose parent has not been published must not suppress
    # either replica's already established cache score.
    events = parse([stored(block=2, parent="r20", hashes=["r22"], tokens=[20, 21])], "b")
    store.apply_kv_update(KVCacheUpdate(node_id="b", events=events))
    assert store.estimate_prefixes(["a", "b"], hashes, 17) == results
    # Pending aliases affect retention while established prefix hits stay intact.
    assert store.per_replica_block_counts() == {"a": 5, "b": 6}
    pending_scores = strategy().score(list(range(17)), store, replicas)
    assert pending_scores == scores  # retained identities are not physical occupancy
    store.apply_kv_update(KVCacheUpdate(node_id="b", events=parse([removed(1, ["r16"])], "b")))
    scores = strategy().score(list(range(17)), store, replicas)
    assert scores[0] > scores[1]


def test_one_warming_replica_keeps_other_replicas_cache_scores(store, populate):
    populate(store)
    store.clear_kv_node("b")
    store.refresh_metrics({"a": {MetricKey.INFLIGHT_COUNT: 3}, "b": {MetricKey.INFLIGHT_COUNT: 0}})
    scores = strategy().score(list(range(17)), store, [ReplicaInfo("a"), ReplicaInfo("b")])
    assert scores[0] > scores[1]


def test_replay_uses_saved_tokens_in_scores_and_capacity_selection(store, layout, parse, stored):
    raw = layout((("full_attention", 4), ("sliding_window_mla", 4)), window=8)
    raw["groups"][0].update(prefix_cacheable=True, prefix_replay_tokens=0)
    raw["groups"][1].update(prefix_cacheable=False, prefix_replay_tokens=8)
    raw["kv_cache_capacity_tokens"] = 1000
    raw["cache_reservation_profile"]["num_blocks"] = 100
    store.configure_cache_registry(raw)
    store.apply_kv_update(KVCacheUpdate(node_id="a", events=parse([stored()])))
    store.apply_kv_update(KVCacheUpdate(node_id="b", events=parse([stored(tokens=list(range(8)))], "b")))
    prompt = list(range(17))
    replicas = [ReplicaInfo("a"), ReplicaInfo("b")]
    prefix = strategy()
    assert prefix.score(prompt, store, replicas) == [STICKY_TOP_SCORE, 0]
    estimates = store.estimate_prefixes(["a", "b"], resolve_prefix_hashes(prompt, None, store), len(prompt))
    assert estimates["a"].estimated_saved_tokens == 8
    assert estimates["b"].estimated_saved_tokens == 0
    assert store.request_block_need(17, 8) < store.request_block_need(17, 0)
    # Native sampled usage cannot override the dispatch budget on any layout.
    store.refresh_metrics(
        {
            "a": {MetricKey.NUM_GPU_BLOCKS: 100, MetricKey.KV_CACHE_USAGE_PERC: 1.0},
            "b": {MetricKey.NUM_GPU_BLOCKS: 100, MetricKey.KV_CACHE_USAGE_PERC: 0.0},
        }
    )
    assert prefix.score(prompt, store, replicas) == [STICKY_TOP_SCORE, 0]


def test_capacity_uses_group_reservations_not_hash_unit_or_sampled_usage(store, layout):
    raw = layout((("full_attention", 4), ("sliding_window", 8)), unit=2)
    raw["cache_reservation_profile"]["num_blocks"] = 12
    raw["kv_cache_capacity_tokens"] = 100000  # equivalent tokens are not physical pool blocks
    store.configure_cache_registry(raw)
    prompt = list(range(17))
    dispatch(store, "a", prompt, "first")
    assert store.inflight_reserved_blocks("a") == 6  # 4 Full blocks + 2 SWA blocks
    store.refresh_metrics({"a": {MetricKey.KV_CACHE_USAGE_PERC: 0.0}, "b": {MetricKey.KV_CACHE_USAGE_PERC: 1.0}})
    strat = strategy()
    strat.load_threshold = 0.4  # a has six free blocks, below the 7.2-block gate
    replicas = [ReplicaInfo("a"), ReplicaInfo("b")]
    assert strat._total_block_capacity(store) == 12
    assert strat.score(prompt, store, replicas) == [0, STICKY_TOP_SCORE]
    # Dispatching the same prefix on a shares every prefix reservation.
    dispatch(store, "a", prompt, "shared")
    assert store.inflight_reserved_blocks("a") == 6
    store.unpin_request("first")
    assert store.inflight_reserved_blocks("a") == 6
    store.unpin_request("shared")
    assert store.inflight_reserved_blocks("a") == 0
    # Released reservations vanish, while #257's predicted cache benefit remains.
    assert strat.score(prompt, store, replicas) == [STICKY_TOP_SCORE, 0]


def test_unknown_capacity_preserves_main_least_inflight_fallback(store, layout, populate):
    store.configure_cache_registry(layout())
    populate(store)
    store.refresh_metrics(
        {
            "a": {MetricKey.INFLIGHT_COUNT: 1, MetricKey.INFLIGHT_TOKENS: 3},
            "b": {MetricKey.INFLIGHT_COUNT: 3, MetricKey.INFLIGHT_TOKENS: 100},
        }
    )
    strat = strategy(SlowCut.CAPACITY_TOKEN_AWARE)
    replicas = [ReplicaInfo("a"), ReplicaInfo("b")]
    assert strat.score(list(range(17)), store, replicas) == [STICKY_TOP_SCORE, 0]
    store.put_sticky_binding("session", "a")
    assert strat.score(list(range(17)), store, replicas, "session") == [STICKY_TOP_SCORE, 0]


def test_request_hashes_recompute_after_layout_changes(store, layout):
    prompt = list(range(17))
    resolve_prefix_hashes(prompt, "request", store)
    store.configure_cache_registry(layout(unit=4))
    resolved = resolve_prefix_hashes(prompt, "request", store)
    expected, _ = get_prefix_hashes_incremental(prompt, 4, 0, 0)
    assert resolved == [str(h) for h in expected]


def test_retained_identities_are_not_native_pool_occupancy(store, populate):
    populate(store)
    # Both replicas have five event identities, but different pool sizes.
    # Native usage deliberately orders them in the opposite direction.
    store.refresh_metrics(
        {
            "a": {MetricKey.NUM_GPU_BLOCKS: 5, MetricKey.KV_CACHE_USAGE_PERC: 0.0},
            "b": {MetricKey.NUM_GPU_BLOCKS: 100, MetricKey.KV_CACHE_USAGE_PERC: 0.95},
        }
    )
    assert store.per_replica_block_counts() == {"a": 5, "b": 5}
    assert store.kv_cache_load("a") == 1.0
    assert store.kv_cache_load("b") == 0.05
    strat = strategy()
    replicas = [ReplicaInfo("a"), ReplicaInfo("b")]
    strat.overload_mode = OverloadMode.KV_CACHE_USAGE_PERC
    assert [strat.is_overloaded(store, r) for r in replicas] == [False, True]


def test_group_retention_counts_reset_with_replica_and_registry(store, layout, populate):
    populate(store)
    assert store.per_replica_block_counts() == {"a": 5, "b": 5}
    store.clear_kv_node("a")
    assert store.per_replica_block_counts() == {"b": 5}
    store.configure_cache_registry(layout())
    assert store.per_replica_block_counts() == {}


@pytest.mark.parametrize("mode", ["align", "all"])
def test_missing_private_budget_uses_same_unknown_capacity_fallback(store, layout, mode):
    raw = layout(mode=mode)
    del raw["cache_reservation_profile"]
    store.configure_cache_registry(raw)
    prompt = list(range(17))
    dispatch(store, "a", prompt, "predicted")
    assert store.inflight_reserved_blocks("a") is None
    estimate = store.estimate_prefixes(["a"], resolve_prefix_hashes(prompt, None, store), len(prompt))["a"]
    assert estimate.provisional and estimate.estimated_hit_tokens == 16
    store.refresh_metrics(
        {
            "a": {MetricKey.NUM_GPU_BLOCKS: 100, MetricKey.INFLIGHT_COUNT: 3},
            "b": {MetricKey.NUM_GPU_BLOCKS: 100, MetricKey.INFLIGHT_COUNT: 0},
        }
    )
    assert strategy().score(prompt, store, [ReplicaInfo("a"), ReplicaInfo("b")]) == [0, STICKY_TOP_SCORE]
