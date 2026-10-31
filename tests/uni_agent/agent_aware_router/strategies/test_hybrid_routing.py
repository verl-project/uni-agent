"""Hybrid cache estimates and retention feed the existing routing policies."""

import msgpack
import pytest

from uni_agent.agent_aware_router.collectors.collector import Collector
from uni_agent.agent_aware_router.collectors.parse.base import KVCacheUpdate
from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
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
    value.configure_cache_registry(layout())
    yield value
    for cls in (KVCacheStore, PerReplicaStore, PerRequestStore):
        cls._instance = None


def strategy(mode=SlowCut.PREFIX_LOAD_AWARE):
    value = KVCacheAwareStrategy(
        alpha=0.7,
        load_threshold=0.9,
        layer_weights={"gpu": 1.0, "cpu": 0.0, "ssd": 0.0},
        do_shortcut=False,
        slow_cut=mode,
    )
    value.set_capacity(64, 1024)
    return value


@pytest.fixture
def populate(parse, stored):
    def make_populate(store):
        for node, checkpoint in (("a", "r8"), ("b", "r16")):
            events = parse([stored(), stored(1, "mamba", 8, hashes=[checkpoint])], node)
            store.apply_kv_update(KVCacheUpdate(node_id=node, group_events=events))

    return make_populate


def test_collected_events_route_by_common_hit_and_group_removal(store, parse, removed, stored):
    collector = Collector(object(), VLLMKVParser(group_aware=True))
    try:
        for node, checkpoint in (("a", "r8"), ("b", "r16")):
            payload = msgpack.packb({"events": [stored(1, "mamba", 8, hashes=[checkpoint]), stored()]})
            collector._write_kv_update(collector._parser.parse(payload, node))
    finally:
        collector._loop.close()
    assert store.get_block_size() is None  # Group events must not set physical block size.
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
    store.apply_kv_update(KVCacheUpdate(node_id="b", group_events=events))
    assert store.estimate_prefixes(["a", "b"], hashes, 17) == results
    # Pending aliases affect retention while established prefix hits stay intact.
    assert store.per_replica_block_counts() == {"a": 5, "b": 6}
    pending_scores = strategy().score(list(range(17)), store, replicas)
    assert pending_scores[1] < scores[1] and pending_scores[1] > pending_scores[0]
    store.apply_kv_update(KVCacheUpdate(node_id="b", group_events=parse([removed(1, ["r16"])], "b")))
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
    store.configure_cache_registry(raw)
    store.apply_kv_update(KVCacheUpdate(node_id="a", group_events=parse([stored()])))
    store.apply_kv_update(KVCacheUpdate(node_id="b", group_events=parse([stored(tokens=list(range(8)))], "b")))
    prompt = list(range(17))
    replicas = [ReplicaInfo("a"), ReplicaInfo("b")]
    prefix = strategy()
    prefix.alpha = 1.0
    assert prefix.score(prompt, store, replicas) == pytest.approx([8 / 17, 0])
    store.refresh_metrics(
        {
            "a": {MetricKey.KV_CACHE_USAGE_PERC: 0.995},
            "b": {MetricKey.KV_CACHE_USAGE_PERC: 0.98},
        }
    )
    capacity = strategy(SlowCut.CAPACITY_TOKEN_AWARE)
    capacity.tie_tolerance = 0
    store.put_sticky_binding("session", "a")
    # Replaying a's tail needs 9 tokens, above its 5-token free capacity.
    assert capacity.score(prompt, store, replicas, "session") == [0, STICKY_TOP_SCORE]


def test_capacity_uses_real_tokens_not_local_hash_or_group_entry_counts(store, layout, parse, populate, stored):
    meta = layout()
    meta["kv_cache_capacity_tokens"] = 1000
    store.configure_cache_registry(meta)
    populate(store)
    store.set_block_size(4096)  # Must not become hybrid physical capacity.
    # A five-token occupancy cost is less than b's eight-token hit advantage.
    # Substituting legacy block capacity would overstate that cost and select a.
    for node, usage in (("a", 0.0), ("b", 0.005)):
        store.refresh_metrics({node: {MetricKey.NUM_GPU_BLOCKS: 100, MetricKey.KV_CACHE_USAGE_PERC: usage}})
    events = parse([stored(parent="r20", block=2, tokens=[20, 21], hashes=["r22"])], "b")
    store.apply_kv_update(KVCacheUpdate(node_id="b", group_events=events))
    assert store.kv_cache_load("b") - store.kv_cache_load("a") == pytest.approx(0.01)
    strat = strategy(SlowCut.CAPACITY_TOKEN_AWARE)
    strat.tie_tolerance = 0.0  # Isolate token-capacity arithmetic from near-best sampling.
    store.put_sticky_binding("session", "a")  # Warm capacity branch, not sticky shortcut.
    assert strat.score(list(range(17)), store, [ReplicaInfo("a"), ReplicaInfo("b")], "session") == [0, STICKY_TOP_SCORE]


def test_unknown_hybrid_capacity_preserves_cold_and_warm_selection(store, populate):
    populate(store)
    store.refresh_metrics(
        {
            "a": {MetricKey.NUM_GPU_BLOCKS: 100, MetricKey.INFLIGHT_COUNT: 3, MetricKey.INFLIGHT_TOKENS: 3},
            "b": {MetricKey.NUM_GPU_BLOCKS: 100, MetricKey.INFLIGHT_COUNT: 1, MetricKey.INFLIGHT_TOKENS: 100},
        }
    )
    strat = strategy(SlowCut.CAPACITY_TOKEN_AWARE)
    replicas = [ReplicaInfo("a"), ReplicaInfo("b")]
    assert strat.score(list(range(17)), store, replicas) == [STICKY_TOP_SCORE, 0]
    store.put_sticky_binding("session", "a")
    assert strat.score(list(range(17)), store, replicas, "session") == [0, STICKY_TOP_SCORE]


def test_request_hashes_recompute_after_layout_changes(store, layout):
    prompt = list(range(17))
    resolve_prefix_hashes(prompt, "request", store)
    store.configure_cache_registry(layout(unit=4))
    resolved = resolve_prefix_hashes(prompt, "request", store)
    expected, _ = get_prefix_hashes_incremental(prompt, 4, 0, 0)
    assert resolved == [str(h) for h in expected]


def test_retained_load_drives_prefix_and_kv_load_but_not_native_usage_mode(store, populate):
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
    strat.alpha = 0.0  # Isolate the existing load component from prefix hits.
    strat.load_weights = (1.0, 0.0, 0.0, 0.0)
    replicas = [ReplicaInfo("a"), ReplicaInfo("b")]
    assert strat.score(list(range(17)), store, replicas) == pytest.approx([0.0, 0.95])
    strat.overload_mode = OverloadMode.KV_LOAD
    assert [strat.is_overloaded(store, r) for r in replicas] == [True, False]
    strat.overload_mode = OverloadMode.KV_CACHE_USAGE_PERC
    assert [strat.is_overloaded(store, r) for r in replicas] == [False, True]


def test_group_retention_counts_reset_with_replica_and_registry(store, layout, populate):
    populate(store)
    assert store.per_replica_block_counts() == {"a": 5, "b": 5}
    store.clear_kv_node("a")
    assert store.per_replica_block_counts() == {"b": 5}
    store.configure_cache_registry(layout())
    assert store.per_replica_block_counts() == {}
