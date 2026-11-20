"""Per-replica native capacity, common-layout validation and safe metadata fallback."""

from copy import deepcopy

import pytest

from uni_agent.agent_aware_router.collectors.parse import InflightBlockUpdate
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
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental
from uni_agent.agent_aware_router.utils.prefix_cache import resolve_prefix_hashes


@pytest.fixture
def store(monkeypatch):
    for cls in (KVCacheStore, PerReplicaStore, PerRequestStore):
        monkeypatch.setattr(cls, "_instance", None)
    return DataStore()


def profile(layout, capacity):
    raw = layout((("full_attention", 4),), unit=4)
    raw["cache_reservation_profile"].update(num_blocks=capacity + 1, capacity_blocks=capacity)
    return raw


def dispatch(store, node, request, tokens):
    updates = InflightParser().parse(
        StatisticEvent("on_acquire", request_id=request, replica_id=node, prompt_ids=tuple(tokens)),
        node,
        schema=store.dispatch_schema(node),
    )
    (pin,) = (update for update in updates if isinstance(update, InflightBlockUpdate) and update.action == "pin")
    store.pin_request(request, node, pin.block_hashes, pin.schema, reservations=pin.reservations)


def strategy():
    return KVCacheAwareStrategy(
        load_threshold=0.9,
        overload_mode=OverloadMode.REMAINING_CAP,
        do_shortcut=True,
        tie_tolerance=0,
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("source", ["metadata", "metrics"])
@pytest.mark.parametrize("prefix_caching", [True, False])
def test_capacity_gate_and_sticky_use_the_current_replica(store, layout, source, prefix_caching, install_metadata):
    a, b = profile(layout, 99), profile(layout, 9)
    for raw in (a, b):
        raw["cache_config"]["enable_prefix_caching"] = prefix_caching
    if source == "metrics":
        for raw in (a, b):
            raw["cache_reservation_profile"].pop("capacity_blocks")
    install_metadata(store, "a", a)
    install_metadata(store, "b", b)
    # Reverse the scraped capacities when metadata is present to distinguish the two sources.
    scraped_blocks = {"a": 10, "b": 100} if source == "metadata" else {"a": 100, "b": 10}
    for node, configured in scraped_blocks.items():
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


@pytest.mark.cpu
@pytest.mark.level0
def test_disabled_prefix_cache_keeps_capacity_and_separate_request_reservations(store, layout, install_metadata):
    raw = profile(layout, 9)
    raw["cache_config"]["enable_prefix_caching"] = False
    install_metadata(store, "a", raw)
    assert store.capacity_blocks("a") == 9
    dispatch(store, "a", "first", range(17))
    dispatch(store, "a", "second", range(17))
    assert store.inflight_reserved_blocks("a") == 8
    assert store.request_block_need(17, 16, replica_id="a") == 4.25
    store.unpin_request("first")
    assert store.inflight_reserved_blocks("a") == 4
    store.unpin_request("second")
    assert store.inflight_reserved_blocks("a") == 0


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("a_prefix_caching", [True, False])
def test_disabled_replica_participates_in_capacity_selection(store, layout, a_prefix_caching, install_metadata):
    a, b = profile(layout, 9), profile(layout, 99)
    a["cache_config"]["enable_prefix_caching"] = a_prefix_caching
    b["cache_config"]["enable_prefix_caching"] = False
    install_metadata(store, "a", a)
    install_metadata(store, "b", b)
    dispatch(store, "b", "busy", range(4))
    store.refresh_metrics({"a": {MetricKey.INFLIGHT_COUNT: 0}, "b": {MetricKey.INFLIGHT_COUNT: 5}})
    assert strategy().score(list(range(17)), store, [ReplicaInfo("a"), ReplicaInfo("b")]) == [
        0,
        STICKY_TOP_SCORE,
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_missing_replica_capacity_does_not_borrow_sibling_metadata_or_metrics(store, layout, install_metadata):
    a, b = profile(layout, 99), profile(layout, 9)
    b["cache_reservation_profile"].pop("capacity_blocks")
    install_metadata(store, "a", a)
    install_metadata(store, "b", b)
    store.refresh_metrics({"a": {MetricKey.KV_CACHE_CAPACITY_BLOCKS: 999}})
    assert store.capacity_blocks("a") == 99  # own metadata beats sampled metric
    assert store.capacity_blocks("b") is None
    assert store.capacity_blocks("new-node") is None
    assert not strategy().is_overloaded(store, ReplicaInfo("b"))


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("change", ["block_size", "hash_unit", "prefix_policy", "model"])
def test_conflicting_replica_is_disabled_without_mutating_existing_replica(store, layout, change, install_metadata):
    a = layout((("full_attention", 4), ("sliding_window_mla", 4)), unit=4, window=8)
    a["cache_reservation_profile"].update(num_blocks=100, capacity_blocks=99)
    install_metadata(store, "a", a)
    dispatch(store, "a", "old-request", range(8))
    before = store.cache_registry_state("a")
    held_before = store.inflight_reserved_blocks("a")
    b = deepcopy(a)
    if change == "block_size":
        b["groups"][0]["block_size"] = 8
    elif change == "hash_unit":
        b["native_hash_config"]["unit"] = 2
    elif change == "prefix_policy":
        b["groups"][1].update(prefix_cacheable=False, prefix_replay_tokens=4)
    else:
        b["model_config"]["model"] = "different-model"
    assert not store.refresh_replica_metadata("b", b)
    assert store.cache_registry_state("b").reason == "incompatible_replica_layout"
    assert store.cache_registry_state("b").registry is None
    assert store.cache_registry_state("a") is before
    assert store.capacity_blocks("a") == 99
    assert store.inflight_reserved_blocks("a") == held_before
    store.unpin_request("old-request")
    assert store.inflight_reserved_blocks("a") == 0


@pytest.mark.cpu
@pytest.mark.level0
def test_optional_metadata_details_and_capacities_do_not_make_layouts_conflict(store, layout, install_metadata):
    a, b = profile(layout, 99), profile(layout, 9)
    b["groups"][0].update(prefix_cacheable=True, prefix_replay_tokens=0)
    a["model_config"]["model"] = None
    b["native_hash_config"].update(root_hash_hex="22" * 32)
    install_metadata(store, "a", a)
    state = install_metadata(store, "b", b)
    assert state.status == "registered"
    assert store.capacity_blocks("b") == 9


@pytest.mark.cpu
@pytest.mark.level0
def test_unavailable_replica_does_not_borrow_a_valid_peers_layout(store, layout, parse, stored, install_metadata):
    missing = {"status": "unsupported", "reason": "metadata_method_unavailable"}
    b = profile(layout, 9)
    install_metadata(store, "a", missing)
    state = install_metadata(store, "b", b)
    assert state.status == "registered"
    assert store.capacity_blocks("a") is None
    assert store.capacity_blocks("b") == 9
    event = stored(tokens=list(range(8)))
    event.pop("group_idx")
    event.pop("kv_cache_spec_kind")
    store.apply_kv_update(KVCacheUpdate("a", events=parse([event], "a")))
    assert store.per_replica_block_counts() == {}
    assert store.dispatch_schema("a") is None
    store.apply_kv_update(KVCacheUpdate("b", events=parse([event], "b")))
    assert store.per_replica_block_counts() == {"b": 2}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "invalid,prefix_caching",
    [
        ("malformed", True),
        ("malformed", False),
        ("unsupported", False),
        ("unsupported_private", False),
        ("parallel", False),
    ],
)
def test_explicitly_unusable_metadata_cannot_use_peer_layout_or_own_capacity_metric(
    store, layout, invalid, prefix_caching, install_metadata
):
    a, b = profile(layout, 99), profile(layout, 9)
    b["cache_config"]["enable_prefix_caching"] = prefix_caching
    if invalid == "malformed":
        b["groups"][0]["block_size"] = 0
    elif invalid == "unsupported":
        b["groups"][0]["kind"] = "unsupported_attention"
    elif invalid == "unsupported_private":
        b["groups"][0].update(kind="unsupported_attention", prefix_cacheable=False, prefix_replay_tokens=0)
    else:
        b["parallel_config"]["data_parallel_size"] = 2
    install_metadata(store, "a", a)
    install_metadata(store, "b", b)
    store.refresh_metrics({"b": {MetricKey.KV_CACHE_CAPACITY_BLOCKS: 9}})
    assert store.capacity_blocks("b") is None
    assert store.dispatch_schema("b") is None
    assert store.estimate_prefixes(["b"], [], 5)["b"].estimated_hit_tokens == 0


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("prefix_caching,needs", [(True, (2.25, 4.25)), (False, (6.25, 8.25))])
def test_private_reservations_and_need_come_from_each_replica_profile(
    store, layout, prefix_caching, needs, install_metadata
):
    a = layout()
    b = deepcopy(a)
    for raw in (a, b):
        raw["cache_config"]["enable_prefix_caching"] = prefix_caching
    a["cache_reservation_profile"]["groups"][1]["max_blocks_per_request"] = 2
    b["cache_reservation_profile"]["groups"][1]["max_blocks_per_request"] = 4
    install_metadata(store, "a", a)
    install_metadata(store, "b", b)
    dispatch(store, "a", "a-request", range(17))
    dispatch(store, "b", "b-request", range(17))
    assert store.inflight_reserved_blocks("a") == 6
    assert store.inflight_reserved_blocks("b") == 8
    assert store.request_block_need(17, 16, replica_id="a") == needs[0]
    assert store.request_block_need(17, 16, replica_id="b") == needs[1]


@pytest.mark.cpu
@pytest.mark.level0
def test_disabled_prefix_cache_with_unknown_private_budget_uses_inflight_fallback(store, layout, install_metadata):
    raw = layout()
    raw["cache_config"]["enable_prefix_caching"] = False
    raw["cache_reservation_profile"] = None
    install_metadata(store, "a", raw)
    install_metadata(store, "b", raw)
    dispatch(store, "a", "busy", range(17))
    store.refresh_metrics(
        {
            "a": {MetricKey.KV_CACHE_CAPACITY_BLOCKS: 99, MetricKey.INFLIGHT_COUNT: 5},
            "b": {MetricKey.KV_CACHE_CAPACITY_BLOCKS: 9, MetricKey.INFLIGHT_COUNT: 0},
        }
    )
    assert store.capacity_blocks("a") == 99
    assert store.inflight_reserved_blocks("a") is None
    assert store.request_block_need(17, 17, replica_id="a") is None
    assert strategy().score(list(range(17)), store, [ReplicaInfo("a"), ReplicaInfo("b")]) == [0, STICKY_TOP_SCORE]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("prefix_caching", [True, False])
def test_zero_capacity_breaks_sticky_even_when_private_budget_is_unknown(
    store, layout, prefix_caching, install_metadata
):
    raw = layout()
    raw["cache_config"]["enable_prefix_caching"] = prefix_caching
    raw["cache_reservation_profile"] = None
    install_metadata(store, "a", raw)
    store.refresh_metrics({"a": {MetricKey.KV_CACHE_CAPACITY_BLOCKS: 0}})
    assert store.inflight_reserved_blocks("a") is None
    assert strategy().is_overloaded(store, ReplicaInfo("a"))


@pytest.mark.cpu
@pytest.mark.level0
def test_matching_metadata_refresh_preserves_cache_and_request_budget(store, layout, install_metadata):
    raw = profile(layout, 9)
    install_metadata(store, "a", raw)
    dispatch(store, "a", "busy", range(17))
    schema = store.dispatch_schema("a")
    assert not store.refresh_replica_metadata("a", deepcopy(raw))
    assert store.dispatch_schema("a") == schema
    assert store.inflight_reserved_blocks("a") == 4
    assert store.estimate_prefixes(["a"], store.get_marker("busy")["hashes"], 17)["a"].estimated_hit_tokens == 16


@pytest.mark.cpu
@pytest.mark.level0
def test_failed_metadata_refresh_keeps_verified_capacity_and_cache(store, layout, install_metadata):
    raw = profile(layout, 9)
    install_metadata(store, "a", raw)
    dispatch(store, "a", "busy", range(17))
    assert not store.refresh_replica_metadata("a", {"status": "error", "reason": "rpc_timeout"})
    assert store.capacity_blocks("a") == 9
    assert store.inflight_reserved_blocks("a") == 4
    assert store.estimate_prefixes(["a"], store.get_marker("busy")["hashes"], 17)["a"].estimated_hit_tokens == 16
    assert not store.metadata_status("a")["ready"]
    assert store.metadata_status("a")["reason"] == "rpc_timeout"


@pytest.mark.cpu
@pytest.mark.level0
def test_request_dispatched_without_metadata_remains_unknown_until_release(store, layout, install_metadata):
    install_metadata(store, "a", {"status": "not_ready"})
    assert store.dispatch_schema("a") is None
    dispatch(store, "a", "unknown", range(17))
    store.refresh_replica_metadata("a", profile(layout, 9))
    assert store.capacity_blocks("a") == 9
    assert store.inflight_reserved_blocks("a") is None
    store.unpin_request("unknown")
    assert store.inflight_reserved_blocks("a") == 0


@pytest.mark.cpu
@pytest.mark.level0
def test_changed_engine_root_rebuilds_only_its_cache_and_keeps_request_budgets(store, layout, install_metadata):
    raw = profile(layout, 9)
    install_metadata(store, "a", raw)
    install_metadata(store, "b", raw)
    dispatch(store, "a", "a-busy", range(17))
    dispatch(store, "b", "b-busy", range(17))
    before = store.dispatch_schema("a")
    changed = deepcopy(raw)
    changed["native_hash_config"]["root_hash_hex"] = "11" * 32
    assert store.refresh_replica_metadata("a", changed)
    assert store.dispatch_schema("a") == before
    assert (store.inflight_reserved_blocks("a"), store.inflight_reserved_blocks("b")) == (4, 4)
    hashes = store.get_marker("b-busy")["hashes"]
    store.mark_kv_replica_ready("a")
    estimates = store.estimate_prefixes(["a", "b"], hashes, 17)
    assert estimates["a"].estimated_hit_tokens == 0
    assert estimates["b"].estimated_hit_tokens == 16


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("errors", [[], ["timeout"]])
def test_malformed_metadata_errors_preserve_layout_and_cached_prefix(store, layout, install_metadata, errors):
    raw = profile(layout, 9)
    install_metadata(store, "a", raw)
    dispatch(store, "a", "request", range(17))
    hashes = store.get_marker("request")["hashes"]
    before = store.dispatch_schema("a")

    changed = layout((("full_attention", 8),), unit=4)
    changed["metadata_errors"] = errors
    assert not store.refresh_replica_metadata("a", changed)
    assert store.metadata_status("a")["reason"] == "invalid_metadata_errors"
    assert store.dispatch_schema("a") == before
    assert store.capacity_blocks("a") == 9
    assert store.estimate_prefixes(["a"], hashes, 17)["a"].estimated_hit_tokens == 16

    changed["metadata_errors"] = {}
    assert store.refresh_replica_metadata("a", changed)
    assert store.metadata_status("a")["gated"]
    assert store.estimate_prefixes(["a"], hashes, 17)["a"].estimated_hit_tokens == 0


@pytest.mark.cpu
@pytest.mark.level0
def test_partial_refresh_does_not_reuse_private_budget_after_layout_changes(store, layout, install_metadata):
    raw = layout()
    raw["cache_reservation_profile"].update(capacity_blocks=99)
    install_metadata(store, "a", raw)
    changed = deepcopy(raw)
    changed["groups"][1]["block_size"] = 16
    changed["cache_reservation_profile"] = None
    changed["metadata_errors"] = {"cache_reservation_profile": {"reason": "rpc_timeout"}}
    store.refresh_replica_metadata("a", changed)
    assert store.request_block_need(17, replica_id="a") is None
    assert store.inflight_reserved_blocks("a") is None
    assert store.metadata_status("a")["reason"] == "rpc_timeout"


@pytest.mark.cpu
@pytest.mark.level0
def test_request_hashes_recompute_after_layout_changes(store, layout, install_metadata):
    install_metadata(store, "a", layout(unit=4))
    prompt = list(range(17))
    assert len(resolve_prefix_hashes(prompt, "request", store)) == 4
    install_metadata(store, "a", layout(unit=2))
    resolved = resolve_prefix_hashes(prompt, "request", store)
    expected, _ = get_prefix_hashes_incremental(prompt, 2, 0, 0)
    assert resolved == list(map(str, expected))


@pytest.mark.cpu
@pytest.mark.level0
def test_retention_load_is_distinct_from_native_usage(store, layout, parse, stored, install_metadata):
    for node in ("a", "b"):
        install_metadata(store, node, layout())
        events = parse([stored(), stored(1, "mamba", 8, hashes=["r16"])], node)
        store.apply_kv_update(KVCacheUpdate(node, events))
    store.refresh_metrics(
        {
            "a": {MetricKey.NUM_GPU_BLOCKS: 5, MetricKey.KV_CACHE_USAGE_PERC: 0.0},
            "b": {MetricKey.NUM_GPU_BLOCKS: 100, MetricKey.KV_CACHE_USAGE_PERC: 1.0},
        }
    )
    assert store.kv_cache_load("a") == 1.0
    assert store.kv_cache_load("b") == 0.05
    store.clear_kv_resident_blocks("a")
    assert store.kv_cache_load("a") == 0.0
    assert store.kv_cache_load("b") == 0.05


@pytest.mark.cpu
@pytest.mark.level0
def test_reordered_groups_preserve_outstanding_reservations(store, layout, install_metadata):
    raw = layout()
    install_metadata(store, "a", raw)
    dispatch(store, "a", "busy", range(17))
    schema = store.dispatch_schema("a")
    held = store.inflight_reserved_blocks("a")
    assert held == 6

    reordered = deepcopy(raw)
    reordered["groups"].reverse()
    reordered["cache_reservation_profile"]["groups"].reverse()
    store.refresh_replica_metadata("a", reordered)
    assert store.dispatch_schema("a") == schema
    assert store.inflight_reserved_blocks("a") == held

    changed = deepcopy(reordered)
    changed["cache_reservation_profile"]["groups"][0]["max_blocks_per_request"] += 1
    store.refresh_replica_metadata("a", changed)
    assert store.dispatch_schema("a") != schema
    assert store.inflight_reserved_blocks("a") is None


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("length", [1, 2])
def test_reported_predicted_and_reserved_blocks_use_the_same_lookahead(
    store, layout, install_metadata, parse, stored, removed, length
):
    raw = layout((("full_attention", 4),), unit=2, alignment=4)
    raw["native_hash_config"]["lookahead_tokens"] = length
    for node in ("reported", "predicted"):
        install_metadata(store, node, raw)

    tokens = list(range(4 + length))
    other = [*tokens[:-1], 99]
    assert store.estimate_prefixes(["reported"], [], len(tokens) - 1)["reported"].lookup_limit_tokens == 2
    event = stored(block=4, tokens=tokens[:4], hashes=["block"], lookahead_token_ids=[tokens[4:]])
    store.apply_kv_update(KVCacheUpdate("reported", parse([event], "reported")))
    dispatch(store, "predicted", "first", tokens)

    def hits(prompt):
        hashes = resolve_prefix_hashes(prompt, "session", store)
        estimates = store.estimate_prefixes(["reported", "predicted"], hashes, len(prompt))
        expected_limit = max(0, (len(prompt) - length) // 2 * 2)
        assert all(estimate.lookup_limit_tokens == expected_limit for estimate in estimates.values())
        return [entry.estimated_hit_tokens for entry in estimates.values()]

    assert hits(tokens) == [4, 4]
    assert hits(other) == [0, 0]
    assert hits(tokens[:-1]) == [0, 0]
    dispatch(store, "predicted", "repeat", tokens)
    assert store.inflight_reserved_blocks("predicted") == 1
    dispatch(store, "predicted", "fork", other)
    assert store.inflight_reserved_blocks("predicted") == 2
    # A full physical block with too few following tokens cannot share a reservation.
    dispatch(store, "predicted", "short-a", tokens[:4])
    dispatch(store, "predicted", "short-b", tokens[:4])
    assert store.inflight_reserved_blocks("predicted") == 4
    store.apply_kv_update(KVCacheUpdate("reported", parse([removed(0, ["block"])], "reported")))
    assert hits(tokens) == [0, 4]
