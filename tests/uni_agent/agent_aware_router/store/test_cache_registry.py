"""Cache layouts describe group semantics, independently of cache residency."""

from copy import deepcopy

import pytest

from uni_agent.agent_aware_router.store.cache_registry import build_cache_registry
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_required_groups_gcd_and_input_isolation(metadata):
    raw = metadata()
    # PP alone does not reject a resolved layout or change its hash unit.
    raw["parallel_config"]["pipeline_parallel_size"] = 2
    raw["kv_cache_capacity_tokens"] = 20480
    state = build_cache_registry(raw)
    assert state.status == "registered"
    registry = state.registry
    assert registry.local_hash_unit == 32
    assert [g.group_idx for g in registry.required_groups] == [0, 3]
    raw["groups"][0]["block_size"] = 2
    raw["kv_cache_capacity_tokens"] = 1
    assert registry.group(0).block_size == 64
    assert registry.kv_cache_capacity_tokens == 20480


@pytest.mark.parametrize(
    "kind,match_kind",
    [
        ("full_attention", "full"),
        ("mla_attention", "full"),
        ("sink_full_attention", "full"),
        ("sliding_window", "sliding"),
        ("sliding_window_mla", "sliding"),
        ("chunked_local_attention", "chunked"),
        ("mamba", "mamba"),
    ],
)
def test_all_seven_kinds_keep_their_parameters(kind, match_kind, metadata):
    raw = metadata()
    window = 257 if match_kind == "sliding" else None
    raw["groups"] = [{"group_idx": 4, "kind": kind, "block_size": 64, "sliding_window": window}]
    state = build_cache_registry(raw)
    assert state.status == "registered"
    group = state.registry.group(4)
    assert group.kind == kind
    assert group.match_kind == match_kind
    assert group.sliding_window == window
    assert group.attention_chunk_size == (8192 if match_kind == "chunked" else None)


@pytest.mark.parametrize(
    "mode,field,expected",
    [("align", "prefix_match_unit", 16), ("align", "hash_block_size", 16), ("all", None, 192)],
)
def test_hash_granularity_follows_group_alignment(mode, field, expected, metadata):
    raw = metadata()
    raw["cache_config"]["mamba_cache_mode"] = mode
    if field:
        raw["cache_config"][field] = 16
    assert build_cache_registry(raw).registry.local_hash_unit == expected


def test_disabled_prefix_caching_does_not_choose_a_hash_unit(metadata):
    raw = metadata()
    raw["cache_config"].update(enable_prefix_caching=False, mamba_cache_mode="all")
    state = build_cache_registry(raw)
    assert state.status == "disabled"
    assert state.registry is None


# Representative contract failures: non-positive and incompatible units.
# These are malformed metadata inputs, not native vLLM startup scenarios.
@pytest.mark.parametrize("value", [0, 3])
def test_invalid_explicit_unit_has_no_partial_registry(value, metadata):
    raw = metadata()
    raw["cache_config"]["prefix_match_unit"] = value
    state = build_cache_registry(raw)
    assert state.registry is None
    assert "prefix_match_unit" in state.reason


@pytest.mark.parametrize(
    "groups,reason",
    [
        ([{"group_idx": True, "kind": "full_attention", "block_size": 16}], "group_idx"),
        ([{"group_idx": 0, "kind": "full_attention", "block_size": 0}], "block_size"),
        ([{"group_idx": 0, "kind": "sliding_window", "block_size": 16}], "sliding_window"),
        ([{"group_idx": 0, "kind": "encoder_only_attention", "block_size": 16}], "no_prefix_cache_groups"),
    ],
)
def test_incomplete_or_unsupported_groups_are_not_installed(groups, reason, metadata):
    raw = metadata()
    raw["groups"] = groups
    state = build_cache_registry(raw)
    assert state.registry is None
    assert reason in state.reason


def test_duplicate_ids_and_missing_chunk_size(metadata):
    raw = metadata()
    raw["groups"].append(deepcopy(raw["groups"][0]))
    assert "duplicate_group_idx" in build_cache_registry(raw).reason
    raw = metadata()
    raw["groups"][0]["kind"] = "chunked_local_attention"
    raw["model_config"]["attention_chunk_size"] = None
    assert "attention_chunk_size" in build_cache_registry(raw).reason


@pytest.mark.parametrize("kind", ["sliding_window", "unknown"])
def test_native_non_cacheable_groups_are_retained_but_not_required(kind, metadata, caplog):
    raw = metadata()
    raw["groups"].append(
        dict(group_idx=8, kind=kind, block_size=16, sliding_window=128, prefix_cacheable=False, prefix_replay_tokens=0)
    )
    state = build_cache_registry(raw)
    assert state.status == "registered"
    assert state.registry.local_hash_unit == 32
    assert [g.group_idx for g in state.registry.required_groups] == [0, 3]
    auxiliary = state.registry.group(8)
    assert auxiliary.prefix_cacheable is False
    assert not auxiliary.participates
    assert state.registry.prefix_replay_tokens == 0
    del raw["groups"][-1]["prefix_cacheable"]
    del raw["groups"][-1]["prefix_replay_tokens"]
    state = build_cache_registry(raw)
    assert state.status == "registered"
    if kind == "unknown":
        assert not state.registry.group(8).participates
        assert state.registry.group(8).prefix_cacheable is False
        assert state.registry.local_hash_unit == 32
        assert "unknown groups [8]" in caplog.text
        assert "prefix_cacheable" not in raw["groups"][-1]
        raw["groups"][-1].update(prefix_cacheable=True, prefix_replay_tokens=0)
        assert build_cache_registry(raw).status == "unsupported"
        del raw["groups"][-1]["prefix_replay_tokens"]
        assert build_cache_registry(raw).status == "invalid"
    else:
        assert state.registry.group(8).participates


@pytest.mark.parametrize(
    "policy,reason",
    [
        ({"prefix_cacheable": 0, "prefix_replay_tokens": 0}, "prefix_cacheable"),
        ({"prefix_cacheable": False}, "prefix_replay_tokens"),
        ({"prefix_replay_tokens": 128}, "prefix_cacheable"),
        ({"prefix_cacheable": False, "prefix_replay_tokens": -1}, "prefix_replay_tokens"),
        ({"prefix_cacheable": False, "prefix_replay_tokens": True}, "prefix_replay_tokens"),
        ({"prefix_cacheable": True, "prefix_replay_tokens": 128}, "cacheable_prefix_replay"),
    ],
)
def test_malformed_prefix_policy_never_creates_a_partial_registry(policy, reason, metadata):
    raw = metadata()
    raw["groups"][0].update(policy)
    state = build_cache_registry(raw)
    assert state.registry is None
    assert reason in state.reason


def test_replay_windows_must_agree_and_some_group_must_participate(metadata):
    raw = metadata()
    for idx, replay in [(8, 128), (9, 256)]:
        raw["groups"].append(
            dict(
                group_idx=idx,
                kind="sliding_window_mla",
                block_size=16,
                prefix_cacheable=False,
                prefix_replay_tokens=replay,
            )
        )
    assert build_cache_registry(raw).reason == "inconsistent_prefix_replay_tokens"
    raw["groups"] = raw["groups"][-1:]
    assert build_cache_registry(raw).reason == "no_prefix_cache_groups"


@pytest.mark.parametrize("key", ["data_parallel_size", "decode_context_parallel_size", "prefill_context_parallel_size"])
def test_parallel_layouts_are_not_silently_assumed_supported(key, metadata):
    raw = metadata()
    raw["parallel_config"][key] = 2
    assert build_cache_registry(raw).status == "unsupported"
    raw["parallel_config"][key] = None
    assert build_cache_registry(raw).registry is None


def test_unavailable_metadata_does_not_create_a_registry():
    assert build_cache_registry({"status": "not_ready"}).status == "unavailable"
    assert build_cache_registry({"status": "error", "reason": "rpc_timeout"}).reason == "rpc_timeout"
    assert build_cache_registry({"status": []}).status == "invalid"
    assert build_cache_registry(None).status == "invalid"


def test_store_owns_registry_separately_from_event_granularity_and_residency(metadata):
    store = KVCacheStore()
    store.block_size = 64
    store.add_blocks("replica-a", ["hash-a"])
    state = store.configure_registry(metadata())
    assert store.registry_state is state
    assert state.registry.local_hash_unit == 32
    assert store.block_size == 64
    assert store.replicas_by_block == {"hash-a": {"replica-a"}}
    store.clear_replica("replica-a")
    assert store.registry_state is state
    invalid = metadata()
    invalid["groups"] = []
    state = store.configure_registry(invalid)
    assert state.registry is None
    assert store.registry_state.reason == "missing_groups"
    # An explicitly invalid/unsupported layout must not be treated as Full.
    assert store.registry_configured
    assert store.estimate_prefixes(["replica-a"], ["hash-a"], 64)["replica-a"].status == "invalid"
    invalid["groups"] = [{"group_idx": 0, "kind": "unknown", "block_size": 64}]
    store.configure_registry(invalid)
    assert store.registry_configured
    assert store.estimate_prefixes(["replica-a"], ["hash-a"], 64)["replica-a"].status == "unsupported"
