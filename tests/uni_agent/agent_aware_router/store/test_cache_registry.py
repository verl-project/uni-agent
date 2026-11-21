"""Cache layouts describe group semantics, independently of cache residency."""

from copy import deepcopy

import pytest

from uni_agent.agent_aware_router.collectors.parse.base import KVCacheUpdate
from uni_agent.agent_aware_router.store.cache_registry import build_cache_registry
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore


@pytest.mark.cpu
@pytest.mark.level0
def test_native_unit_and_input_isolation(metadata):
    raw = metadata()
    # PP alone does not reject a resolved layout or change its hash unit.
    raw["parallel_config"]["pipeline_parallel_size"] = 2
    state = build_cache_registry(raw)
    assert state.status == "registered"
    registry = state.registry
    assert registry.local_hash_unit == 32
    assert [g.group_idx for g in registry.required_groups] == [0, 3]
    raw["groups"][0]["block_size"] = 2
    assert registry.group(0).block_size == 64


@pytest.mark.cpu
@pytest.mark.level0
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


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("mode,unit,alignment", [("align", 16, 192), ("all", 32, 192)])
def test_hash_granularity_uses_native_unit_and_separate_alignment(mode, unit, alignment, metadata):
    raw = metadata()
    raw["cache_config"].update(mamba_cache_mode=mode, prefix_match_unit=3, hash_block_size=5)
    raw["native_hash_config"].update(unit=unit, hit_alignment=alignment)
    registry = build_cache_registry(raw).registry
    assert registry.local_hash_unit == unit
    assert registry.prefix_match_alignment == alignment


@pytest.mark.cpu
@pytest.mark.level0
def test_disabled_prefix_caching_does_not_choose_a_hash_unit(metadata):
    raw = metadata()
    raw["cache_config"].update(enable_prefix_caching=False, mamba_cache_mode="all")
    state = build_cache_registry(raw)
    assert state.status == "disabled"
    assert state.registry is None


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "alignment,reason",
    [(None, "missing_hit_alignment"), (0, "invalid_hit_alignment"), (48, "incompatible_hit_alignment")],
)
def test_unusable_hit_alignment_disables_matching_but_keeps_reservations(alignment, reason, layout):
    raw = layout((("full_attention", 32), ("sliding_window", 64)), unit=32)
    raw["native_hash_config"]["hit_alignment"] = alignment
    state = build_cache_registry(raw)
    assert state.registry is None
    assert state.reason == reason
    assert state.reservation_groups


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("prefix_cacheable", [True, False])
def test_mamba_all_requires_representable_sliding_block_ends(layout, prefix_cacheable):
    raw = layout((("full_attention", 8), ("sliding_window", 4), ("mamba", 8)), unit=8, window=9, mode="all")
    raw["groups"][1].update(prefix_cacheable=prefix_cacheable, prefix_replay_tokens=0)
    raw["cache_reservation_profile"].update(num_blocks=65, capacity_blocks=64)
    state = build_cache_registry(raw)

    assert state.capacity_blocks == 64
    assert [spec.group_idx for spec in state.reservation_groups] == [0, 1, 2]
    if prefix_cacheable:
        assert state.registry is None
        assert state.reason == "incompatible_native_hash_unit"
    else:
        assert state.registry is not None
        assert [group.group_idx for group in state.registry.required_groups] == [0, 2]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "policy", [{"drop_eagle_group_ids": [1]}, {"codec_unavailable_reason": "unsupported_lookahead"}]
)
def test_unknown_or_hybrid_eagle_drop_preserves_capacity(layout, policy):
    raw = layout()
    raw["native_hash_config"].update(lookahead_tokens=0, **policy)
    raw["cache_reservation_profile"]["capacity_blocks"] = 64
    state = build_cache_registry(raw)
    assert state.registry is None and state.reservation_groups and state.capacity_blocks == 64


# Representative contract failures: non-positive and incompatible units.
# These are malformed metadata inputs, not native vLLM startup scenarios.
@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("value", [0, 3])
def test_invalid_native_unit_has_no_partial_registry(value, metadata):
    raw = metadata()
    raw["native_hash_config"]["unit"] = value
    state = build_cache_registry(raw)
    assert state.registry is None
    assert "native_hash_unit" in state.reason


@pytest.mark.cpu
@pytest.mark.level0
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


@pytest.mark.cpu
@pytest.mark.level0
def test_duplicate_ids_and_missing_chunk_size(metadata):
    raw = metadata()
    raw["groups"].append(deepcopy(raw["groups"][0]))
    assert "duplicate_group_idx" in build_cache_registry(raw).reason
    raw = metadata()
    raw["groups"][0]["kind"] = "chunked_local_attention"
    raw["model_config"]["attention_chunk_size"] = None
    assert "attention_chunk_size" in build_cache_registry(raw).reason


@pytest.mark.cpu
@pytest.mark.level0
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
    if kind == "sliding_window":
        raw["native_hash_config"]["unit"] = 16
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


@pytest.mark.cpu
@pytest.mark.level0
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


@pytest.mark.cpu
@pytest.mark.level0
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


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("key", ["data_parallel_size", "decode_context_parallel_size", "prefill_context_parallel_size"])
def test_parallel_layouts_are_not_silently_assumed_supported(key, metadata):
    raw = metadata()
    raw["parallel_config"][key] = 2
    assert build_cache_registry(raw).status == "unsupported"
    raw["parallel_config"][key] = None
    assert build_cache_registry(raw).registry is None


@pytest.mark.cpu
@pytest.mark.level0
def test_unavailable_metadata_does_not_create_a_registry():
    assert build_cache_registry({"status": "not_ready"}).status == "unavailable"
    assert build_cache_registry({"status": "error", "reason": "rpc_timeout"}).reason == "rpc_timeout"
    assert build_cache_registry({"status": []}).status == "invalid"
    assert build_cache_registry(None).status == "invalid"


@pytest.mark.cpu
@pytest.mark.level0
def test_store_owns_registry_separately_from_event_granularity_and_residency(metadata, stored, parse, install_metadata):
    store = KVCacheStore()
    state = install_metadata(store, "replica-a", metadata())
    events = parse([stored(block=64, tokens=list(range(64)), hashes=["h64"])])
    store.apply_update(KVCacheUpdate("replica-a", events=events))
    assert store.registry_state("replica-a") is state
    assert state.registry.local_hash_unit == 32
    assert store.per_replica_block_counts() == {"replica-a": 1}
    store.clear_replica("replica-a")
    assert store.registry_state("replica-a") is state
    assert store.per_replica_block_counts() == {}
    invalid = metadata()
    invalid["groups"] = []
    assert not store.refresh_replica_metadata("replica-a", invalid)
    assert store.registry_state("replica-a") is state
    assert store.metadata_status("replica-a")["reason"] == "missing_groups"
    invalid["groups"] = [
        {"group_idx": 0, "kind": "unknown", "block_size": 64, "prefix_cacheable": True, "prefix_replay_tokens": 0}
    ]
    assert not store.refresh_replica_metadata("replica-a", invalid)
    assert store.registry_state("replica-a") is state
    assert "unsupported_kind" in store.metadata_status("replica-a")["reason"]


@pytest.mark.cpu
@pytest.mark.level0
def test_malformed_optional_budget_cannot_suppress_private_group_cost(layout):
    raw = layout()
    profile = raw["cache_reservation_profile"]
    profile["capacity_blocks"] = 99
    assert build_cache_registry(raw).capacity_blocks == 99
    profile["groups"].append(dict(profile["groups"][1]))  # duplicate group must not overwrite a budget
    state = build_cache_registry(raw)
    assert state.registry is not None  # cache matching remains usable
    assert state.capacity_blocks is None
    assert state.reservation_groups[1].private_blocks is None


@pytest.mark.cpu
@pytest.mark.level0
def test_native_unit_includes_non_matching_group_without_deriving_from_groups(layout):
    raw = layout((("full_attention", 32), ("sliding_window", 16), ("encoder_only_attention", 8)), unit=8, alignment=32)
    raw["vllm_version"] = "0.23.0"
    raw["native_hash_config"]["vllm_version"] = "0.23.0"
    state = build_cache_registry(raw)
    assert state.registry.local_hash_unit == 8
    assert [group.group_idx for group in state.registry.required_groups] == [0, 1]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("profile", [None, {}, {"schema_version": 1, "unit": 0}])
def test_missing_native_unit_keeps_capacity_and_request_budget_without_matching(layout, profile):
    raw = layout((("full_attention", 4),), unit=2)
    raw["native_hash_config"] = profile
    raw["cache_reservation_profile"]["capacity_blocks"] = 9
    state = build_cache_registry(raw)
    assert state.registry is None
    assert state.reason in {"missing_native_hash_unit", "invalid_native_hash_unit"}
    assert state.capacity_blocks == 9
    assert state.reservation_groups[0].block_size == 4


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("length", [None, -1, True, "missing"])
def test_ambiguous_new_hash_lookahead_keeps_capacity_without_matching(layout, length):
    raw = layout((("full_attention", 4),), unit=2)
    profile = raw["native_hash_config"]
    profile["vllm_version"] = "0.33.0"
    raw["cache_reservation_profile"]["capacity_blocks"] = 24
    if length != "missing":
        profile["lookahead_tokens"] = length
    state = build_cache_registry(raw)
    assert state.registry is None
    assert state.reason in {"missing_hash_lookahead", "invalid_hash_lookahead"}
    assert state.capacity_blocks == 24
