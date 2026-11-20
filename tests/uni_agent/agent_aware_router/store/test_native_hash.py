"""Native Full events restore sparse identities without copying residency."""

import pickle
from copy import deepcopy
from types import SimpleNamespace

import pytest

from uni_agent.agent_aware_router.collectors.parse.vllm.native_hash import VLLMHashCodec
from uni_agent.agent_aware_router.store.cache_registry import build_cache_registry
from uni_agent.agent_aware_router.store.group_cache import ReplicaCacheIndex
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.store.prefix_identity import PrefixIdentityIndex, PrefixIdentityMismatch
from uni_agent.agent_aware_router.types.kv_cache import KVCacheUpdate
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental


@pytest.fixture
def native(monkeypatch, request):
    vllm = pytest.importorskip("vllm")
    from vllm import envs
    from vllm.utils.hashing import sha256
    from vllm.v1.core import kv_cache_utils

    int_hashes = getattr(request, "param", True)
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setattr(envs, "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", int_hashes)
    monkeypatch.setattr(kv_cache_utils, "NONE_HASH", None, raising=False)
    monkeypatch.setattr(kv_cache_utils, "_NONE_HASH_SEED", None, raising=False)
    kv_cache_utils.init_none_hash(sha256)
    root = bytes(kv_cache_utils.NONE_HASH)

    def chain(parent=root):
        digests = []
        for start in range(0, 16, 4):
            parent = kv_cache_utils.hash_block_tokens(sha256, parent, list(range(start, start + 4)), None)
            digests.append(bytes(parent))
        return SimpleNamespace(
            digests=digests,
            hashes=[kv_cache_utils.maybe_convert_block_hash(digest) for digest in digests],
            # Use the installed version's native encoding: concatenated bytes
            # in 0.23, endpoint bytes in 0.29, or converted integer identities.
            full_hashes=[
                kv_cache_utils.maybe_convert_block_hash(digest)
                for digest in kv_cache_utils.BlockHashListWithBlockSize(digests, 4, 8)
            ],
        )

    def profile(parent=root):
        return {
            "schema_version": 1,
            "vllm_version": vllm.__version__,
            "algorithm": "sha256",
            "root_hash_hex": parent.hex(),
            "unit": 4,
            "hit_alignment": 4,
            "pickle_protocol": pickle.HIGHEST_PROTOCOL,
            "int_hashes": int_hashes,
            "lookahead_tokens": 0,
        }

    return SimpleNamespace(root=root, chain=chain, profile=profile, utils=kv_cache_utils, sha256=sha256)


@pytest.fixture
def native_metadata(layout, native):
    metadata = layout((("sliding_window", 4), ("full_attention", 8)), unit=4, window=9)
    metadata["native_hash_config"] = native.profile()
    return metadata


@pytest.fixture
def store(native_metadata, install_metadata):
    value = KVCacheStore()
    value.set_native_hash_codec_factory(VLLMHashCodec.from_metadata)
    install_metadata(value, "a", native_metadata)
    install_metadata(value, "b", native_metadata)
    return value, native_metadata


@pytest.fixture
def events(stored, parse, native):
    chain = native.chain()
    full = parse([stored(1, "full_attention", 8, hashes=chain.full_hashes)])
    sparse = parse([stored(0, "sliding_window", 4, hashes=chain.hashes[1:], kv_cache_spec_sliding_window=9)])
    return full, sparse


def hit(cache):
    hashes, _ = get_prefix_hashes_incremental(list(range(17)), 4, 0, 0)
    return cache.estimate([str(h) for h in hashes], 17).estimated_hit_tokens


def direct_cache(metadata):
    registry = build_cache_registry(metadata).registry
    codec = VLLMHashCodec.from_metadata(metadata, 4).codec
    assert codec is not None
    index = PrefixIdentityIndex(codec)
    return ReplicaCacheIndex(registry, "a", native_hashes=index), index


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("full_node", ["a", "b"])
def test_full_event_identity_is_shared_across_replica_caches(full_node, store, events):
    value, _ = store
    full, sparse = events
    value.apply_update(KVCacheUpdate(replica_id="a", events=sparse))
    a = value.group_caches["a"]
    assert hit(a) == 0 and a.pending
    value.apply_update(KVCacheUpdate(replica_id=full_node, events=full))
    assert not a.pending and len(a.by_group[0]) == 3
    assert hit(a) == (16 if full_node == "a" else 0)
    value.apply_update(KVCacheUpdate(replica_id="a", events=full))
    assert hit(a) == 16


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("elapsed", [59, 61], ids=["before-ttl", "after-ttl"])
def test_peer_can_only_restore_unexpired_pending_identity(elapsed, store, events, monkeypatch):
    value, _ = store
    full, sparse = events
    now = 0
    clock = SimpleNamespace(monotonic=lambda: now)
    monkeypatch.setattr("uni_agent.agent_aware_router.store.group_cache.time", clock)
    value.apply_update(KVCacheUpdate("a", events=sparse))
    a = value.group_caches["a"]
    assert a.pending

    now = elapsed
    value.apply_update(KVCacheUpdate("b", events=full))
    assert not a.pending
    assert len(a.by_group.get(0, {})) == (3 if elapsed < 60 else 0)
    value.apply_update(KVCacheUpdate("a", events=full))
    assert hit(a) == (16 if elapsed < 60 else 0)
    value.apply_update(KVCacheUpdate("a", events=sparse))
    assert hit(a) == 16


@pytest.mark.cpu
@pytest.mark.level0
def test_different_engine_roots_do_not_share_identities(
    store, events, native, monkeypatch, stored, parse, install_metadata
):
    value, metadata = store
    with monkeypatch.context() as other_engine:
        other_engine.setenv("PYTHONHASHSEED", "43")
        other_engine.setattr(native.utils, "NONE_HASH", None)
        other_engine.setattr(native.utils, "_NONE_HASH_SEED", None, raising=False)
        native.utils.init_none_hash(native.sha256)
        other_root = bytes(native.utils.NONE_HASH)
    assert other_root != native.root
    other_metadata = deepcopy(metadata)
    other_metadata["native_hash_config"] = native.profile(other_root)
    install_metadata(value, "a", metadata)
    install_metadata(value, "b", other_metadata)
    full, sparse = events
    value.apply_update(KVCacheUpdate(replica_id="a", events=sparse))
    other_full = parse([stored(1, "full_attention", 8, hashes=native.chain(other_root).full_hashes)], node="b")
    value.apply_update(KVCacheUpdate(replica_id="b", events=other_full))
    a, b = value.group_caches["a"], value.group_caches["b"]
    assert a.native_hashes is not None and b.native_hashes is not None
    assert a.native_hashes is not b.native_hashes
    assert a.pending and not a.by_group[0] and hit(a) == 0
    value.apply_update(KVCacheUpdate(replica_id="a", events=full))
    assert not a.pending and hit(a) == 16


@pytest.mark.cpu
@pytest.mark.level0
def test_restored_identity_preserves_copy_counts_and_clear(native_metadata, events, native, parse, removed):
    cache, _ = direct_cache(native_metadata)
    full, sparse = events
    cache.apply(full + sparse + sparse)
    assert hit(cache) == 16
    for expected in (16, 0):
        cache.apply(parse([removed(0, [native.chain().hashes[2]])]))
        assert hit(cache) == expected
    cache.apply(parse([{"type": "AllBlocksCleared"}]))
    cache.apply(full)
    assert hit(cache) == 0
    cache.apply(sparse)
    assert hit(cache) == 16


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("native", [False], indirect=True, ids=["native-bytes"])
def test_native_coarse_aliases_resolve_sparse_events_and_parent_continuation(
    native_metadata, native, stored, parse, removed
):
    cache, index = direct_cache(native_metadata)
    chain = native.chain()
    aliases = chain.full_hashes
    cache.apply(parse([stored(0, "sliding_window", 4, hashes=chain.hashes[1:], kv_cache_spec_sliding_window=9)]))
    assert cache.pending and hit(cache) == 0
    for start, alias, parent in [(0, aliases[0], None), (8, aliases[1], aliases[0])]:
        cache.apply(
            parse([stored(1, "full_attention", 8, tokens=list(range(start, start + 8)), hashes=[alias], parent=parent)])
        )
    assert not cache.native_reconstruction_paused and not cache.pending and hit(cache) == 16
    for alias, endpoint in zip(aliases, chain.digests[1::2], strict=True):
        assert index.identities[str(alias)].digest == endpoint
    assert all(str(digest) in index.identities for digest in chain.digests)
    cache.apply(parse([removed(1, [aliases[1]])]))
    assert hit(cache) == 0


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("native", [False], indirect=True, ids=["native-bytes"])
def test_native_coarse_alias_rejects_corruption(native_metadata, native, stored, parse):
    _, index = direct_cache(native_metadata)
    alias = native.chain().full_hashes[0]
    # In the older concatenated encoding this preserves the final digest,
    # guarding against accepting an alias by blindly truncating to its tail.
    tampered = bytes([alias[0] ^ 1]) + alias[1:]
    event = parse([stored(1, "full_attention", 8, tokens=list(range(8)), hashes=[tampered])])[0]
    with pytest.raises(PrefixIdentityMismatch, match="native_hash_mismatch"):
        index.resolve_event(event)
    assert not index.identities


@pytest.mark.cpu
@pytest.mark.level0
def test_engine_root_profile_is_independent_of_router_global_seed(native_metadata, events, native, monkeypatch):
    local_root = bytes([native.root[0] ^ 1]) + native.root[1:]
    monkeypatch.setenv("PYTHONHASHSEED", "router-different-seed")
    monkeypatch.setattr(native.utils, "NONE_HASH", local_root)
    cache, index = direct_cache(native_metadata)
    full, sparse = events
    cache.apply(full + sparse)
    assert index.root == native.root and hit(cache) == 16
    assert native.utils.NONE_HASH == local_root


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("pickle_protocol", pickle.HIGHEST_PROTOCOL - 1, "pickle_protocol_mismatch"),
        ("int_hashes", False, "int_hashes_mismatch"),
        ("vllm_version", "0.99.0", "vllm_version_mismatch"),
        ("root_hash_hex", None, "invalid_root_hash"),
    ],
)
def test_incompatible_native_profile_keeps_conservative_estimates(
    field, value, reason, store, events, install_metadata
):
    cache_store, metadata = store
    local = metadata["native_hash_config"][field]
    if value is None:
        metadata["native_hash_config"].pop(field)
    else:
        metadata["native_hash_config"][field] = value
    assert VLLMHashCodec.from_metadata(metadata, 4).codec is None
    install_metadata(cache_store, "a", metadata)
    status = cache_store.metadata_status("a")
    assert status["native"] == "unavailable"
    assert status["reason"] == reason
    assert status["native_details"] == ({"remote": value, "local": local} if value is not None else {})
    full, sparse = events
    cache_store.apply_update(KVCacheUpdate(replica_id="a", events=full + sparse))
    cache = cache_store.group_caches["a"]
    assert hit(cache) == 0 and cache.pending


@pytest.mark.cpu
@pytest.mark.level0
def test_wrong_engine_root_fails_native_verification(native_metadata, events, native):
    native_metadata["native_hash_config"]["root_hash_hex"] = (bytes([native.root[0] ^ 1]) + native.root[1:]).hex()
    cache, index = direct_cache(native_metadata)
    full, sparse = events
    cache.apply(full + sparse)
    assert cache.native_reconstruction_paused and hit(cache) == 0 and cache.pending
    assert not index.identities


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("groups", [(("full_attention", 4), ("mamba", 8)), (("sliding_window", 4),)])
def test_layouts_without_missing_full_endpoints_skip_native_hashing(groups, layout, install_metadata):
    value = KVCacheStore()

    def unexpected_factory(metadata, unit):
        pytest.fail("This layout does not require native identity reconstruction")

    value.set_native_hash_codec_factory(unexpected_factory)
    metadata = layout(groups, unit=4)
    install_metadata(value, "a", metadata)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("unit", 8, "unit_mismatch"),
        ("vllm_version", "0.99.0", "vllm_version_mismatch"),
        ("int_hashes", False, "int_hashes_mismatch"),
    ],
)
def test_codec_rejection_exposes_remote_and_local_values(native_metadata, field, value, reason):
    native_metadata["native_hash_config"][field] = value
    result = VLLMHashCodec.from_metadata(native_metadata, 4)
    assert result is not None
    assert result.codec is None
    assert result.reason == reason
    assert result.details["remote"] == value
    assert "local" in result.details


@pytest.mark.cpu
@pytest.mark.level0
def test_bad_native_event_isolated_to_replica(store, events, native, stored, parse):
    value, _ = store
    full, sparse = events
    for node in ("a", "b"):
        value.apply_update(KVCacheUpdate(node, events=full + sparse))
    a, b = value.group_caches["a"], value.group_caches["b"]
    assert a.native_hashes is b.native_hashes
    before = value.per_replica_block_counts()
    bad = parse([stored(1, "full_attention", 8, hashes=["corrupt-native-hash"], tokens=list(range(8)))])
    value.apply_update(KVCacheUpdate("a", events=bad))
    assert a.native_reconstruction_paused and not b.native_reconstruction_paused
    status = value.metadata_status("a")
    assert status["native"] == "paused"
    assert status["reason"] == "native_hash_mismatch"
    assert value.metadata_status("b")["native"] == "ready"
    assert hit(a) == hit(b) == 16
    assert not a.pending and value.per_replica_block_counts() == before

    parent = native.root
    digests = []
    for start in range(100, 116, 4):
        parent = native.utils.hash_block_tokens(native.sha256, parent, list(range(start, start + 4)), None)
        digests.append(parent)
    full_hashes = [
        native.utils.maybe_convert_block_hash(h) for h in native.utils.BlockHashListWithBlockSize(digests, 4, 8)
    ]
    sparse_hashes = [native.utils.maybe_convert_block_hash(h) for h in digests[1:]]
    fresh = parse(
        [
            stored(1, "full_attention", 8, hashes=full_hashes, tokens=list(range(100, 116))),
            stored(
                0,
                "sliding_window",
                4,
                hashes=sparse_hashes,
                tokens=list(range(100, 116)),
                kv_cache_spec_sliding_window=9,
            ),
        ],
        "b",
    )
    value.apply_update(KVCacheUpdate("b", events=fresh))
    hashes, _ = get_prefix_hashes_incremental(list(range(100, 117)), 4, 0, 0)
    assert b.estimate(list(map(str, hashes)), 17).estimated_hit_tokens == 16


@pytest.mark.cpu
@pytest.mark.level0
def test_failed_codec_rebind_preserves_shared_identities_and_can_retry(store, events):
    value, _ = store
    full, sparse = events
    value.apply_update(KVCacheUpdate("a", events=full))
    attempts = 0

    def factory(metadata, unit):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("could not bind second replica")
        return VLLMHashCodec.from_metadata(metadata, unit)

    with pytest.raises(RuntimeError, match="second replica"):
        value.set_native_hash_codec_factory(factory)
    assert value.metadata_status("b")["native"] == "ready"
    value.apply_update(KVCacheUpdate("b", events=sparse))
    assert not value.group_caches["b"].pending
    assert len(value.group_caches["b"].by_group[0]) == 3

    value.set_native_hash_codec_factory(factory)
    assert attempts == 4
    assert not value.group_caches


@pytest.mark.cpu
@pytest.mark.level0
def test_failed_metadata_refresh_preserves_cache_until_successful_retry(store, events):
    value, metadata = store
    root = bytes.fromhex(metadata["native_hash_config"]["root_hash_hex"])
    new_root = bytes(b ^ 0xFF for b in root).hex()
    fail_next = True

    def factory(snapshot, unit):
        nonlocal fail_next
        if snapshot["native_hash_config"]["root_hash_hex"] == new_root and fail_next:
            fail_next = False
            raise RuntimeError("new codec temporarily unavailable")
        return VLLMHashCodec.from_metadata(snapshot, unit)

    value.set_native_hash_codec_factory(factory)
    full, sparse = events
    for replica in ("a", "b"):
        value.apply_update(KVCacheUpdate(replica, events=full + sparse))
    a, b = value.group_caches["a"], value.group_caches["b"]
    assert a.native_hashes is b.native_hashes and hit(a) == hit(b) == 16
    assert not value.refresh_replica_metadata("a", metadata)
    assert value.group_caches["a"] is a
    old_state = value.registry_state("a")

    # A caller may reuse and mutate the very object it supplied to both replicas.
    metadata["native_hash_config"]["root_hash_hex"] = new_root
    metadata["cache_reservation_profile"]["capacity_blocks"] = 7
    with pytest.raises(RuntimeError, match="new codec"):
        value.refresh_replica_metadata("a", metadata)
    assert value.registry_state("a") is old_state and value.capacity_blocks("a") is None
    assert value.group_caches["a"] is a and not value.metadata_status("a")["gated"]
    assert value.refresh_replica_metadata("a", metadata)
    assert value.capacity_blocks("a") == 7 and value.metadata_status("a")["gated"]
    assert "a" not in value.group_caches
    value.apply_update(KVCacheUpdate("a", events=[]))
    assert value.group_caches["a"].native_hashes.root.hex() == new_root
    assert value.group_caches["b"] is b and hit(b) == 16
