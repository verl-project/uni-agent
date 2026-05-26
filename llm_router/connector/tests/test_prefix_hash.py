"""Versioned cache key derived from token prefix + weight_version."""
import pytest

from llm_router.connector.prefix_hash import (
    PREFIX_HASH_DIGEST_BYTES,
    VersionedKey,
    build_prefix_signature,
    build_prefix_signature_at_len,
    coerce_weight_version,
    hash_token_prefix,
    iter_prefix_signatures,
    make_versioned_key,
)


def test_coerce_weight_version_handles_none():
    assert coerce_weight_version(None) == "unknown"


def test_coerce_weight_version_stringifies_int():
    assert coerce_weight_version(42) == "42"


def test_hash_token_prefix_is_deterministic():
    a = hash_token_prefix([1, 2, 3], 3)
    b = hash_token_prefix([1, 2, 3], 3)
    assert a == b
    assert len(a) == PREFIX_HASH_DIGEST_BYTES * 2  # hex


def test_hash_token_prefix_changes_with_length():
    a = hash_token_prefix([1, 2, 3], 3)
    b = hash_token_prefix([1, 2, 3, 4], 4)
    assert a != b


def test_hash_token_prefix_changes_with_tokens():
    a = hash_token_prefix([1, 2, 3], 3)
    b = hash_token_prefix([1, 2, 9], 3)
    assert a != b


def test_hash_truncates_to_prefix_len():
    # Hashing only the first 2 tokens of a 3-token list must equal hashing
    # the same 2-token list outright.
    a = hash_token_prefix([1, 2, 999], 2)
    b = hash_token_prefix([1, 2], 2)
    assert a == b


def test_make_versioned_key_combines_hash_and_version():
    k = make_versioned_key([1, 2, 3], prefix_len=3, weight_version=7)
    assert isinstance(k, VersionedKey)
    assert k.weight_version == "7"
    assert k.prefix_hash == hash_token_prefix([1, 2, 3], 3)
    assert k.prefix_len == 3


def test_versioned_key_is_hashable_and_value_equal():
    a = make_versioned_key([1, 2, 3], 3, "v0")
    b = make_versioned_key([1, 2, 3], 3, "v0")
    assert a == b
    assert hash(a) == hash(b)
    assert a in {b}


def test_versioned_key_serializable_to_string():
    k = make_versioned_key([1, 2, 3], 3, "v0")
    assert k.to_string().startswith("v0:")
    assert k.to_string().endswith(":3")


def test_versioned_key_roundtrip_string():
    k = make_versioned_key([1, 2, 3], 3, "v0")
    parsed = VersionedKey.from_string(k.to_string())
    assert parsed == k


def test_make_versioned_key_rejects_bad_prefix_len():
    with pytest.raises(ValueError):
        make_versioned_key([1, 2, 3], prefix_len=0, weight_version="v0")
    with pytest.raises(ValueError):
        make_versioned_key([1, 2, 3], prefix_len=4, weight_version="v0")


def test_build_prefix_signature_matches_versioned_key():
    sig = build_prefix_signature([1, 2, 3], "v0")
    key = make_versioned_key([1, 2, 3], 3, "v0")
    assert sig == (key.weight_version, key.prefix_hash, key.prefix_len)


def test_build_prefix_signature_at_len_matches_versioned_key():
    sig = build_prefix_signature_at_len([1, 2, 3, 4], "v0", 2)
    key = make_versioned_key([1, 2, 3, 4], 2, "v0")
    assert sig == (key.weight_version, key.prefix_hash, key.prefix_len)


def test_iter_prefix_signatures_samples_stride_plus_full_prompt():
    sigs = iter_prefix_signatures(list(range(1, 10)), "v0", stride=4)
    lengths = [sig[2] for sig in sigs]
    assert lengths == [4, 8, 9]
    assert sigs[0] == build_prefix_signature_at_len(list(range(1, 10)), "v0", 4)
    assert sigs[-1] == build_prefix_signature(list(range(1, 10)), "v0")


def test_iter_prefix_signatures_handles_short_prompt_and_empty_prompt():
    assert iter_prefix_signatures([], "v0", stride=4) == []
    sigs = iter_prefix_signatures([1, 2], "v0", stride=4)
    assert sigs == [build_prefix_signature([1, 2], "v0")]


def test_iter_prefix_signatures_clamps_bad_stride():
    sigs = iter_prefix_signatures([1, 2, 3], "v0", stride=0)
    assert [sig[2] for sig in sigs] == [1, 2, 3]
