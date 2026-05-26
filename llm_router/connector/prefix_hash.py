"""Versioned cache key for the Mooncake KV pool (RFC §5.2).

The key combines a hash of the prompt-token prefix with the actor weight
version that produced (or will consume) the KV. The version field provides
the cross-step semantic safety described in RFC §5.2: KV produced under
weight v_n is never reused under weight v_{n+1}, because the key fails to
match.

The hashing scheme intentionally mirrors verl's _build_prefix_signature in
verl/verl/experimental/agent_loop/agent_loop.py so that any prefix index
reported by a verl-managed replica produces the same key as the connector
side.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

PREFIX_HASH_DIGEST_BYTES = 16
DEFAULT_PREFIX_PROBE_STRIDE = 256


def coerce_weight_version(weight_version: Any) -> str:
    return "unknown" if weight_version is None else str(weight_version)


def hash_token_prefix(prompt_ids: list[int], prefix_len: int) -> str:
    """Hash the first `prefix_len` tokens of `prompt_ids` to a hex string.

    The hash is salted with the prefix length so that hash([1,2,3], 2) is
    distinct from hash([1,2,3], 3) even when tokens beyond `prefix_len` are
    present in the list.
    """
    hasher = hashlib.blake2b(digest_size=PREFIX_HASH_DIGEST_BYTES)
    for token_id in prompt_ids[:prefix_len]:
        hasher.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    hasher.update(b":")
    hasher.update(str(prefix_len).encode("ascii"))
    return hasher.hexdigest()


@dataclass(frozen=True)
class VersionedKey:
    """`(weight_version, prefix_hash, prefix_len)` triple identifying KV in the pool."""

    weight_version: str
    prefix_hash: str
    prefix_len: int

    def to_string(self) -> str:
        return f"{self.weight_version}:{self.prefix_hash}:{self.prefix_len}"

    @classmethod
    def from_string(cls, s: str) -> VersionedKey:
        version, prefix_hash, prefix_len_str = s.rsplit(":", 2)
        return cls(
            weight_version=version,
            prefix_hash=prefix_hash,
            prefix_len=int(prefix_len_str),
        )


def make_versioned_key(
    prompt_ids: list[int],
    prefix_len: int,
    weight_version: Any,
) -> VersionedKey:
    if prefix_len <= 0 or prefix_len > len(prompt_ids):
        raise ValueError(
            f"prefix_len must be in (0, {len(prompt_ids)}], got {prefix_len}"
        )
    return VersionedKey(
        weight_version=coerce_weight_version(weight_version),
        prefix_hash=hash_token_prefix(prompt_ids, prefix_len),
        prefix_len=prefix_len,
    )


def build_prefix_signature(
    prompt_ids: list[int],
    weight_version: Any,
) -> tuple[str, str, int]:
    """Return verl-compatible `(version, hash, prefix_len)` for full prompt."""
    prefix_len = len(prompt_ids)
    if prefix_len <= 0:
        raise ValueError("prompt_ids must be non-empty")
    return (
        coerce_weight_version(weight_version),
        hash_token_prefix(prompt_ids, prefix_len),
        prefix_len,
    )


def build_prefix_signature_at_len(
    prompt_ids: list[int],
    weight_version: Any,
    prefix_len: int,
) -> tuple[str, str, int]:
    """Return verl-compatible `(version, hash, prefix_len)` at a given length."""
    key = make_versioned_key(prompt_ids, prefix_len, weight_version)
    return key.weight_version, key.prefix_hash, key.prefix_len


def iter_prefix_signatures(
    prompt_ids: list[int],
    weight_version: Any,
    *,
    stride: int = DEFAULT_PREFIX_PROBE_STRIDE,
) -> list[tuple[str, str, int]]:
    """Mirror verl `_iter_prefix_signatures`.

    The worker reports sampled prefixes at every stride boundary plus the
    full prompt length. The connector uses the same helper so a router hit
    implies the KV store can use the same key granularity.
    """
    if not prompt_ids:
        return []
    stride = max(1, int(stride))
    lengths = list(range(stride, len(prompt_ids), stride))
    if not lengths or lengths[-1] != len(prompt_ids):
        lengths.append(len(prompt_ids))

    version = coerce_weight_version(weight_version)
    sampled_lengths = set(lengths)
    hasher = hashlib.blake2b(digest_size=PREFIX_HASH_DIGEST_BYTES)
    signatures: list[tuple[str, str, int]] = []
    for index, token_id in enumerate(prompt_ids, start=1):
        hasher.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
        if index in sampled_lengths:
            prefix_hasher = hasher.copy()
            prefix_hasher.update(b":")
            prefix_hasher.update(str(index).encode("ascii"))
            signatures.append((version, prefix_hasher.hexdigest(), index))
    return signatures
