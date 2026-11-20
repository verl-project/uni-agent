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

"""Shared KV event and ordered update contracts for producers and consumers."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

LOCAL_GPU_DOMAIN = ("GPU", "LOCAL", None)


class NativeHashCodec(Protocol):
    """A stateless, validated source hash profile and its encoding operations."""

    unit: int
    root: bytes
    profile_key: Hashable

    def hash_block(self, parent: bytes, tokens: Sequence[int], lookahead: Sequence[int] = ()) -> bytes:
        """Hash one unit with an explicit initialized parent digest."""
        ...

    def external_key(self, digest: bytes) -> str:
        """Encode a digest as the normalized identity published by this source."""
        ...

    def coarse_keys(self, digests: Sequence[bytes], block_size: int) -> Mapping[str, int]:
        """Map published coarse identities to their final zero-based unit index."""
        ...


@dataclass(frozen=True)
class NativeHashCodecResult:
    codec: NativeHashCodec | None = None
    reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


NativeHashCodecFactory = Callable[[Mapping[str, Any], int], NativeHashCodecResult]


@dataclass(frozen=True)
class KVCacheEvent:
    """Normalized cache identity and residency event shared across layers.

    Attributes:
        event_type: ``"stored"`` / ``"removed"`` / ``"clear"``.
        replica_id: Replica or event-source identifier.
        block_hashes: Normalized remote identities reported by the source.
        parent_block_hash: Parent identity for the reported token range.
        block_size: Block size (only present in BlockStored events).
        raw_token_ids: Complete token range, including partial/sparse-event tokens.
        group_idx: Cache group identifier when known.
        kind: Cache type used to select the matching rule.
        sliding_window: Window size reported by a sliding-attention event.
    """

    event_type: str
    replica_id: str
    block_hashes: list[str]
    parent_block_hash: str | None
    block_size: int | None
    medium: str | None = None
    raw_token_ids: tuple[int, ...] | None = None
    group_idx: int | None = None
    kind: str | None = None
    sliding_window: int | None = None
    lora_id: Any = None
    lora_name: str | None = None
    extra_keys: Any = None
    locality: str | None = None
    ownership: str | None = None
    lookahead_token_ids: tuple[tuple[int, ...], ...] | None = None

    def hash_input_tokens(self, ends: Sequence[int], lookahead: int) -> tuple[int, ...] | None:
        """Extend the event range to include the suffix of its last reported block.

        Each suffix is relative to its own reported endpoint. Intersecting
        suffixes must agree with the tokens inside the event and with each other.
        """
        tokens = self.raw_token_ids
        if tokens is None or not ends:
            return None
        if lookahead == 0:
            return tokens
        rows = self.lookahead_token_ids
        if rows is None or len(rows) != len(ends) or ends[-1] > len(tokens):
            return None
        last = ends[-1]
        overlap = min(lookahead, len(tokens) - last)
        if rows[-1][:overlap] != tokens[last : last + overlap]:
            return None
        extended = tokens[:last] + rows[-1]
        for end, row in zip(ends, rows, strict=True):
            if end <= 0 or len(row) != lookahead or extended[end : end + lookahead] != row:
                return None
        return extended

    @property
    def storage_domain(self) -> tuple[str, str, str | None] | None:
        """Classify residency and evictions consistently, including older events without locality."""
        medium = self.medium.upper() if isinstance(self.medium, str) else None
        if medium not in {"GPU", "CPU", "STORAGE", "SSD"}:
            return None
        locality = self.locality.upper() if isinstance(self.locality, str) else "LOCAL"
        return medium, locality, self.ownership

    @property
    def is_store(self) -> bool:
        """True if this is a block-stored event."""
        return "stored" in self.event_type.lower()

    @property
    def is_remove(self) -> bool:
        """True if this is a block-removed event."""
        return any(k in self.event_type.lower() for k in ("removed", "evicted"))

    @property
    def is_clear(self) -> bool:
        """True if this is an all-blocks-cleared event."""
        return "clear" in self.event_type.lower()


@dataclass
class KVCacheUpdate:
    """An ordered cache-event batch for one replica.

    Transport reset/readiness callbacks independently control index validity.
    """

    replica_id: str
    events: list[KVCacheEvent] = field(default_factory=list)
