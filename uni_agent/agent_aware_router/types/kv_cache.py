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

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GroupReservationSpec:
    """One group's prompt-prefix budget plus request-private pool blocks.

    ``private_blocks=None`` means the engine has not supplied a usable budget.
    Prefix budgets retain #257's complete-prompt-block estimate; bounded
    attention groups conservatively retain that chain until request release.
    """

    group_idx: int
    block_size: int
    share_prefix: bool = True
    private_blocks: int | None = 0


@dataclass(frozen=True)
class GroupReservation:
    group_idx: int
    block_hashes: tuple[str, ...] = ()
    private_blocks: int | None = 0


@dataclass(frozen=True)
class DispatchSchema:
    """Acquire-time layout; matching and reservation hash units are independent."""

    generation: int
    hash_unit: int
    groups: tuple[GroupReservationSpec, ...]

    def request_blocks(self, prompt_tokens: int, saved_tokens: int = 0) -> float | None:
        """Prefill-work cost in shared-pool block units, including private budgets."""
        if any(group.private_blocks is None for group in self.groups):
            return None
        remaining = max(0, prompt_tokens - saved_tokens)
        return (
            sum(
                (remaining / group.block_size if group.share_prefix else 0) + (group.private_blocks or 0)
                for group in self.groups
            )
            if prompt_tokens
            else 0.0
        )


@dataclass(frozen=True)
class KVCacheEvent:
    """Normalized cache identity and residency event shared across layers.

    Attributes:
        event_type: ``"stored"`` / ``"removed"`` / ``"clear"``.
        node_id: Replica or event-source identifier.
        block_hashes: Normalized remote identities reported by the source.
        parent_block_hash: Parent identity for the reported token range.
        block_size: Block size (only present in BlockStored events).
        raw_token_ids: Complete token range, including partial/sparse-event tokens.
        group_idx: Cache group identifier when known.
        kind: Cache type used to select the matching rule.
        sliding_window: Window size reported by a sliding-attention event.
    """

    event_type: str
    node_id: str
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
    """An ordered cache-event batch with stream validity information.

    Attributes:
        node_id: Target endpoint identifier.
        events: Stored, Removed and Clear events in their original order.
        sequence: Source batch position used to exclude duplicate delivery.
        stream_error: Missing events invalidate this node until Clear or a new source.
        reset_stream: Clear the old index and sequence when replacing an event source.
    """

    node_id: str
    events: list[KVCacheEvent] = field(default_factory=list)
    sequence: int | None = None
    stream_error: str | None = None
    reset_stream: bool = False
