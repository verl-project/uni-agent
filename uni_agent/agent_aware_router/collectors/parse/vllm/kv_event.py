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

"""
KVCacheEvent — standardized KV cache event data structure.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KVCacheEvent:
    """Standardized KV cache event — normalized from backend-specific ZMQ payloads.

    Attributes:
        event_type: ``"stored"`` / ``"removed"`` / ``"clear"``.
        node_id: Source endpoint (from ZMQ connection or group_idx).
        block_hashes: Block hashes involved in the event (list from msgpack).
        parent_block_hash: Parent block hash — single value shared by all
                           block_hashes in a BlockStored event.
        token_ids: Legacy view of full blocks encoded as uint32 big-endian bytes.
        block_size: Block size (only present in BlockStored events).
        raw_token_ids: Complete token range, including partial/sparse-event tokens.
        group_idx: Native cache group identifier.
        kind: Native cache type used to select the matching rule.
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
    def token_ids(self) -> list[bytes] | None:
        if self.raw_token_ids is None or self.block_size is None:
            return None
        end = len(self.raw_token_ids) // self.block_size * self.block_size
        return _convert_token_ids(self.raw_token_ids[:end], self.block_size)

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_raw(cls, raw_data: Any, default_node_id: str | None = None) -> list[KVCacheEvent]:
        """Parse msgpack-decoded raw data into a list of KVCacheEvent instances.

        Parsing pattern::

            timestamp  = raw_data[0]  # ignored for routing
            event_list = raw_data[1]

            # Array event:
            event_tag    = event_entry[0]
            event_fields = event_entry[1:]

            # Mapping event:
            event_tag    = event_entry["type"]
            event_fields = event_entry

        Args:
            raw_data: Msgpack-decoded batch ``[timestamp, [event, ...], ...]``.
            default_node_id: Fallback node_id when raw data lacks it.

        Returns:
            A list of KVCacheEvent instances.  Malformed events are skipped.

        Raises:
            ValueError: If the top-level format is invalid.
        """
        if isinstance(raw_data, Mapping):
            event_list = raw_data.get("events")
        elif isinstance(raw_data, list | tuple) and len(raw_data) >= 2:
            event_list = raw_data[1]
        else:
            raise ValueError(f"Expected a KV event batch, got {type(raw_data)}")
        if not isinstance(event_list, list | tuple):
            raise ValueError(f"Expected event list, got {type(event_list)}")

        results: list[KVCacheEvent] = []
        node_id = default_node_id or ""
        malformed_count = 0
        first_error: tuple[str, str] | None = None
        for event_entry in event_list:
            if isinstance(event_entry, Mapping):
                tag = event_entry.get("type")
                fields = event_entry
            elif isinstance(event_entry, list | tuple) and event_entry:
                tag = event_entry[0]
                fields = event_entry[1:]
            else:
                continue

            event_type = cls._resolve_event_type(tag)
            if event_type.startswith("unknown"):
                continue

            try:
                event = cls._build_event(event_type, fields, node_id)
                if event is not None:
                    results.append(event)
            except (ValueError, TypeError, IndexError, KeyError) as exc:
                malformed_count += 1
                if first_error is None:
                    first_error = (event_type, str(exc))
                continue

        if first_error is not None:
            logger.warning(
                "Skipping malformed KV events (node=%s, count=%d, first_type=%s): %s",
                node_id,
                malformed_count,
                *first_error,
            )
        return results

    # ── Build helpers ────────────────────────────────────────────────────

    @classmethod
    def _build_event(
        cls,
        event_type: str,
        fields: list | tuple | Mapping,
        node_id: str,
    ) -> KVCacheEvent | None:
        """Normalize named fields to the legacy layout and share the builders."""
        if isinstance(fields, Mapping):
            if event_type == "stored":
                fields = [
                    fields["block_hashes"],
                    fields.get("parent_block_hash"),
                    fields.get("token_ids"),
                    fields["block_size"],
                    fields.get("lora_id"),
                    fields.get("medium"),
                    fields.get("lora_name"),
                    fields.get("extra_keys"),
                    fields.get("group_idx"),
                    fields.get("kv_cache_spec_kind"),
                    fields.get("kv_cache_spec_sliding_window"),
                    fields.get("locality"),
                    fields.get("ownership"),
                ]
            elif event_type == "removed":
                fields = [
                    fields["block_hashes"],
                    fields.get("medium"),
                    fields.get("group_idx"),
                    fields.get("locality"),
                    fields.get("ownership"),
                ]

        if event_type in {"stored", "removed"}:
            hashes = fields[0]
            if not isinstance(hashes, list | tuple):
                raise TypeError("block_hashes must be a list or tuple")
            if any(not isinstance(h, bytes | int | str) or isinstance(h, bool) for h in hashes):
                raise ValueError("Invalid block hash identity")

        if event_type == "stored":
            return cls._build_block_stored(fields, node_id)
        elif event_type == "removed":
            return cls._build_block_removed(fields, node_id)
        elif event_type == "clear":
            return cls._build_all_blocks_cleared(node_id)
        return None

    @staticmethod
    def _opt_str(fields: list | tuple, idx: int) -> str | None:
        """A field as str, or None when absent/None.

        Older vLLM omits trailing event fields, so absence is normal here.
        """
        if idx >= len(fields) or fields[idx] is None:
            return None
        if not isinstance(fields[idx], str):
            raise ValueError(f"Invalid string field at index {idx}")
        return fields[idx]

    @classmethod
    def _build_block_stored(cls, fields: list | tuple, node_id: str) -> KVCacheEvent:
        """Build a BlockStored event from its field list.

        Field order (after tag):
            0: block_hashes         — list of block hash values
            1: parent_block_hash    — single value or None
            2: token_ids            — list of int or None
            3: block_size           — int
            4: lora_id              — optional adapter identity
            5: medium               — optional storage layer
            6: lora_name            — optional adapter name
            7: extra_keys           — optional additional cache identity
            8: group_idx            — optional cache group
            9: kv_cache_spec_kind   — optional cache type
           10: kv_cache_spec_sliding_window — optional window size
           11: locality             — optional LOCAL/REMOTE domain
           12: ownership            — optional storage owner

        ``raw_token_ids`` retains the complete token range.
        """
        if len(fields) < 4:
            raise ValueError(f"BlockStored needs >= 4 fields, got {len(fields)}")

        block_hashes = [str(bh) for bh in fields[0]]
        parent = fields[1]
        if parent is not None and (not isinstance(parent, bytes | int | str) or isinstance(parent, bool)):
            raise ValueError("Invalid parent hash identity")
        parent_block_hash = str(parent) if parent is not None else None
        raw_token_ids = fields[2]
        block_size = fields[3]
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("Invalid event block size")
        if raw_token_ids is not None and (
            not isinstance(raw_token_ids, list | tuple)
            or any(type(t) is not int or not 0 <= t < 2**32 for t in raw_token_ids)
        ):
            raise ValueError("Invalid token ids")
        medium = cls._opt_str(fields, 5)  # vLLM medium: "GPU" / "cpu"
        extra_keys = fields[7] if len(fields) > 7 else None
        if extra_keys is not None and not isinstance(extra_keys, list | tuple):
            raise ValueError("Invalid extra_keys")
        group_idx = fields[8] if len(fields) > 8 else None
        if group_idx is not None and (type(group_idx) is not int or group_idx < 0):
            raise ValueError("Invalid group index")

        return cls(
            event_type="stored",
            node_id=node_id,
            block_hashes=block_hashes,
            parent_block_hash=parent_block_hash,
            block_size=block_size,
            medium=medium,
            raw_token_ids=tuple(raw_token_ids) if raw_token_ids is not None else None,
            group_idx=group_idx,
            kind=cls._opt_str(fields, 9),
            sliding_window=fields[10] if len(fields) > 10 else None,
            lora_id=fields[4] if len(fields) > 4 else None,
            lora_name=cls._opt_str(fields, 6),
            extra_keys=extra_keys,
            locality=cls._opt_str(fields, 11),
            ownership=cls._opt_str(fields, 12),
        )

    @classmethod
    def _build_block_removed(cls, fields: list | tuple, node_id: str) -> KVCacheEvent:
        """Build a BlockRemoved event from its field list.

        Field order (after tag):
            0: block_hashes — list of block hash values
            1: medium       — str or None
            2: group_idx    — int or None
            3: locality     — str or None
            4: ownership    — str or None
        """
        block_hashes = [str(bh) for bh in fields[0]]
        medium = cls._opt_str(fields, 1)
        group_idx = fields[2] if len(fields) > 2 else None
        if group_idx is not None and (type(group_idx) is not int or group_idx < 0):
            raise ValueError("Invalid group index")

        return cls(
            event_type="removed",
            node_id=node_id,
            block_hashes=block_hashes,
            parent_block_hash=None,
            block_size=None,
            medium=medium,
            group_idx=group_idx,
            locality=cls._opt_str(fields, 3),
            ownership=cls._opt_str(fields, 4),
        )

    @classmethod
    def _build_all_blocks_cleared(cls, node_id: str) -> KVCacheEvent:
        """Build an AllBlocksCleared event — no fields after tag."""
        return cls(
            event_type="clear",
            node_id=node_id,
            block_hashes=[],
            parent_block_hash=None,
            block_size=None,
        )

    # ── Tag resolution ───────────────────────────────────────────────────

    @staticmethod
    def _resolve_event_type(tag: Any) -> str:
        """Map msgspec struct tag to canonical event type string.

        vLLM msgspec uses numeric tags:
          - 0 → BlockStored
          - 1 → BlockRemoved
          - 2 → AllBlocksCleared

        String tags are also supported.
        """
        if isinstance(tag, int):
            return {0: "stored", 1: "removed", 2: "clear"}.get(tag, f"unknown_{tag}")

        tag_str = str(tag).lower()
        if "stored" in tag_str:
            return "stored"
        elif "removed" in tag_str or "evicted" in tag_str:
            return "removed"
        elif "clear" in tag_str:
            return "clear"
        return f"unknown_{tag}"

    # ── Convenience properties ──────────────────────────────────────────

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


# ── Token ID conversion ────────────────────────────────────────────────────


def _convert_token_ids(raw_ids: Sequence[int], block_size: int) -> list[bytes]:
    """Convert raw token IDs to list of block-sized uint32 big-endian byte chunks.

    Each block of token IDs is encoded as uint32 big-endian (4 bytes per token),
    matching aibrix's ``convertTokenIDs`` / ``tokenIDsToBytes``.

    Args:
        raw_ids: Raw token IDs as ``list[int]``.
        block_size: Number of tokens per block (must be > 0).

    Returns:
        List of bytes objects, one per full block.

    Raises:
        ValueError: If ``block_size <= 0``, ``len(raw_ids)`` is not
            divisible by ``block_size``, or any token ID is out of the
            uint32 range (``[0, 2**32)``).
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if len(raw_ids) % block_size != 0:
        raise ValueError(f"token_ids len={len(raw_ids)} not divisible by block_size={block_size}")
    num_blocks = len(raw_ids) // block_size
    result: list[bytes] = []
    try:
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size
            result.append(struct.pack(f">{block_size}I", *raw_ids[start:end]))
    except struct.error as e:
        raise ValueError(f"Invalid token ID value for struct packing: {e}") from e
    return result
