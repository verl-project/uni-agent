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

"""Decode native vLLM arrays and mappings into shared KV events."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from ....types.kv_cache import KVCacheEvent

logger = logging.getLogger(__name__)


def decode_kv_events(raw_data: Any, default_node_id: str | None = None) -> list[KVCacheEvent]:
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

        event_type = _resolve_event_type(tag)
        if event_type.startswith("unknown"):
            continue

        try:
            event = _build_event(event_type, fields, node_id)
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


def _build_event(
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
        return _build_block_stored(fields, node_id)
    elif event_type == "removed":
        return _build_block_removed(fields, node_id)
    elif event_type == "clear":
        return _build_all_blocks_cleared(node_id)
    return None


def _opt_str(fields: list | tuple, idx: int) -> str | None:
    """A field as str, or None when absent/None.

    Older vLLM omits trailing event fields, so absence is normal here.
    """
    if idx >= len(fields) or fields[idx] is None:
        return None
    if not isinstance(fields[idx], str):
        raise ValueError(f"Invalid string field at index {idx}")
    return fields[idx]


def _build_block_stored(fields: list | tuple, node_id: str) -> KVCacheEvent:
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
    medium = _opt_str(fields, 5)  # vLLM medium: "GPU" / "cpu"
    if medium is None:
        medium = "GPU"  # Native older events omit the GPU medium.
    extra_keys = fields[7] if len(fields) > 7 else None
    if extra_keys is not None and not isinstance(extra_keys, list | tuple):
        raise ValueError("Invalid extra_keys")
    group_idx = fields[8] if len(fields) > 8 else None
    if group_idx is not None and (type(group_idx) is not int or group_idx < 0):
        raise ValueError("Invalid group index")

    return KVCacheEvent(
        event_type="stored",
        node_id=node_id,
        block_hashes=block_hashes,
        parent_block_hash=parent_block_hash,
        block_size=block_size,
        medium=medium,
        raw_token_ids=tuple(raw_token_ids) if raw_token_ids is not None else None,
        group_idx=group_idx,
        kind=_opt_str(fields, 9),
        sliding_window=fields[10] if len(fields) > 10 else None,
        lora_id=fields[4] if len(fields) > 4 else None,
        lora_name=_opt_str(fields, 6),
        extra_keys=extra_keys,
        locality=_opt_str(fields, 11),
        ownership=_opt_str(fields, 12),
    )


def _build_block_removed(fields: list | tuple, node_id: str) -> KVCacheEvent:
    """Build a BlockRemoved event from its field list.

    Field order (after tag):
        0: block_hashes — list of block hash values
        1: medium       — str or None
        2: group_idx    — int or None
        3: locality     — str or None
        4: ownership    — str or None
    """
    block_hashes = [str(bh) for bh in fields[0]]
    medium = _opt_str(fields, 1)
    if medium is None:
        medium = "GPU"
    group_idx = fields[2] if len(fields) > 2 else None
    if group_idx is not None and (type(group_idx) is not int or group_idx < 0):
        raise ValueError("Invalid group index")

    return KVCacheEvent(
        event_type="removed",
        node_id=node_id,
        block_hashes=block_hashes,
        parent_block_hash=None,
        block_size=None,
        medium=medium,
        group_idx=group_idx,
        locality=_opt_str(fields, 3),
        ownership=_opt_str(fields, 4),
    )


def _build_all_blocks_cleared(node_id: str) -> KVCacheEvent:
    """Build an AllBlocksCleared event — no fields after tag."""
    return KVCacheEvent(
        event_type="clear",
        node_id=node_id,
        block_hashes=[],
        parent_block_hash=None,
        block_size=None,
    )


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
