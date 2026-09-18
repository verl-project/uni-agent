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

"""Per-request incremental cache: skip re-hashing the unchanged prefix across turns."""

from __future__ import annotations

from typing import Any

from .hash import get_prefix_hashes_incremental

_PREFIX_HASH_KEY = "prefix_hashes"


def resolve_prefix_hashes(
    prompt_ids: list[int],
    request_id: str | None,
    store: Any,
) -> list[str]:
    """Return ``str(h)`` for each full-block chained prefix hash of ``prompt_ids``.

    With a ``request_id``, the ``(parent_hash, len(hash_strs))`` checkpoint is
    memoized in the per-request store so only appended blocks are re-hashed on
    later turns. Assumes multi-turn prompts are append-only; a shrunk prompt
    triggers a full recompute. Returns ``[]`` when ``block_size`` is unknown.
    """
    block_size = store.get_block_size()
    if not block_size or block_size <= 0:
        return []

    cached = store.get_per_request(request_id, _PREFIX_HASH_KEY) if request_id else None
    if cached and len(prompt_ids) // block_size >= len(cached["hash_strs"]):
        # cached is the live store row (get returns a reference, not a copy):
        # the in-place extend + parent update persist, and get() already touched
        # LRU recency — no write-back needed.
        tail, new_parent = get_prefix_hashes_incremental(
            prompt_ids, block_size, cached["parent_hash"], len(cached["hash_strs"])
        )
        cached["hash_strs"].extend([str(h) for h in tail])
        cached["parent_hash"] = new_parent
        return cached["hash_strs"]

    # full recompute (miss / shrink)
    hashes, parent = get_prefix_hashes_incremental(prompt_ids, block_size, 0, 0)
    hash_strs = [str(h) for h in hashes]
    if request_id:
        store.set_per_request(request_id, _PREFIX_HASH_KEY, {"hash_strs": hash_strs, "parent_hash": parent})
    return hash_strs
