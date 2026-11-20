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

from ..store.contracts import RoutingState
from .hash import get_prefix_hashes_incremental


def resolve_prefix_hashes(
    prompt_ids: list[int],
    request_id: str | None,
    store: RoutingState,
) -> list[str]:
    """Return ``str(h)`` for each full-block chained prefix hash of ``prompt_ids``.

    With a ``request_id``, the ``(parent_hash, len(hash_strs))`` checkpoint is
    memoized in the per-request store so only appended blocks are re-hashed on
    later turns. As in the original router, prompts are assumed append-only;
    fewer complete blocks or a changed hash schema trigger a full recompute.
    Returns ``[]`` when ``block_size`` is unknown.
    """
    schema = store.kv.hash_schema()
    _, block_size, lookahead = schema
    if not block_size or block_size <= 0:
        return []
    # A branched continuation may change a previously hashed block's suffix.
    # The request checkpoint does not record the suffix it consumed.
    if lookahead:
        hashes, _ = get_prefix_hashes_incremental(prompt_ids, block_size, 0, 0, lookahead=lookahead)
        return [str(h) for h in hashes]
    cached = store.requests.get_prefix_cache(request_id) if request_id is not None else None
    if cached and cached.get("schema") == schema and len(prompt_ids) // block_size >= len(cached["hash_strs"]):
        tail, new_parent = get_prefix_hashes_incremental(
            prompt_ids, block_size, cached["parent_hash"], len(cached["hash_strs"])
        )
        cached["hash_strs"].extend([str(h) for h in tail])
        cached["parent_hash"] = new_parent
        store.requests.set_prefix_cache(request_id, cached)
        return cached["hash_strs"]

    hashes, parent = get_prefix_hashes_incremental(prompt_ids, block_size, 0, 0)
    hash_strs = [str(h) for h in hashes]
    if request_id is not None:
        store.requests.set_prefix_cache(request_id, {"schema": schema, "hash_strs": hash_strs, "parent_hash": parent})
    return hash_strs
