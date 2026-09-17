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
Chained xxhash prefix hash computation.
"""

from __future__ import annotations

import struct

import xxhash


def compute_hash(parent_hash: int, block_bytes: bytes, seed: int = 0) -> int:
    """Compute xxhash for a single block given parent hash and token bytes.

    Algorithm (matching aibrix ``SyncPrefixHashTable.computeHash``):
        h = xxhash.xxh64(seed=seed)
        h.update(parent_hash as 8-byte little-endian)
        h.update(block_bytes)

    The parent hash is **always** written — for the first block in a
    chain, ``parent_hash`` should be ``seed`` (typically ``0``).

    Args:
        parent_hash: Parent (predecessor) prefix hash as ``int``.
            Use ``seed`` for the first block in a chain.
        block_bytes: Token bytes for this block, already encoded as
            uint32 big-endian (4 bytes per token).
        seed: xxhash seed value. Defaults to ``0``.

    Returns:
        Prefix hash as ``int`` (equivalent to Go ``uint64``).
    """
    h = xxhash.xxh64(seed=seed)
    h.update(parent_hash.to_bytes(8, "little"))
    h.update(block_bytes)
    return h.intdigest()


def get_prefix_hashes_incremental(
    prompt_ids: list[int],
    block_size: int,
    parent_hash: int,
    n_done: int,
    seed: int = 0,
) -> tuple[list[int], int]:
    """Continue a chained-prefix-hash computation from a checkpoint.

    Given the chain head ``parent_hash`` (= the hash of block ``n_done - 1``,
    or ``seed`` when ``n_done == 0``) and the number of blocks ``n_done``
    already hashed, hash only blocks ``[n_done, n_full_blocks)`` and append.
    The chain is deterministic, so the appended hashes are byte-identical to
    what a full computation from scratch would have produced.

    Args:
        prompt_ids: Prompt token IDs (the full, current-turn prompt).
        block_size: Tokens per block (must match the checkpoint's block_size).
        parent_hash: Chain head at block ``n_done`` (hash of block n_done-1,
            or ``seed`` when ``n_done == 0``).
        n_done: Number of leading blocks already hashed (the checkpoint).
        seed: xxhash seed (must match the original full computation).

    Returns:
        ``(appended_hashes, final_parent_hash)`` where ``appended_hashes``
        covers only blocks ``[n_done, n_full_blocks)`` (empty when the prompt
        did not grow past the checkpoint) and ``final_parent_hash`` is the new
        chain head to cache for the next turn. Concatenate
        ``existing_hashes + appended_hashes`` for the full hash sequence.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    n_full_blocks = len(prompt_ids) // block_size
    appended: list[int] = []
    cur = parent_hash
    for i in range(n_done, n_full_blocks):
        start = i * block_size
        end = start + block_size
        block_bytes = struct.pack(f">{block_size}I", *prompt_ids[start:end])
        cur = compute_hash(cur, block_bytes, seed)
        appended.append(cur)
    return appended, cur
