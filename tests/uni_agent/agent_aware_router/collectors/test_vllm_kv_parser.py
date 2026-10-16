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

"""Unit tests for VLLMKVParser layer bucketing (mixed-medium frames)."""

from __future__ import annotations

import logging

import msgpack
import pytest

from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.types import Layer

pytestmark = [pytest.mark.level0, pytest.mark.cpu]


def _stored_event(block_hash, parent, token_ids, block_size, medium):
    """A stored event entry: [tag, block_hashes, parent, token_ids, block_size, <unused>, medium]."""
    return ["stored", [block_hash], parent, token_ids, block_size, None, medium]


def test_mixed_medium_frame_buckets_per_layer():
    """A single frame with a GPU and a cpu BlockStored keeps layers distinct.

    Regression: the old scalar medium_add aggregation let the later event's
    medium overwrite the earlier one, so the whole batch was written under one
    layer. Per-layer dict bucketing must keep each event's blocks in its layer.
    """
    parser = VLLMKVParser()
    payload = [
        1234567890,  # timestamp
        [
            _stored_event("rh_gpu", None, [1, 2], 2, "GPU"),
            _stored_event("rh_cpu", None, [3, 4], 2, "cpu"),
        ],
    ]

    update = parser.parse(msgpack.packb(payload), "node1")

    assert update is not None
    # Both layers present — no cross-layer overwrite.
    assert Layer.GPU in update.add_blocks
    assert Layer.CPU in update.add_blocks
    assert len(update.add_blocks[Layer.GPU]) == 1
    assert len(update.add_blocks[Layer.CPU]) == 1
    # Different token ids → different local hashes per layer.
    assert update.add_blocks[Layer.GPU] != update.add_blocks[Layer.CPU]


def test_none_medium_defaults_to_gpu():
    """Older vLLM events without medium default to the GPU layer."""
    parser = VLLMKVParser()
    payload = [0, [_stored_event("rh", None, [1, 2], 2, None)]]

    update = parser.parse(msgpack.packb(payload), "node1")

    assert update is not None
    assert Layer.GPU in update.add_blocks
    assert Layer.CPU not in update.add_blocks


def test_clear_event_sets_clear_all():
    """An AllBlocksCleared event marks the update for a full replica clear."""
    parser = VLLMKVParser()
    payload = [0, [["clear"]]]

    update = parser.parse(msgpack.packb(payload), "node1")

    assert update is not None
    assert update.clear_all is True


def test_parse_failure_surfaces_exception_not_swallowed():
    """A malformed payload returns None and logs the real error.

    Regression: the ``except`` used a non-f-string with an unbound ``{exc}``,
    so the warning rendered the literal text ``"{exc}"`` and the actual parse
    error was silently lost.
    """
    parser = VLLMKVParser()
    # 0xc1 is a reserved/invalid msgpack byte → unpackb raises UnpackException,
    # exercising the failed-to-parse branch (not the unexpected-format branch).
    garbage = b"\xc1\xc1\xc1"

    records: list[logging.LogRecord] = []
    capture = logging.Handler()
    capture.emit = records.append  # type: ignore[method-assign]
    capture.setLevel(logging.WARNING)
    module_logger = logging.getLogger(VLLMKVParser.__module__)
    module_logger.addHandler(capture)
    try:
        update = parser.parse(garbage, "node1")
    finally:
        module_logger.removeHandler(capture)

    assert update is None
    text = "\n".join(record.getMessage() for record in records)
    assert "{exc}" not in text  # placeholder must be gone
    assert "node1" in text  # node_id must interpolate
    assert "len=" in text and "head=c1c1c1" in text  # diagnostic preview present


# ── Per-replica remote→local hash map (shared-map regression) ─────────────


def _chained_stored(node_hashes, tokens_per_block=2, parent=None, medium="GPU"):
    """One BlockStored frame: [timestamp, [[tag, hashes, parent, tokens, block_size, None, medium]]]."""
    ids = list(range(len(node_hashes) * tokens_per_block))
    return [0, [["stored", list(node_hashes), parent, ids, tokens_per_block, None, medium]]]


def _removed(hashes, medium="GPU"):
    """One BlockRemoved frame."""
    return [0, [["removed", list(hashes), medium, None]]]


def _apply(parser, payload, node_id, store):
    update = parser.parse(msgpack.packb(payload), node_id)
    assert update is not None
    for layer, hashes in update.remove_blocks.items():
        if hashes:
            store.remove_kv_blocks(node_id, hashes, layer=layer)
    for layer, hashes in update.add_blocks.items():
        if hashes:
            store.add_kv_blocks(node_id, hashes, layer=layer)
    return update


def test_hash_map_is_per_replica_and_survives_sibling_eviction():
    """A sibling replica's eviction must not break this replica's translation.

    Regression: a flat shared map let replica A's eviction pop the entry
    replica B still needed, so B's retained count kept drifting high
    (design doc §5.3.5).
    """
    parser = VLLMKVParser()
    store = DataStore()
    remote_hashes = ["rh0", "rh1", "rh2"]

    # Same content-chained prefix stored on both replicas → identical local chain.
    _apply(parser, _chained_stored(remote_hashes), "nodeA", store)
    _apply(parser, _chained_stored(remote_hashes), "nodeB", store)
    assert store.per_replica_block_counts() == {"nodeA": 3, "nodeB": 3}

    # Per-replica maps: one entry per node, and local hashes agree across nodes
    # (the prompt-side chain is replica-independent, so they must).
    node_maps = parser.remote_to_local_block_hash
    assert set(node_maps) == {"nodeA", "nodeB"}
    assert node_maps["nodeA"] == node_maps["nodeB"], "local chain must not depend on the replica"

    # nodeA drops the whole prefix first …
    _apply(parser, _removed(remote_hashes), "nodeA", store)
    assert store.per_replica_block_counts() == {"nodeA": 0, "nodeB": 3}

    # … nodeB's eviction of the same remote hashes must still translate.
    update = _apply(parser, _removed(remote_hashes), "nodeB", store)
    assert len(update.remove_blocks[Layer.GPU]) == 3, "sibling eviction must not eat this replica's mapping"
    assert store.per_replica_block_counts()["nodeB"] == 0, "nodeB must not retain blocks its engine freed"
    assert parser.stats["removed_unmapped"] == 0
