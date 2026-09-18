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
