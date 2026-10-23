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

"""Unit tests for GPU-only VLLMKVParser updates and mixed-medium isolation."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import msgpack
import pytest
from conftest import mapping_event

from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.collectors.parse.vllm.kv_event import KVCacheEvent
from uni_agent.agent_aware_router.types import Layer


def _stored_event(block_hash, parent, token_ids, block_size, medium):
    """A stored event entry: [tag, block_hashes, parent, token_ids, block_size, <unused>, medium]."""
    return ["BlockStored", [block_hash], parent, token_ids, block_size, None, medium]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("non_gpu_medium", ["cpu", "CPU", "ssd", "SSD", "nvme", ""])
@pytest.mark.parametrize("wire_format", ["array", "mapping"])
@pytest.mark.parametrize("replay", [False, True])
def test_non_gpu_events_do_not_change_gpu_state(non_gpu_medium, wire_format, replay):
    """Excluded media cannot learn block size or corrupt overlapping GPU hashes."""
    parser = VLLMKVParser()

    def parse(*events):
        if wire_format == "mapping":
            events = [mapping_event(event) for event in events]
        batch = [0, events]
        return parser.parse(msgpack.packb([batch] if replay else batch), "node1")

    # Intentionally different sizes diagnose isolation; not a real Mooncake layout claim.
    excluded_only = parse(_stored_event("rh_gpu", None, [9, 8, 7, 6], 4, non_gpu_medium))
    assert excluded_only is not None and not excluded_only.add_blocks
    assert parser._block_size is None
    assert not parser.remote_to_local_block_hash

    update = parse(
        _stored_event("rh_gpu", None, [1, 2], 2, "GPU"),
        _stored_event("rh_gpu", None, [9, 8, 7, 6], 4, non_gpu_medium),
        ["BlockRemoved", ["rh_gpu"], non_gpu_medium],
        _stored_event("rh_child", "rh_gpu", [3, 4], 2, "GPU"),
    )
    assert update is not None
    gpu_only = VLLMKVParser().parse(
        msgpack.packb(
            [
                0,
                [
                    _stored_event("parent", None, [1, 2], 2, "GPU"),
                    _stored_event("child", "parent", [3, 4], 2, "GPU"),
                ],
            ]
        ),
        "node1",
    )
    assert update.add_blocks == gpu_only.add_blocks
    assert not update.remove_blocks
    assert parser._block_size == 2
    assert parser.remote_to_local_block_hash["rh_gpu"] == update.add_blocks[Layer.GPU][0]

    removed = parse(["BlockRemoved", ["rh_gpu"], "GPU"])
    assert removed is not None
    assert removed.remove_blocks == {Layer.GPU: update.add_blocks[Layer.GPU][:1]}
    assert "rh_gpu" not in parser.remote_to_local_block_hash
    cleared = parse(["AllBlocksCleared"])
    assert cleared is not None and cleared.clear_all


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "medium,level", [("cpu", logging.INFO), ("SSD", logging.INFO), ("nvme", logging.WARNING), ("", logging.WARNING)]
)
def test_filtered_counts_and_summary_are_per_node_medium_and_throttled(caplog, monkeypatch, medium, level):
    parser = VLLMKVParser()
    now = [100.0]
    monkeypatch.setattr(
        "uni_agent.agent_aware_router.collectors.parse.vllm.kv.time",
        SimpleNamespace(monotonic=lambda: now[0]),
    )
    module_logger = logging.getLogger(VLLMKVParser.__module__)
    monkeypatch.setattr(module_logger, "propagate", False)
    module_logger.addHandler(caplog.handler)

    def parse(node_id, event):
        update = parser.parse(msgpack.packb([0, [event]]), node_id)
        assert update is not None and not update.add_blocks and not update.remove_blocks

    try:
        with caplog.at_level(logging.INFO, logger=VLLMKVParser.__module__):
            parse("node1", ["BlockStored", [101, 102], None, [1, 2, 3, 4], 2, None, medium])
            now[0] = 101.0
            parse("node1", ["BlockRemoved", [101], medium.upper()])
            now[0] = 102.0
            parse("node2", ["BlockRemoved", [201], medium])
            now[0] = 130.0
            parse("node1", _stored_event(103, None, [5, 6], 2, medium))
            now[0] = 131.0
            other_medium = "SSD" if medium.upper() == "CPU" else "CPU"
            parse("node1", _stored_event(104, None, [7, 8], 2, other_medium))
    finally:
        module_logger.removeHandler(caplog.handler)

    assert parser._filtered_event_counts[("node1", medium.upper())] == {
        "stored_events": 2,
        "removed_events": 1,
        "stored_blocks": 3,
        "removed_blocks": 1,
    }
    assert parser._filtered_event_counts[("node2", medium.upper())] == {"removed_events": 1, "removed_blocks": 1}
    assert parser._filtered_event_counts[("node1", other_medium)] == {"stored_events": 1, "stored_blocks": 1}
    records = [record for record in caplog.records if record.name == VLLMKVParser.__module__]
    assert len(records) == 4
    assert all(record.levelno == level for record in records[:3])
    assert "node=node1" in records[0].getMessage()
    assert "node=node2" in records[1].getMessage()
    assert "stored_events=2 removed_events=1 stored_blocks=3 removed_blocks=1" in records[2].getMessage()
    assert f"medium={medium.upper()!r}" in records[2].getMessage()
    assert "only GPU residency is tracked" in records[2].getMessage()
    assert records[3].levelno == logging.INFO
    assert f"medium={other_medium!r}" in records[3].getMessage()
    assert parser._block_size is None and not parser.remote_to_local_block_hash


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("medium", [None, "GPU", "gpu"])
def test_legacy_and_gpu_medium_preserve_gpu_updates(medium):
    """Keep missing-medium compatibility and GPU case folding."""
    parser = VLLMKVParser()
    payload = [0, [_stored_event("rh", None, [1, 2], 2, medium)]]

    update = parser.parse(msgpack.packb(payload), "node1")

    assert update is not None
    assert Layer.GPU in update.add_blocks
    assert Layer.CPU not in update.add_blocks
    removed = parser.parse(msgpack.packb([0, [["BlockRemoved", ["rh"], medium]]]), "node1")
    assert removed is not None and removed.remove_blocks == {Layer.GPU: update.add_blocks[Layer.GPU]}


@pytest.mark.cpu
@pytest.mark.level0
def test_clear_event_sets_clear_all():
    """An AllBlocksCleared event marks the update for a full replica clear."""
    parser = VLLMKVParser()
    payload = [0, [["clear"]]]

    update = parser.parse(msgpack.packb(payload), "node1")

    assert update is not None
    assert update.clear_all is True


@pytest.mark.cpu
@pytest.mark.level0
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


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("medium", [None, "GPU", "CPU"])
def test_mapping_matches_array_normalization(medium):
    """Field-named events normalize identically, including real upstream tags."""
    array_events = [
        ["BlockStored", [101, 102], 100, [1, 2, 3, 4], 2, None, medium],
        ["BlockRemoved", [101], medium, 0],
        ["AllBlocksCleared"],
    ]
    mapping_events = [
        {
            "type": "BlockStored",
            "block_hashes": [101, 102],
            "parent_block_hash": 100,
            "token_ids": [1, 2, 3, 4],
            "block_size": 2,
            "medium": medium,
            "group_idx": 0,
            "kv_cache_spec_kind": "full_attention",
            "future_field": "ignored",
        },
        {"type": "BlockRemoved", "block_hashes": [101], "medium": medium, "group_idx": 0},
        {"type": "AllBlocksCleared"},
    ]
    if medium is None:
        for event in mapping_events:
            event.pop("medium", None)
    expected = KVCacheEvent.from_raw([0, array_events, 0], "node1")
    assert len(expected) == 3
    assert KVCacheEvent.from_raw([0, mapping_events, 0], "node1") == expected


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("replay", [False, True])
def test_wire_events_update_hash_mapping_and_remove(replay):
    """Mapping events preserve parent chains across batches and remove hashes."""
    parser = VLLMKVParser()
    batches = [
        [0, [mapping_event(["BlockStored", [101], None, [1, 2], 2, None, "GPU"])], 0],
        [1, [mapping_event(["BlockStored", [102, 103], 101, [3, 4, 5, 6], 2, None, "GPU"])], 0],
    ]
    payloads = [batches] if replay else batches
    added = []
    for payload in payloads:
        update = parser.parse(msgpack.packb(payload), "node1")
        assert update is not None
        added.extend(update.add_blocks[Layer.GPU])
    assert len(added) == 3
    assert list(parser.remote_to_local_block_hash.values()) == added
    removed = parser.parse(msgpack.packb([2, [mapping_event(["BlockRemoved", [102], "GPU", 0])], 0]), "node1")
    assert removed is not None
    assert removed.remove_blocks[Layer.GPU] == [added[1]]
    assert set(parser.remote_to_local_block_hash) == {"101", "103"}
    cleared = parser.parse(msgpack.packb([3, [mapping_event(["AllBlocksCleared"])]]), "node1")
    assert cleared is not None and cleared.clear_all


@pytest.mark.cpu
@pytest.mark.level0
def test_bad_mapping_does_not_discard_valid_sibling(caplog):
    bad_event = {"type": "BlockStored"}
    valid = {"type": "BlockRemoved", "block_hashes": [7]}
    module_logger = logging.getLogger(KVCacheEvent.__module__)
    module_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=KVCacheEvent.__module__):
            events = KVCacheEvent.from_raw([0, [bad_event, valid]], "node1")
    finally:
        module_logger.removeHandler(caplog.handler)
    assert "Skipping malformed KV events (node=node1, count=1, first_type=stored)" in caplog.text
    assert "block_hashes" in caplog.text
    assert len(events) == 1
    assert events[0].event_type == "removed"
    assert events[0].block_hashes == ["7"]


@pytest.mark.cpu
@pytest.mark.level0
def test_malformed_events_emit_one_warning_per_batch_and_keep_valid_sibling(caplog, monkeypatch):
    bad_stored = {
        "type": "BlockStored",
        "block_hashes": "private-event-value",
        "parent_block_hash": None,
        "token_ids": [1, 2],
        "block_size": 2,
    }
    bad_removed = {"type": "BlockRemoved", "block_hashes": b"private-event-value"}
    valid = {"type": "BlockRemoved", "block_hashes": [7]}
    module_logger = logging.getLogger(KVCacheEvent.__module__)
    monkeypatch.setattr(module_logger, "propagate", False)
    module_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=KVCacheEvent.__module__):
            events = KVCacheEvent.from_raw([0, [bad_stored, bad_removed, valid]], "node1")
            records = [record for record in caplog.records if record.name == KVCacheEvent.__module__]
            assert len(records) == 1 and records[0].levelno == logging.WARNING
            message = records[0].getMessage()
            assert "node=node1, count=2, first_type=stored" in message
            assert "block_hashes must be a list or tuple" in message
            assert "private-event-value" not in message
            assert len(events) == 1 and events[0].block_hashes == ["7"]

            caplog.clear()
            assert len(KVCacheEvent.from_raw([1, [valid]], "node1")) == 1
            assert not caplog.records
    finally:
        module_logger.removeHandler(caplog.handler)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("event_type", ["BlockStored", "BlockRemoved"])
@pytest.mark.parametrize("wire_format", ["array", "mapping"])
@pytest.mark.parametrize("bad_hashes", ["101", b"101", {"101": "ignored"}], ids=["str", "bytes", "mapping"])
def test_bad_hash_collection_preserves_gpu_state_and_valid_sibling(event_type, wire_format, bad_hashes):
    """Malformed hash collections cannot overwrite or remove unrelated GPU entries."""
    parser = VLLMKVParser()

    def parse(*events):
        if wire_format == "mapping":
            events = [mapping_event(event) for event in events]
        return parser.parse(msgpack.packb([0, list(events)]), "node1")

    initial = ["BlockStored", [1, 0, 101, 49, 48], None, list(range(10)), 2, None, "GPU"]
    assert parse(initial) is not None
    original_hashes = parser.remote_to_local_block_hash.copy()
    if event_type == "BlockStored":
        bad = [event_type, bad_hashes, None, [20, 21, 22, 23, 24, 25], 2, None, "GPU"]
    else:
        bad = [event_type, bad_hashes, "GPU", 0]
    valid = _stored_event(200, None, [10, 11], 2, "GPU")

    update = parse(bad, valid)

    assert update is not None
    assert not update.remove_blocks
    assert len(update.add_blocks[Layer.GPU]) == 1
    assert parser.remote_to_local_block_hash == {**original_hashes, "200": update.add_blocks[Layer.GPU][0]}
