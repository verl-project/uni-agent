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

"""Native KV frames preserve ordered event semantics without cache state."""

from __future__ import annotations

import logging

import msgpack
import pytest
from conftest import mapping_event

from uni_agent.agent_aware_router.collectors.parse.vllm import kv_event as kv_decoder
from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.collectors.transport.base import TransportMessage

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def _stored_event(block_hash, parent, token_ids, block_size, medium):
    return ["BlockStored", [block_hash], parent, token_ids, block_size, None, medium]


def _frame(batch, *, sequence=None):
    return TransportMessage(msgpack.packb(batch), sequence=sequence)


@pytest.mark.parametrize("wire_format", ["array", "mapping"])
@pytest.mark.parametrize("replay", [False, True])
def test_mixed_media_preserve_order_and_identity_fields(wire_format, replay):
    """The parser preserves domains; the store decides which events establish residency."""
    non_gpu_medium = "cpu"
    parser = VLLMKVParser()
    events = [
        _stored_event("rh_gpu", None, [1, 2], 2, "GPU"),
        _stored_event("rh_gpu", None, [9, 8, 7, 6], 4, non_gpu_medium),
        ["BlockRemoved", ["rh_gpu"], non_gpu_medium],
        _stored_event("rh_child", "rh_gpu", [3, 4], 2, "GPU"),
    ]
    if wire_format == "mapping":
        events = [mapping_event(event) for event in events]
    batch = [0, events]
    update = parser.parse(_frame([batch] if replay else batch, sequence=11), "node1")

    assert update.sequence == 11 and update.stream_error is None
    assert [event.event_type for event in update.events] == ["stored", "stored", "removed", "stored"]
    assert [event.medium for event in update.events] == ["GPU", non_gpu_medium, non_gpu_medium, "GPU"]
    assert [event.block_hashes for event in update.events] == [["rh_gpu"], ["rh_gpu"], ["rh_gpu"], ["rh_child"]]
    assert update.events[1].raw_token_ids == (9, 8, 7, 6)
    assert update.events[1].block_size == 4
    assert update.events[-1].parent_block_hash == "rh_gpu"
    assert update.events[-1].raw_token_ids == (3, 4)
    assert all(event.node_id == "node1" for event in update.events)


@pytest.mark.parametrize("medium", ["cpu", "CPU", "ssd", "SSD", "nvme", ""])
def test_storage_domain_fields_survive_normalization(medium):
    """Remote or owned events must not become ordinary local GPU events."""
    batch = {
        "events": [
            {
                **mapping_event(_stored_event(101, None, [1, 2], 2, medium)),
                "locality": "REMOTE",
                "ownership": "external-store",
            },
            {
                "type": "BlockRemoved",
                "block_hashes": [101],
                "medium": medium,
                "locality": "REMOTE",
                "ownership": "external-store",
            },
        ]
    }
    events = VLLMKVParser().parse(_frame(batch), "node2").events

    assert len(events) == 2
    assert all(event.medium == medium for event in events)
    assert all(event.locality == "REMOTE" and event.ownership == "external-store" for event in events)
    assert all(event.node_id == "node2" and event.block_hashes == ["101"] for event in events)


@pytest.mark.parametrize("medium", [None, "GPU", "gpu"])
def test_legacy_medium_defaults_to_gpu_without_inventing_a_layout(medium):
    parser = VLLMKVParser()
    events = [_stored_event("rh", None, [1, 2], 2, medium), ["BlockRemoved", ["rh"], medium]]

    update = parser.parse(_frame([0, events]), "node1")

    assert [event.medium for event in update.events] == ["GPU" if medium is None else medium] * 2
    assert all(event.group_idx is None and event.kind is None for event in update.events)
    assert update.events[0].raw_token_ids == (1, 2)
    assert update.events[1].block_hashes == ["rh"]


def test_clear_event_remains_an_ordered_replica_event():
    parser = VLLMKVParser()
    before = _stored_event("before", None, [1, 2], 2, "GPU")
    after = _stored_event("after", None, [3, 4], 2, "GPU")

    update = parser.parse(_frame([0, [before, ["clear"], after]]), "node1")

    assert [event.event_type for event in update.events] == ["stored", "clear", "stored"]
    assert update.events[1].node_id == "node1" and update.events[1].is_clear
    assert [event.block_hashes for event in update.events] == [["before"], [], ["after"]]


def test_parse_failure_reports_stream_error_and_real_exception():
    """Invalid framed msgpack must invalidate the stream and log the actual error."""
    records = []
    capture = logging.Handler()
    capture.emit = records.append
    capture.setLevel(logging.WARNING)
    module_logger = logging.getLogger(VLLMKVParser.__module__)
    module_logger.addHandler(capture)
    try:
        update = VLLMKVParser().parse(TransportMessage(b"\xc1\xc1\xc1", sequence=17), "node1")
    finally:
        module_logger.removeHandler(capture)

    assert update.stream_error == "invalid_kv_event_payload" and not update.events
    assert update.sequence == 17
    text = "\n".join(record.getMessage() for record in records)
    assert "{exc}" not in text and "node1" in text
    assert "FormatError" in text and "head=c1c1c1" in text


@pytest.mark.parametrize("medium", [None, "GPU", "CPU"])
def test_mapping_matches_array_normalization(medium):
    """Named and positional fields retain the same native identities and full token range."""
    array_events = [
        ["BlockStored", [101, 102], 100, [1, 2, 3, 4], 2, None, medium, None, None, 0, "full_attention"],
        ["BlockRemoved", [101], medium, 0],
        ["AllBlocksCleared"],
        ["BlockStored", [103], None, list(range(17)), 8, None, medium, None, [None], 1, "mamba"],
    ]
    mapping_events = [mapping_event(event) for event in array_events]
    mapping_events[0]["future_field"] = "ignored"
    if medium is None:
        for event in mapping_events:
            event.pop("medium", None)
    parser = VLLMKVParser()
    expected = parser.parse(_frame([0, array_events, 0]), "node1").events
    actual = parser.parse(_frame({"ts": 0, "events": mapping_events}), "node1").events

    assert actual == expected and len(expected) == 4
    assert expected[0].parent_block_hash == "100"
    assert expected[-1].raw_token_ids == tuple(range(17))
    assert expected[-1].block_hashes == ["103"]
    assert (expected[-1].group_idx, expected[-1].kind, expected[-1].block_size) == (1, "mamba", 8)


@pytest.mark.parametrize("replay", [False, True])
def test_parent_and_removed_identities_do_not_depend_on_parser_history(replay):
    parser = VLLMKVParser()
    parent = mapping_event(["BlockStored", [101], None, [1, 2], 2, None, "GPU"])
    child = mapping_event(["BlockStored", [102, 103], 101, [3, 4, 5, 6], 2, None, "GPU"])
    batches = [[0, [parent], 0], [1, [child], 0]]
    events = [
        event
        for payload in ([batches] if replay else batches)
        for event in parser.parse(_frame(payload), "node1").events
    ]

    assert [event.block_hashes for event in events] == [["101"], ["102", "103"]]
    assert events[1].parent_block_hash == "101"
    child_frame = _frame([1, [child], 0])
    assert parser.parse(child_frame, "node1") == VLLMKVParser().parse(child_frame, "node1")
    assert parser.parse(child_frame, "node2").events[0].parent_block_hash == "101"
    removed = parser.parse(_frame([2, [mapping_event(["BlockRemoved", [102], "GPU", 0])], 0]), "node1")
    assert removed.events[0].block_hashes == ["102"] and removed.events[0].is_remove
    cleared = parser.parse(_frame([3, [mapping_event(["AllBlocksCleared"])]]), "node1")
    assert len(cleared.events) == 1 and cleared.events[0].is_clear
    assert parser.parse(child_frame, "node1").events == [events[1]]


def test_bad_mapping_does_not_discard_valid_sibling(caplog):
    bad_event = {"type": "BlockStored"}
    valid = {"type": "BlockRemoved", "block_hashes": [7]}
    module_logger = kv_decoder.logger
    module_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=module_logger.name):
            update = VLLMKVParser().parse(_frame([0, [bad_event, ["FutureEvent", "unknown"], valid]]), "node1")
    finally:
        module_logger.removeHandler(caplog.handler)
    assert "Skipping malformed KV events (node=node1, count=1, first_type=stored)" in caplog.text
    assert "block_hashes" in caplog.text
    assert len(update.events) == 1 and update.events[0].event_type == "removed"
    assert update.events[0].block_hashes == ["7"]


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
    module_logger = kv_decoder.logger
    monkeypatch.setattr(module_logger, "propagate", False)
    module_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=module_logger.name):
            parser = VLLMKVParser()
            update = parser.parse(_frame([0, [bad_stored, bad_removed, valid]]), "node1")
            records = [record for record in caplog.records if record.name == module_logger.name]
            assert len(records) == 1 and records[0].levelno == logging.WARNING
            message = records[0].getMessage()
            assert "node=node1, count=2, first_type=stored" in message
            assert "block_hashes must be a list or tuple" in message
            assert "private-event-value" not in message
            assert len(update.events) == 1 and update.events[0].block_hashes == ["7"]
            caplog.clear()
            assert len(parser.parse(_frame([1, [valid]]), "node1").events) == 1
            assert not caplog.records
    finally:
        module_logger.removeHandler(caplog.handler)


@pytest.mark.parametrize("event_type", ["BlockStored", "BlockRemoved"])
@pytest.mark.parametrize("wire_format", ["array", "mapping"])
@pytest.mark.parametrize("bad_hashes", ["101", b"101", {"101": "ignored"}], ids=["str", "bytes", "mapping"])
def test_bad_hash_collection_is_rejected_without_losing_valid_sibling(event_type, wire_format, bad_hashes):
    if event_type == "BlockStored":
        bad = [event_type, bad_hashes, None, [20, 21, 22, 23, 24, 25], 2, None, "GPU"]
    else:
        bad = [event_type, bad_hashes, "GPU", 0]
    valid = _stored_event(200, None, [10, 11], 2, "GPU")
    events = [bad, valid]
    if wire_format == "mapping":
        events = [mapping_event(event) for event in events]

    update = VLLMKVParser().parse(_frame([0, events]), "node1")

    assert len(update.events) == 1
    assert update.events[0].block_hashes == ["200"] and update.events[0].raw_token_ids == (10, 11)
    assert update.events[0].is_store and update.stream_error is None


@pytest.mark.parametrize("bad_hash", [True, None, [], {}, 1.5], ids=["bool", "null", "list", "mapping", "float"])
@pytest.mark.parametrize("event_type", ["BlockStored", "BlockRemoved"])
@pytest.mark.parametrize("wire_format", ["array", "mapping"])
def test_invalid_hash_identity_does_not_escape_row_validation(bad_hash, event_type, wire_format):
    if event_type == "BlockStored":
        event = _stored_event(bad_hash, None, [1, 2], 2, "GPU")
    else:
        event = [event_type, [bad_hash], "GPU"]
    if wire_format == "mapping":
        event = mapping_event(event)

    update = VLLMKVParser().parse(_frame([0, [event]]), "node1")

    assert not update.events and update.stream_error is None


def test_raw_payload_requires_the_collector_frame_boundary():
    with pytest.raises(TypeError, match="requires a TransportMessage"):
        VLLMKVParser().parse(msgpack.packb([0, []]), "node1")


@pytest.mark.parametrize(
    "status,reset,error",
    [
        ("new_stream", True, None),
        ("no_continuity", False, "event_stream_no_continuity"),
        ("unreachable", False, "event_stream_unreachable"),
    ],
)
def test_transport_status_is_translated_without_decoding_payload(status, reset, error):
    frame = TransportMessage(b"\xc1", sequence=23, status=status)

    update = VLLMKVParser().parse(frame, "node1")

    assert update.node_id == "node1" and update.sequence == 23 and not update.events
    assert update.reset_stream is reset
    assert update.stream_error == error


def test_unexpected_decode_error_is_not_swallowed(monkeypatch):
    def broken_unpack(payload, **kwargs):
        raise RuntimeError("unexpected decode error")

    monkeypatch.setattr("uni_agent.agent_aware_router.collectors.parse.vllm.kv.msgpack.unpackb", broken_unpack)
    with pytest.raises(RuntimeError, match="unexpected decode error"):
        VLLMKVParser().parse(_frame([0, []]), "node1")
