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

"""Unit tests for vLLM ZMQ KV-cache event collection with a fake ZMQ transport.

Test flow:
1. Inject pre-canned msgpack KV events via FakeZMQTransport.
2. Create a Collector(FakeZMQTransport, VLLMKVParser).
3. Call start() — the collector parses events and writes to the KV cache store.
4. Verify that KV cache data is accessible via DataStore.

No real vLLM service is required.
"""

from __future__ import annotations

import time

import pytest
from conftest import BLOCK_SIZE, NODE_ID, FakeZMQTransport, kv_payload, make_stored_event

from uni_agent.agent_aware_router.collectors.collector import Collector
from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.store.data_store import DataStore

pytestmark = [pytest.mark.level0, pytest.mark.cpu]

WAIT = 0.3


def _make_collector(payloads):
    return Collector(FakeZMQTransport(payloads, interval=0.05), VLLMKVParser())


def _block_ids(start: int = 0) -> list[int]:
    """Return a single block's worth of token IDs."""
    return list(range(start, start + BLOCK_SIZE))


def _run(payloads):
    """Start the collector over the given payloads, stop, return (store, collector)."""
    store = DataStore()
    collector = _make_collector(payloads)
    collector.start()
    time.sleep(WAIT)
    collector.stop()
    return store, collector


class TestVLLMKVEventCollector:
    """Unit tests: vLLM ZMQ KV-cache collector with injected KV events."""

    def test_start_and_kv_store_updated(self):
        """
        Feature: Collector receives ZMQ events and updates KV cache store
        Expectation:
            block_size is set (learned from first event).
            At least one block is cached.
            NODE_ID appears in at least one cached block.
        """
        store, _ = _run([kv_payload(make_stored_event("rh0", _block_ids(0)))])

        assert store.get_block_size() is not None, "block_size should be learned from KV events"
        assert store.get_block_size() > 0
        assert store.get_kv_block_count() > 0, "KV cache should have blocks after BlockStored events"
        assert store.kv_node_has_blocks(NODE_ID), f"Expected NODE_ID '{NODE_ID}' in at least one cached block"

    def test_block_size_learned(self):
        """
        Feature: block_size is learned from the first BlockStored KV event
        Expectation:
            block_size is a positive integer (vLLM default is 16).
        """
        store, _ = _run([kv_payload(make_stored_event("rh0", _block_ids(0)))])

        assert isinstance(store.get_block_size(), int)
        assert store.get_block_size() > 0
        assert store.get_block_size() == BLOCK_SIZE, f"Expected block_size={BLOCK_SIZE}, got {store.get_block_size()}"

    def test_multiple_inferences_accumulate_blocks(self):
        """
        Feature: Multiple inference requests accumulate more blocks in the store
        Expectation:
            After multiple block-stored events, the KV cache has entries.
        """
        payloads = [
            kv_payload(make_stored_event("rh0", _block_ids(0))),
            kv_payload(make_stored_event("rh1", _block_ids(BLOCK_SIZE))),
            kv_payload(make_stored_event("rh2", _block_ids(2 * BLOCK_SIZE))),
        ]
        store, _ = _run(payloads)

        assert store.get_kv_block_count() > 0, "Expected blocks after multiple event payloads"
        assert store.get_kv_block_count() == 3, f"Expected 3 blocks (one per event), got {store.get_kv_block_count()}"

    def test_clear_kv_node_removes_all_blocks(self):
        """
        Feature: DataStore.clear_kv_node removes all blocks for a node
        Expectation:
            After clear_kv_node, NODE_ID no longer appears in any cached block.
        """
        store, _ = _run([kv_payload(make_stored_event("rh0", _block_ids(0)))])

        assert store.get_kv_block_count() > 0, "Precondition: blocks should exist before clear"
        store.clear_kv_node(NODE_ID)
        assert not store.kv_node_has_blocks(NODE_ID), (
            f"NODE_ID '{NODE_ID}' should not appear in any block after clear_kv_node"
        )

    def test_parser_hash_mapping_populated(self):
        """
        Feature: VLLMKVParser.remote_to_local_block_hash is populated after events
        Description:
            Verify that the parser's hash mapping tracks remote->local block hashes,
            and that every local hash appears in the KV cache store.
        Expectation:
            remote_to_local_block_hash is non-empty.
            All local hashes are present in the KV cache store.
        """
        store, collector = _run([kv_payload(make_stored_event("rh0", _block_ids(0)))])

        mapping = collector._parser.remote_to_local_block_hash
        assert len(mapping) > 0, "remote_to_local_block_hash should have entries after processing events"
        for remote_bh, local_bh in mapping.items():
            assert isinstance(remote_bh, str)
            assert isinstance(local_bh, str)
            assert store.has_kv_block(local_bh), f"Local hash '{local_bh}' from mapping not found in KV cache store"
