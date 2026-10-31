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

import asyncio
import time
from threading import Event, Thread, Timer
from unittest.mock import Mock

import pytest
from conftest import BLOCK_SIZE, NODE_ID, FakeZMQTransport, kv_payload, make_stored_event, mapping_event

from uni_agent.agent_aware_router.collectors.collector import Collector
from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.collectors.parse.vllm.kv_event import _convert_token_ids
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.types import Layer
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental

WAIT = 0.3


@pytest.mark.cpu
@pytest.mark.level0
def test_stop_timeout_keeps_thread_reference_and_prevents_restart(monkeypatch):
    from uni_agent.agent_aware_router.collectors import collector as module

    collector = _make_collector([])
    collector.start()
    thread = collector._loop_thread
    join = thread.join
    entered, release = Event(), Event()

    def block():
        entered.set()
        release.wait(3)

    collector._loop.call_soon_threadsafe(block)
    assert entered.wait(1)
    submit = asyncio.run_coroutine_threadsafe

    def short_wait(coroutine, loop):
        future = submit(coroutine, loop)
        result = future.result
        patch.setattr(future, "result", lambda timeout: result(0.01))
        return future

    try:
        with monkeypatch.context() as patch:
            patch.setattr(module.asyncio, "run_coroutine_threadsafe", short_wait)
            patch.setattr(thread, "join", lambda timeout: join(0.01))
            with pytest.raises(TimeoutError, match="cleanup timed out"):
                collector.stop()
            assert collector._loop_thread is thread and thread.is_alive()
            assert collector._loop.is_running() and collector._stop_future is not None
            with pytest.raises(RuntimeError, match="stop the collector"):
                collector.start()
    finally:
        release.set()
        try:
            collector.stop()
        finally:
            if thread.is_alive():
                collector._loop.call_soon_threadsafe(collector._loop.stop)
            join(3)
            if not collector._loop.is_closed():
                pending = asyncio.all_tasks(collector._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    collector._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                collector._loop.close()
        assert not thread.is_alive() and collector._loop.is_closed()


@pytest.mark.cpu
@pytest.mark.level0
def test_stop_waits_for_a_thread_whose_event_loop_has_not_started(monkeypatch):
    collector = _make_collector([])
    entered, release = Event(), Event()

    def delayed_loop():
        entered.set()
        release.wait(3)
        collector._loop.run_forever()

    thread = Thread(target=delayed_loop, daemon=True)
    collector._loop_thread = thread
    thread.start()
    assert entered.wait(1)
    join = thread.join
    monkeypatch.setattr(thread, "join", lambda timeout: join(1))
    resume = Timer(0.02, release.set)
    resume.start()
    try:
        collector.stop()
        assert not thread.is_alive()
        assert collector._loop.is_closed()
    finally:
        release.set()
        if not collector._loop.is_closed():
            collector._loop.call_soon_threadsafe(collector._loop.stop)
        join(3)
        resume.join()
        collector.stop()


def _make_collector(payloads, *, group_aware=False):
    return Collector(FakeZMQTransport(payloads, interval=0.05), VLLMKVParser(group_aware=group_aware))


def _block_ids(start: int = 0) -> list[int]:
    """Return a single block's worth of token IDs."""
    return list(range(start, start + BLOCK_SIZE))


def _run(payloads, *, group_aware=False):
    """Start the collector over the given payloads, stop, return (store, collector)."""
    store = DataStore()
    collector = _make_collector(payloads, group_aware=group_aware)
    collector.start()
    time.sleep(WAIT)
    collector.stop()
    return store, collector


class TestVLLMKVEventCollector:
    """Unit tests: vLLM ZMQ KV-cache collector with injected KV events."""

    @pytest.mark.cpu
    @pytest.mark.level0
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

    @pytest.mark.cpu
    @pytest.mark.level0
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

    @pytest.mark.cpu
    @pytest.mark.level0
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

    @pytest.mark.cpu
    @pytest.mark.level0
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

    @pytest.mark.cpu
    @pytest.mark.level0
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


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("group_aware", [False, True], ids=["legacy", "group-aware"])
def test_wire_events_change_store(group_aware, layout, monkeypatch):
    """Both paths preserve valid updates alongside a malformed event."""
    conversion = Mock(wraps=_convert_token_ids)
    monkeypatch.setattr("uni_agent.agent_aware_router.collectors.parse.vllm.kv_event._convert_token_ids", conversion)
    store = DataStore()
    if group_aware:
        store.configure_cache_registry(layout((("full_attention", BLOCK_SIZE),), unit=None))
    events = [
        [
            "BlockStored",
            [101, 102],
            None,
            list(range(2 * BLOCK_SIZE)),
            BLOCK_SIZE,
            None,
            "GPU",
            None,
            None,
            0,
            "full_attention",
        ],
        ["BlockStored", ["bad"], None, [1], 0, None, "GPU"],
        ["BlockRemoved", [102], "GPU", 0],
    ]
    payloads = [kv_payload(mapping_event(events[0])), kv_payload(*(mapping_event(event) for event in events[1:]))]
    store, collector = _run(payloads, group_aware=group_aware)
    assert conversion.call_count == (0 if group_aware else 1)
    assert store.get_block_size() == (None if group_aware else BLOCK_SIZE)
    if group_aware:
        assert store.per_replica_block_counts() == {NODE_ID: 1}
    else:
        assert store.get_kv_block_count() == 1
        assert set(collector._parser.remote_to_local_block_hash) == {"101"}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("medium", ["cpu", "SSD", "nvme", ""])
@pytest.mark.parametrize("remove_gpu", [False, True])
def test_excluded_media_preserve_gpu_hit_and_counts(medium, remove_gpu):
    """Excluded writes/removals cannot hide later GPU eviction or change its hits."""
    tokens = list(range(2 * BLOCK_SIZE))
    gpu_store = ["BlockStored", [101, 102], None, tokens, BLOCK_SIZE, None, "GPU"]
    excluded_store = ["BlockStored", [101], None, tokens, 2 * BLOCK_SIZE, None, medium]
    events = [excluded_store, gpu_store, excluded_store, ["BlockRemoved", [101], medium]]
    payloads = [kv_payload(*(mapping_event(event) for event in events))]
    if remove_gpu:
        # A later eviction must still resolve the GPU mapping after excluded removals.
        payloads.append(kv_payload(mapping_event(["BlockRemoved", [101], "GPU"])))
    store, _ = _run(payloads)
    hashes, _ = get_prefix_hashes_incremental(tokens, BLOCK_SIZE, 0, 0)
    assert store.get_block_size() == BLOCK_SIZE
    assert store.get_layer_prefix_hit_rate(NODE_ID, [str(h) for h in hashes]) == (0.0 if remove_gpu else 1.0)
    assert store.per_replica_block_counts() == {NODE_ID: 1 if remove_gpu else 2}
    assert store._kv._replica_layer_counts[Layer.CPU] == {}
    assert store._kv._replica_layer_counts[Layer.SSD] == {}
