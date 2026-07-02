"""Tests for vLLM ZMQ KV-cache event collection with real vLLM service.

Test flow:
1. Launch a real vLLM model service (Qwen3-4B) with kv-events-config enabled.
2. Create a Collector(ZMQTransport, VLLMKVDecoder) via BUILTIN_REGISTRY.
3. Call start() — the decoder writes KV events to KVCacheStore.
4. Send an inference request to trigger BlockStored events.
5. Verify that KVCacheStore receives block data via ZMQ events.
"""

from __future__ import annotations

import time

import pytest
from conftest import NODE_ID, VLLM_MODEL, ZMQ_REPLAY_PORT, ZMQ_SUB_PORT, send_inference_request

from uni_agent.llm_router.collectors.registry import BUILTIN_REGISTRY
from uni_agent.llm_router.store.kv_cache_store import KVCacheStore

pytestmark = [pytest.mark.st, pytest.mark.gpu]


def _make_collector():
    return BUILTIN_REGISTRY.get_collector(
        "vllm_zmq",
        endpoints={NODE_ID: [f"127.0.0.1:{ZMQ_SUB_PORT}", f"127.0.0.1:{ZMQ_REPLAY_PORT}"]},
    )


class TestVLLMKVEventCollectorWithRealService:
    """Integration tests: vLLM ZMQ KV-cache collector against a live vLLM ZMQ publisher."""

    def test_start_and_kv_store_updated(self, vllm_kv_service):
        """
        Feature: Collector receives ZMQ events and updates KVCacheStore
        Expectation:
            KVCacheStore.block_size is set (learned from first event).
            replicas_by_block is non-empty.
            NODE_ID appears in at least one block's replica set.
        """
        store = KVCacheStore.default()
        collector = _make_collector()

        collector.start()
        time.sleep(5.0)
        send_inference_request(vllm_kv_service, VLLM_MODEL, "hello world")
        time.sleep(5.0)
        collector.stop()

        assert store.block_size is not None, "block_size should be learned from KV events"
        assert store.block_size > 0
        assert len(store.replicas_by_block) > 0, "replicas_by_block should be non-empty after BlockStored events"
        replica_found = any(NODE_ID in replicas for replicas in store.replicas_by_block.values())
        assert replica_found, f"Expected NODE_ID '{NODE_ID}' in at least one block's replica set"

    def test_block_size_learned(self, vllm_kv_service):
        """
        Feature: block_size is learned from the first BlockStored KV event
        Expectation:
            block_size is a positive integer (vLLM default is 16).
        """
        store = KVCacheStore.default()
        collector = _make_collector()

        collector.start()
        time.sleep(5.0)
        send_inference_request(vllm_kv_service, VLLM_MODEL)
        time.sleep(5.0)
        collector.stop()

        assert isinstance(store.block_size, int)
        assert store.block_size > 0
        assert store.block_size == 16, f"Expected block_size=16 (vLLM default), got {store.block_size}"

    def test_multiple_inferences_accumulate_blocks(self, vllm_kv_service):
        """
        Feature: Multiple inference requests accumulate more blocks in the store
        Expectation:
            After multiple requests, replicas_by_block has entries.
        """
        store = KVCacheStore.default()
        collector = _make_collector()

        collector.start()
        time.sleep(5.0)
        for prompt in [
            "What is machine learning?",
            "Explain quantum computing briefly.",
            "Tell me about deep reinforcement learning.",
        ]:
            send_inference_request(vllm_kv_service, VLLM_MODEL, prompt)
            time.sleep(3.0)
        time.sleep(3.0)
        collector.stop()

        assert len(store.replicas_by_block) > 0, "Expected blocks after multiple inferences"

    def test_clear_replica_removes_all_blocks(self, vllm_kv_service):
        """
        Feature: KVCacheStore.clear_replica removes all blocks for a replica
        Expectation:
            After clear_replica, no block in replicas_by_block contains NODE_ID.
        """
        store = KVCacheStore.default()
        collector = _make_collector()

        collector.start()
        time.sleep(5.0)
        send_inference_request(vllm_kv_service, VLLM_MODEL)
        time.sleep(5.0)
        collector.stop()

        if len(store.replicas_by_block) == 0:
            pytest.skip("No blocks received from KV events")

        store.clear_replica(NODE_ID)

        for block_hash, replicas in store.replicas_by_block.items():
            assert NODE_ID not in replicas, (
                f"NODE_ID '{NODE_ID}' should not be in block '{block_hash}' after clear_replica"
            )

    def test_decoder_hash_mapping_populated(self, vllm_kv_service):
        """
        Feature: VLLMKVDecoder.remote_to_local_block_hash is populated after events
        Description:
            Verify that the decoder's hash mapping tracks remote→local block hashes,
            and that every local hash appears in KVCacheStore.replicas_by_block.
        Expectation:
            remote_to_local_block_hash is non-empty.
            All local hashes are present in store.replicas_by_block.
        """
        store = KVCacheStore.default()
        collector = _make_collector()

        collector.start()
        time.sleep(5.0)
        send_inference_request(vllm_kv_service, VLLM_MODEL)
        time.sleep(5.0)
        collector.stop()

        mapping = collector._decoder.remote_to_local_block_hash
        assert len(mapping) > 0, "remote_to_local_block_hash should have entries after processing events"
        for remote_bh, local_bh in mapping.items():
            assert isinstance(remote_bh, str)
            assert isinstance(local_bh, str)
            assert local_bh in store.replicas_by_block, (
                f"Local hash '{local_bh}' from mapping not found in store.replicas_by_block"
            )
