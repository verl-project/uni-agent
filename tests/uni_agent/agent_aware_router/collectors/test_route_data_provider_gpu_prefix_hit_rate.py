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

"""Tests for DataStore.get_layer_prefix_hit_rate via injected KV events.

Test flow:
1. Create Collector(FakeZMQTransport, VLLMKVParser).
2. Call start() to begin event processing.
3. Inject ChainBlocks of a "long" prompt through the fake transport.
4. Compute the prefix-hash chains of the long and short prompts locally.
5. Call DataStore().get_layer_prefix_hit_rate(node_id, hash_chain, Layer.GPU)
   and verify the results.

No real vLLM service is required.
"""

from __future__ import annotations

import time

import pytest
from conftest import BLOCK_SIZE, NODE_ID, FakeZMQTransport, kv_payload, make_stored_event

from uni_agent.agent_aware_router.collectors.collector import Collector
from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.types import Layer
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental

pytestmark = [pytest.mark.level0, pytest.mark.cpu]

WAIT = 0.3

# Two full blocks of tokens: a "long" prompt and its strict prefix "short".
LONG_IDS = list(range(2 * BLOCK_SIZE))
SHORT_IDS = LONG_IDS[:BLOCK_SIZE]


def _make_collector(payloads):
    return Collector(FakeZMQTransport(payloads, interval=0.05), VLLMKVParser())


def _inject_long_prompt(store: DataStore):
    """Inject chained BlockStored events for the long prompt into the KV store.

    The two blocks share a chain: block 1's parent is block 0's remote hash,
    so the parser computes local hashes identical to
    ``get_prefix_hashes_incremental`` on the same token IDs.
    """
    collector = _make_collector(
        [
            kv_payload(make_stored_event("rh0", LONG_IDS[:BLOCK_SIZE], parent=None)),
            kv_payload(make_stored_event("rh1", LONG_IDS[BLOCK_SIZE:], parent="rh0")),
        ]
    )
    collector.start()
    time.sleep(WAIT)
    collector.stop()


def _hash_chain(token_ids: list[int]) -> list[str]:
    """Compute the full-block prefix hash chain for token IDs."""
    hashes, _ = get_prefix_hashes_incremental(token_ids, BLOCK_SIZE, 0, 0)
    return [str(h) for h in hashes]


class TestGpuPrefixHitRate:
    """Unit tests: DataStore.get_layer_prefix_hit_rate with injected KV events."""

    def test_prefix_hit_rate_with_partial_match(self):
        """
        Feature: get_layer_prefix_hit_rate returns 100% hit rate for a shorter prompt
        Description:
            1. Inject KV cache blocks for a long prompt A.
            2. Call get_layer_prefix_hit_rate with the hash chain of prompt B
               that is a strict prefix of A.
        Expectation:
            Since B's blocks are a subset of A's cached blocks, all of B's prefix
            blocks are cached -> hit_rate = 100.
        """
        store = DataStore()
        _inject_long_prompt(store)

        short_chain = _hash_chain(SHORT_IDS)
        assert len(short_chain) < len(_hash_chain(LONG_IDS)), (
            f"Short prompt should have fewer blocks, got short={len(short_chain)}"
        )
        hit = store.get_layer_prefix_hit_rate(NODE_ID, short_chain, Layer.GPU)

        assert hit == 1.0, f"Expected hit_rate=1.0 for prefix match, got {hit}"

    def test_prefix_hit_rate_with_full_match(self):
        """
        Feature: get_layer_prefix_hit_rate returns 100% when the whole chain is cached
        Description:
            1. Inject KV cache blocks for a long prompt.
            2. Call get_layer_prefix_hit_rate with the prompt's full hash chain.
        Expectation:
            All cached blocks are hit -> hit_rate = 1.0.
        """
        store = DataStore()
        _inject_long_prompt(store)

        long_chain = _hash_chain(LONG_IDS)
        assert len(long_chain) == 2, f"Expected 2 full blocks, got {len(long_chain)}"
        hit = store.get_layer_prefix_hit_rate(NODE_ID, long_chain, Layer.GPU)

        assert 0.0 <= hit <= 1.0, f"Hit rate should be in [0.0, 1.0], got {hit}"
        assert hit == 1.0, f"Expected hit_rate=1.0 for fully cached chain, got {hit}"
