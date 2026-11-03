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

"""Canonical token estimates after unsequenced Stored, Removed, and Clear."""

import time

import pytest
from conftest import BLOCK_SIZE, NODE_ID, FakeZMQTransport, kv_payload, make_stored_event, mapping_event

from uni_agent.agent_aware_router.collectors.collector import Collector
from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.collectors.transport.base import TransportMessage
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental

pytestmark = [pytest.mark.level0, pytest.mark.cpu]
LONG_IDS = list(range(2 * BLOCK_SIZE))


def estimate(store, tokens):
    hashes = [str(h) for h in get_prefix_hashes_incremental(tokens, BLOCK_SIZE, 0, 0)[0]]
    return store.estimate_prefixes([NODE_ID], hashes, len(tokens))[NODE_ID]


def test_prefix_estimate_keeps_final_token_computation_boundary():
    collector = Collector(
        FakeZMQTransport(
            [
                kv_payload(make_stored_event("rh0", LONG_IDS[:BLOCK_SIZE])),
                kv_payload(make_stored_event("rh1", LONG_IDS[BLOCK_SIZE:], parent="rh0")),
            ],
            interval=0.05,
        ),
        VLLMKVParser(),
    )
    collector.start()
    try:
        time.sleep(0.3)
        store = DataStore()
        for query_tokens, hit_tokens in [
            (BLOCK_SIZE, 0),
            (2 * BLOCK_SIZE, BLOCK_SIZE),
            (2 * BLOCK_SIZE + 1, 2 * BLOCK_SIZE),
        ]:
            result = estimate(store, list(range(query_tokens)))
            assert result.estimated_hit_tokens == hit_tokens
            assert result.hit_rate == pytest.approx(hit_tokens / query_tokens)
    finally:
        collector.stop()


@pytest.mark.parametrize("removed_hash,expected_hit", [(102, BLOCK_SIZE), (101, 0)])
@pytest.mark.parametrize("metadata_available", [False, True], ids=["old-untagged", "native-layout"])
def test_wire_events_change_prefix_hit_rate(removed_hash, expected_hit, metadata_available, layout):
    store = DataStore()
    if metadata_available:
        store.configure_cache_registry(layout((("full_attention", BLOCK_SIZE),), unit=None))
    collector = Collector(FakeZMQTransport([]), VLLMKVParser())

    def apply(event):
        if metadata_available and event[0] == "BlockStored":
            event = [*event, None, None, 0, "full_attention"]
        elif metadata_available and event[0] == "BlockRemoved":
            event = [*event, 0]
        update = collector._parser.parse(TransportMessage(kv_payload(mapping_event(event))), NODE_ID)
        collector._write_kv_update(update)

    try:
        apply(["BlockStored", [101], None, LONG_IDS[:BLOCK_SIZE], BLOCK_SIZE, None, "GPU"])
        apply(["BlockStored", [102], 101, LONG_IDS[BLOCK_SIZE:], BLOCK_SIZE, None, "GPU"])
        assert estimate(store, LONG_IDS + [999]).estimated_hit_tokens == len(LONG_IDS)
        apply(["BlockRemoved", [removed_hash], "GPU"])
        assert estimate(store, LONG_IDS + [999]).estimated_hit_tokens == expected_hit
        apply(["AllBlocksCleared"])
        assert estimate(store, LONG_IDS + [999]).estimated_hit_tokens == 0
    finally:
        collector.stop()
