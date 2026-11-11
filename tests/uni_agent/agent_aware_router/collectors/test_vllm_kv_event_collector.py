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
from conftest import BLOCK_SIZE, NODE_ID, FakeZMQTransport, kv_payload, make_stored_event, mapping_event

from uni_agent.agent_aware_router.collectors.collector import Collector
from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.collectors.transport.base import TransportMessage
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental

WAIT = 0.3


def _make_collector(payloads):
    return Collector(FakeZMQTransport(payloads, interval=0.05), VLLMKVParser())


def _block_ids(start=0):
    return list(range(start, start + BLOCK_SIZE))


def _run(payloads):
    store = DataStore()
    collector = _make_collector(payloads)
    collector.start()
    time.sleep(WAIT)
    collector.stop()
    return store, collector


@pytest.mark.cpu
@pytest.mark.level0
def test_unsequenced_events_learn_single_full_layout_and_residency():
    store, _ = _run([kv_payload(make_stored_event("rh0", _block_ids()))])
    registry = store.cache_registry_state.registry
    assert registry.local_hash_unit == BLOCK_SIZE
    assert len(registry.groups) == 1 and registry.groups[0].match_kind == "full"
    assert store.per_replica_block_counts() == {NODE_ID: 1}


@pytest.mark.cpu
@pytest.mark.level0
def test_multiple_inferences_accumulate_blocks():
    payloads = [kv_payload(make_stored_event(f"rh{i}", _block_ids(i * BLOCK_SIZE))) for i in range(3)]
    store, _ = _run(payloads)
    assert store.per_replica_block_counts() == {NODE_ID: 3}


@pytest.mark.cpu
@pytest.mark.level0
def test_shared_stateless_parser_preserves_replica_isolation():
    store = DataStore()
    parser = VLLMKVParser()
    prompts = {"a": _block_ids(), "b": _block_ids(100)}
    for node, tokens in prompts.items():
        update = parser.parse(TransportMessage(kv_payload(make_stored_event("same-remote", tokens))), node)
        assert store.apply_kv_update(update)
    for node, tokens in prompts.items():
        query = tokens + [999]
        hashes = [str(h) for h in get_prefix_hashes_incremental(query, BLOCK_SIZE, 0, 0)[0]]
        estimates = store.estimate_prefixes(["a", "b"], hashes, len(query))
        assert estimates[node].estimated_hit_tokens == BLOCK_SIZE
        assert estimates["b" if node == "a" else "a"].estimated_hit_tokens == 0
    assert store.apply_kv_update(parser.parse(TransportMessage(kv_payload(["AllBlocksCleared"])), "a"))
    assert store.per_replica_block_counts().get("a", 0) == 0
    assert store.per_replica_block_counts()["b"] == 1


@pytest.mark.cpu
@pytest.mark.level0
def test_wire_events_change_store(layout):
    """Known group metadata and valid siblings survive a malformed event."""
    store = DataStore()
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
    payloads = [kv_payload(mapping_event(events[0])), kv_payload(*(mapping_event(e) for e in events[1:]))]
    store, _ = _run(payloads)
    assert store.per_replica_block_counts() == {NODE_ID: 1}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "medium,remove_gpu", [("cpu", False), ("SSD", False), ("nvme", False), ("", False), ("cpu", True)]
)
def test_excluded_media_preserve_gpu_hit_and_counts(medium, remove_gpu):
    tokens = list(range(2 * BLOCK_SIZE))
    gpu_store = ["BlockStored", [101, 102], None, tokens, BLOCK_SIZE, None, "GPU"]
    excluded_store = ["BlockStored", [101], None, tokens, 2 * BLOCK_SIZE, None, medium]
    events = [excluded_store, gpu_store, excluded_store, ["BlockRemoved", [101], medium]]
    payloads = [kv_payload(*(mapping_event(event) for event in events))]
    if remove_gpu:
        payloads.append(kv_payload(mapping_event(["BlockRemoved", [101], "GPU"])))
    store, _ = _run(payloads)
    query = tokens + [999]
    hashes = [str(h) for h in get_prefix_hashes_incremental(query, BLOCK_SIZE, 0, 0)[0]]
    estimate = store.estimate_prefixes([NODE_ID], hashes, len(query))[NODE_ID]
    assert estimate.estimated_hit_tokens == (0 if remove_gpu else len(tokens))
    assert store.per_replica_block_counts() == {NODE_ID: 1 if remove_gpu else 2}
