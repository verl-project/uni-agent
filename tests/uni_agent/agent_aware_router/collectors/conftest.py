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

"""Shared fixtures and helpers for collectors unit tests.

Replaces the earlier real-vLLM-service integration test set-up with
injectable fake transports.  Tests run on CPU without any external
vLLM or ZMQ dependency.
"""

from __future__ import annotations

import asyncio
import struct

import msgpack
import pytest

from uni_agent.agent_aware_router.collectors.transport.base import Transport
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.store.per_replica_store import PerReplicaStore
from uni_agent.agent_aware_router.store.per_request_store import PerRequestStore
from uni_agent.agent_aware_router.types import Layer

# ── Shared configuration constants ───────────────────────────────────────

NODE_ID = "127.0.0.1:8000"
BLOCK_SIZE = 16

# Prometheus-format metrics text — covers all assertions in the polling tests.
VLLM_METRICS_TEXT = """\
# HELP vllm:kv_cache_usage_perc GPU KV cache usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc 0.57
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 3
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 7
"""


# ── Fake transports ──────────────────────────────────────────────────────


class FakeHTTPTransport(Transport):
    """Fake HTTP transport — injects pre-canned Prometheus text.

    ``subscribe(handler)`` feeds ``handler(text, NODE_ID)`` periodically
    until cancelled, simulating the real HTTP polling loop.
    """

    is_async = True

    def __init__(self, text: str, interval: float = 1.0) -> None:
        self._text = text
        self._interval = interval

    async def subscribe(self, handler):
        try:
            while True:
                handler(self._text, NODE_ID)
                await asyncio.sleep(self._interval)
        except (asyncio.CancelledError, GeneratorExit):
            pass

    def stop(self) -> None:
        pass


class FakeZMQTransport(Transport):
    """Fake ZMQ transport — injects pre-canned msgpack KV-events payloads.

    ``subscribe(handler)`` replays all payloads in order, then idles until
    cancelled.
    """

    is_async = True

    def __init__(self, payloads: list[bytes], interval: float = 1.0) -> None:
        self._payloads = list(payloads)
        self._interval = interval

    async def subscribe(self, handler):
        try:
            for p in self._payloads:
                handler(p, NODE_ID)
                await asyncio.sleep(self._interval)
            while True:
                await asyncio.sleep(self._interval)
        except (asyncio.CancelledError, GeneratorExit):
            pass

    def stop(self) -> None:
        pass


# ── KV event payload helpers ────────────────────────────────────────────


def _convert_token_ids(raw_ids: list[int], block_size: int) -> bytes:
    """Encode token IDs as uint32 big-endian bytes (one block)."""
    if len(raw_ids) != block_size:
        raise ValueError(f"Expected {block_size} tokens, got {len(raw_ids)}")
    return struct.pack(f">{block_size}I", *raw_ids)


def make_stored_event(
    block_hash: str,
    raw_ids: list[int],
    block_size: int = BLOCK_SIZE,
    parent: str | None = None,
    medium: str = "GPU",
) -> list:
    """Build a BlockStored event entry: [tag, block_hashes, parent, token_ids, block_size, unused, medium]."""
    return ["stored", [block_hash], parent, raw_ids, block_size, None, medium]


def kv_payload(*events: list) -> bytes:
    """Pack one or more events as a single msgpack payload.

    Format: ``[timestamp, [[tag, fields...], ...]]``.
    """
    return msgpack.packb([1234567890, [list(e) for e in events]])


# ── Autouse: reset singleton stores between tests ─────────────────────────


@pytest.fixture(autouse=True)
def _reset_store_singletons():
    """Reset the three singleton stores before each test.

    ``DataStore`` wraps ``PerReplicaStore``, ``KVCacheStore``, and
    ``PerRequestStore`` — all singletons — so state would otherwise
    accumulate across test cases.
    """
    yield
    # KVCacheStore
    kv = KVCacheStore.singleton()
    kv.block_size = None
    kv.replicas_by_block.clear()
    kv._replica_layer_counts = {Layer.GPU: {}, Layer.CPU: {}, Layer.SSD: {}}
    # PerReplicaStore
    pr = PerReplicaStore.singleton()
    pr._data.clear()
    # PerRequestStore
    prq = PerRequestStore.singleton()
    prq.reset()
