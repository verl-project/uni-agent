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

"""Exercise the upstream replay callbacks with a hybrid cache index."""

import asyncio

import msgpack
import pytest

from tests.uni_agent.agent_aware_router.collectors.zmq_helpers import Publisher
from uni_agent.agent_aware_router.collectors.collector import get_collector
from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.config.collector import CollectorConfig
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.types.kv_cache import GroupReservation
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), 5)


def payload(events):
    return msgpack.packb([0, events, 0])


def tokens_hit(store, replica="a"):
    hashes = list(map(str, get_prefix_hashes_incremental(list(range(17)), 2, 0, 0)[0]))
    return store.estimate_prefixes([replica], hashes, 17)[replica].estimated_hit_tokens


def setup(layout, stored, install_metadata):
    store = DataStore()
    raw = layout()
    install_metadata(store, "a", raw)
    install_metadata(store, "b", raw)
    events = [stored(), stored(1, "mamba", 8, hashes=["r16"])]
    publisher = Publisher()
    collector = get_collector("vllm_zmq", CollectorConfig(), kv_event_endpoints=publisher.endpoints)
    collector._transport._base_retry_delay = 0.01
    collector._transport._max_retry_delay = 0.02
    store.apply_kv_update(VLLMKVParser().parse(payload(events), "b"))
    assert tokens_hit(store, "b") == 16
    return store, publisher, collector, events


def test_replayed_clear_keeps_partial_rebuild_gated(layout, stored, install_metadata):
    async def run():
        store, publisher, collector, events = setup(layout, stored, install_metadata)

        def predict():
            hashes = [f"prediction-{index}" for index in range(4)]
            added = store.pin_request("prediction", "a", hashes, store.dispatch_schema("a"), reservations=())
            store.unpin_request("prediction")
            return added

        try:
            collector.start()
            identity, start = await publisher.requested_start()
            assert start == 0
            await publisher.batch(identity, 10, payload([["AllBlocksCleared"]]))
            await publisher.batch(identity, 11, payload(events))
            await publisher.end(identity)
            await until(lambda: tokens_hit(store) == 16)
            publisher.close_socket("pub")
            await until(lambda: tokens_hit(store) == 0)
            await until(lambda: predict() > 0)
            publisher.reopen("pub")
            identity, start = await publisher.requested_start()
            assert start == 0
            await publisher.batch(identity, 10, payload([["AllBlocksCleared"]]))
            await publisher.batch(identity, 11, payload([events[0]]))
            await until(lambda: store.per_replica_block_counts().get("a") == 4)
            assert tokens_hit(store) == 0
            await publisher.batch(identity, 12, payload([events[1]]))
            await until(lambda: store.per_replica_block_counts().get("a") == 5)
            assert tokens_hit(store) == 0
            assert predict() == 0
            await publisher.end(identity)
            await until(lambda: tokens_hit(store) == 16)
            assert predict() > 0
            assert tokens_hit(store, "b") == 16
        finally:
            collector.stop(timeout=2)
            publisher.close()

    asyncio.run(run())


def test_hybrid_replay_overlap_and_gap_keep_groups_and_replicas_independent(layout, stored, removed, install_metadata):
    async def run():
        store, publisher, collector, events = setup(layout, stored, install_metadata)
        try:
            collector.start()
            identity, _ = await publisher.requested_start()
            await publisher.batch(identity, 5, payload(events))
            await publisher.end(identity)
            await until(lambda: tokens_hit(store) == 16)
            await asyncio.wait_for(publisher.pub.recv(), 3)
            # Overlap is deduplicated before it reaches per-group reference counts.
            await publisher.publish(5, payload(events))
            await publisher.publish(6, payload([removed(1, ["r16"])]))
            await until(lambda: tokens_hit(store) == 0)
            assert store.per_replica_block_counts()["a"] == 4
            assert tokens_hit(store, "b") == 16
            # A forward gap starts a fresh replay and hides the incomplete view.
            await publisher.publish(8, payload(events))
            identity, start = await publisher.requested_start()
            assert start == 0
            assert tokens_hit(store) == 0
            assert tokens_hit(store, "b") == 16
            await publisher.batch(identity, 8, payload(events))
            await publisher.end(identity)
            await until(lambda: tokens_hit(store) == 16)
            assert store.per_replica_block_counts()["a"] == 5
        finally:
            collector.stop(timeout=2)
            publisher.close()

    asyncio.run(run())


def test_invalid_payload_invalidates_hybrid_estimates_without_losing_request_holds(layout, stored, install_metadata):
    async def run():
        store, publisher, collector, events = setup(layout, stored, install_metadata)
        try:
            collector.start()
            identity, _ = await publisher.requested_start()
            await publisher.batch(identity, 0, payload(events))
            await publisher.end(identity)
            await until(lambda: tokens_hit(store) == 16)
            await asyncio.wait_for(publisher.pub.recv(), 3)
            store.pin_request(
                "running",
                "a",
                [],
                store.dispatch_schema("a"),
                reservations=(GroupReservation(0, ("held",)), GroupReservation(1, private_blocks=2)),
                predict=False,
            )
            await publisher.publish(1, b"\xc1")
            await until(lambda: tokens_hit(store) == 0)
            assert tokens_hit(store, "b") == 16
            assert store.inflight_reserved_blocks("a") == 3
            assert store.get_marker("running") is not None
        finally:
            collector.stop(timeout=2)
            publisher.close()

    asyncio.run(run())
