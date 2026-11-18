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

"""Exercise publisher replacement through the real collector factory and store."""

import asyncio

import msgpack
import pytest
import zmq
from conftest import BLOCK_SIZE, kv_payload, make_stored_event

from tests.uni_agent.agent_aware_router.collectors.zmq_helpers import Publisher
from uni_agent.agent_aware_router.collectors.collector import get_collector
from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.config.collector import CollectorConfig
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.types.kv_cache import GroupReservation
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def create_store(layout, block_size, install_metadata):
    store = DataStore()
    install_metadata(store, "a", layout((("full_attention", block_size),), unit=None))
    install_metadata(store, "other", layout((("full_attention", block_size),), unit=None))
    return store


def seed(store, replica_id, tokens):
    block = store.hash_schema()[1]
    hashes = [f"seed-{replica_id}-{end}" for end in range(block, len(tokens) + 1, block)]
    event = ["BlockStored", hashes, None, tokens, block, None, "GPU"]
    update = VLLMKVParser().parse(msgpack.packb([0, [event]]), replica_id)
    store.apply_kv_update(update)


def pin(store, request_id, replica_id, hashes):
    schema = store.dispatch_schema(replica_id)
    return store.pin_request(
        request_id, replica_id, hashes, schema, reservations=(GroupReservation(0, tuple(hashes)),), predict=False
    )


def predict(store, replica_id, hashes):
    schema = store.dispatch_schema(replica_id)
    request = "completed-prediction"
    added = store.pin_request(request, replica_id, hashes, schema, reservations=(GroupReservation(0, tuple(hashes)),))
    store.unpin_request(request)
    return added


def hit_rate(store, replica_id, hashes):
    unit = store.hash_schema()[1]
    if not unit or not hashes:
        return 0.0
    length = len(hashes) * unit
    return store.estimate_prefixes([replica_id], hashes, length + 1)[replica_id].estimated_hit_tokens / length


def has_blocks(store, replica_id):
    return bool(store.per_replica_block_counts().get(replica_id))


def remote_hashes(store, replica_id):
    index = store._kv.group_caches.get(replica_id)
    return set(index.anchors) if index is not None else set()


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), 5)


@pytest.mark.parametrize("new_sequence", [0, 200], ids=["numbering-reset", "already-past-old-cursor"])
def test_same_address_publisher_restart_rebuilds_only_its_cache(new_sequence, layout, install_metadata):
    async def run():
        publisher = Publisher()
        addresses = publisher.addresses
        collector = get_collector(
            "vllm_zmq",
            CollectorConfig(),
            kv_event_endpoints={"a": [*addresses, "zmq", "topic"]},
        )
        collector._transport._base_retry_delay = 0.01
        collector._transport._max_retry_delay = 0.02
        store = create_store(layout, BLOCK_SIZE, install_metadata)
        old_tokens = list(range(BLOCK_SIZE))
        new_tokens = list(range(BLOCK_SIZE, 2 * BLOCK_SIZE))
        estimated_tokens = list(range(2 * BLOCK_SIZE, 3 * BLOCK_SIZE))
        old_hashes, _ = get_prefix_hashes_incremental(old_tokens, BLOCK_SIZE, 0, 0)
        new_hashes, _ = get_prefix_hashes_incremental(new_tokens, BLOCK_SIZE, 0, 0)
        estimated_hashes, _ = get_prefix_hashes_incremental(estimated_tokens, BLOCK_SIZE, 0, 0)
        old_hashes = [str(value) for value in old_hashes]
        new_hashes = [str(value) for value in new_hashes]
        estimated_hashes = [str(value) for value in estimated_hashes]
        seed(store, "other", list(range(BLOCK_SIZE)))
        try:
            collector.start()
            identity, start = await publisher.requested_start()
            assert start == 0
            await publisher.batch(identity, 100, kv_payload(make_stored_event("old-remote", old_tokens)))
            await until(lambda: "old-remote" in remote_hashes(store, "a"))
            assert hit_rate(store, "a", old_hashes) == 0
            await publisher.end(identity)
            await until(lambda: hit_rate(store, "a", old_hashes) == 1)
            pin(store, "request", "a", old_hashes)

            publisher.close()
            publisher = None
            await until(lambda: hit_rate(store, "a", old_hashes) == 0 and not remote_hashes(store, "a"))
            await until(lambda: predict(store, "a", estimated_hashes) == 1)
            assert hit_rate(store, "a", estimated_hashes) == 1.0

            # Rebind exactly the same endpoint addresses, as a restarted vLLM does.
            async def rebind():
                while True:
                    try:
                        return Publisher(addresses)
                    except zmq.ZMQError as exc:
                        if exc.errno != zmq.EADDRINUSE:
                            raise
                        await asyncio.sleep(0.01)

            publisher = await asyncio.wait_for(rebind(), 3)
            identity, start = await publisher.requested_start()
            assert start == 0  # Never resume this replacement publisher at the old 101.
            assert not has_blocks(store, "a")
            await publisher.batch(identity, new_sequence, kv_payload(make_stored_event("new-remote", new_tokens)))
            await until(lambda: "new-remote" in remote_hashes(store, "a"))
            assert hit_rate(store, "a", new_hashes) == 0
            assert predict(store, "a", old_hashes) == 0
            await publisher.end(identity)
            await until(lambda: hit_rate(store, "a", new_hashes) == 1)

            assert hit_rate(store, "a", old_hashes) == 0
            assert hit_rate(store, "a", estimated_hashes) == 0
            assert hit_rate(store, "other", old_hashes) == 1
            assert store.inflight_block_count("a") == 1
            assert store.get_marker("request")["replica"] == "a"
            assert remote_hashes(store, "a") == {"new-remote"}
        finally:
            collector.stop(timeout=2)
            if publisher is not None:
                publisher.close()

    asyncio.run(run())


def test_unreachable_publisher_recovers_after_retry_budget_without_restarting_collector(layout, install_metadata):
    async def run():
        publisher = Publisher()
        addresses = publisher.addresses
        publisher.close()
        publisher = None
        collector = get_collector(
            "vllm_zmq",
            CollectorConfig(),
            kv_event_endpoints={"a": [*addresses, "zmq", "topic"]},
        )
        transport = collector._transport
        transport._base_retry_delay = 0.01
        transport._max_retry_delay = 0.03
        transport._max_retry_attempts = 1
        transport._CONNECTION_TIMEOUT = 0.03
        connect = transport._connect_zmq_for
        attempts = []

        async def counted_connect(replica_id):
            attempts.append(replica_id)
            return await connect(replica_id)

        transport._connect_zmq_for = counted_connect
        store = create_store(layout, BLOCK_SIZE, install_metadata)
        hashes, _ = get_prefix_hashes_incremental(list(range(BLOCK_SIZE)), BLOCK_SIZE, 0, 0)
        hashes = [str(value) for value in hashes]

        seed(store, "other", list(range(BLOCK_SIZE)))
        seed(store, "a", list(range(BLOCK_SIZE)))
        pin(store, "request", "a", hashes)
        try:
            collector.start()
            # More attempts than the initial connection and its one retry.
            await until(lambda: len(attempts) >= 3)
            assert not has_blocks(store, "a")
            assert predict(store, "a", hashes) == 1
            assert hit_rate(store, "a", hashes) == 1.0
            assert hit_rate(store, "other", hashes) == 1.0
            assert store.inflight_block_count("a") == 1
            assert store.get_marker("request")["replica"] == "a"
            await until(lambda: len(attempts) >= 5)
            assert hit_rate(store, "a", hashes) == 1.0

            publisher = Publisher(addresses)
            identity, start = await publisher.requested_start()
            assert start == 0
            assert not has_blocks(store, "a")
            assert hit_rate(store, "a", hashes) == 0.0
            assert predict(store, "a", hashes) == 0
            await publisher.batch(identity, 0, kv_payload(make_stored_event("new", list(range(BLOCK_SIZE)))))
            await until(lambda: "new" in remote_hashes(store, "a"))
            assert hit_rate(store, "a", hashes) == 0.0
            await publisher.end(identity)
            await until(lambda: hit_rate(store, "a", hashes) == 1.0)
            assert store.inflight_block_count("a") == 1
            assert store.get_marker("request")["replica"] == "a"
        finally:
            collector.stop(timeout=2)
            if publisher is not None:
                publisher.close()

    asyncio.run(run())


def test_failed_partial_replay_returns_to_fresh_dispatch_estimates(layout, install_metadata):
    async def run():
        publisher = Publisher()
        collector = get_collector(
            "vllm_zmq",
            CollectorConfig(),
            kv_event_endpoints={"a": [*publisher.addresses, "zmq", "topic"]},
        )
        collector._transport._base_retry_delay = 0.2
        collector._transport._max_retry_delay = 0.2
        store = create_store(layout, BLOCK_SIZE, install_metadata)
        partial_tokens = list(range(BLOCK_SIZE))
        estimated_tokens = list(range(BLOCK_SIZE, 2 * BLOCK_SIZE))
        partial, _ = get_prefix_hashes_incremental(partial_tokens, BLOCK_SIZE, 0, 0)
        estimated, _ = get_prefix_hashes_incremental(estimated_tokens, BLOCK_SIZE, 0, 0)
        partial = [str(value) for value in partial]
        estimated = [str(value) for value in estimated]
        pin(store, "request", "a", ["held"])
        try:
            collector.start()
            identity, _ = await publisher.requested_start()
            await publisher.batch(identity, 0, kv_payload(make_stored_event("partial", partial_tokens)))
            await until(lambda: "partial" in remote_hashes(store, "a"))
            assert hit_rate(store, "a", partial) == 0.0
            assert predict(store, "a", estimated) == 0

            await publisher.batch(identity, 2, kv_payload(make_stored_event("gap", estimated_tokens)))
            await until(lambda: not has_blocks(store, "a") and not remote_hashes(store, "a"))
            publisher.close()
            publisher = None
            await until(lambda: predict(store, "a", estimated) == 1)
            assert hit_rate(store, "a", estimated) == 1.0
            assert hit_rate(store, "a", partial) == 0.0
            assert store.inflight_block_count("a") == 1
            assert store.get_marker("request")["replica"] == "a"
        finally:
            collector.stop(timeout=2)
            if publisher is not None:
                publisher.close()

    asyncio.run(run())


def test_consumer_failure_discards_partial_cache_and_keeps_dispatch_gated(layout, install_metadata):
    async def run():
        publisher = Publisher()
        collector = get_collector(
            "vllm_zmq",
            CollectorConfig(),
            kv_event_endpoints={"a": [*publisher.addresses, "zmq", "topic"]},
        )
        write = collector._write_kv_update

        def write_then_fail(update):
            write(update)
            raise ValueError("consumer failed after writing")

        collector._write_kv_update = write_then_fail
        store = create_store(layout, BLOCK_SIZE, install_metadata)
        tokens = list(range(BLOCK_SIZE))
        hashes, _ = get_prefix_hashes_incremental(tokens, BLOCK_SIZE, 0, 0)
        hashes = [str(value) for value in hashes]
        pin(store, "request", "a", ["held"])
        try:
            collector.start()
            identity, _ = await publisher.requested_start()
            await publisher.batch(identity, 0, kv_payload(make_stored_event("partial", tokens)))
            await until(lambda: collector._transport.get_stream_status().get("a", {}).get("state") == "fatal")
            assert not collector._loop_runner.subscription_future.done()
            assert collector.get_status() == {
                "transport": "ZMQTransport",
                "subscription": {"active": True, "error": None},
                "streams": {"a": {"state": "fatal", "reason": "ValueError: consumer failed after writing"}},
            }
            assert not has_blocks(store, "a")
            assert not remote_hashes(store, "a")
            assert predict(store, "a", hashes) == 0
            assert hit_rate(store, "a", hashes) == 0.0
            assert store.inflight_block_count("a") == 1
            assert store.get_marker("request")["replica"] == "a"
        finally:
            collector.stop(timeout=2)
            publisher.close()

    asyncio.run(run())


def test_live_clear_does_not_enter_replay_gate(layout, install_metadata):
    async def run():
        publisher = Publisher()
        collector = get_collector("vllm_zmq", CollectorConfig(), kv_event_endpoints=publisher.endpoints)
        collector._transport._base_retry_delay = 0.01
        collector._transport._max_retry_delay = 0.02
        store = create_store(layout, BLOCK_SIZE, install_metadata)
        tokens = list(range(BLOCK_SIZE))
        hashes, _ = get_prefix_hashes_incremental(tokens, BLOCK_SIZE, 0, 0)
        hashes = [str(value) for value in hashes]
        clear = kv_payload(["AllBlocksCleared"])
        stored = kv_payload(make_stored_event("resident", tokens))
        seed(store, "other", list(range(BLOCK_SIZE)))
        try:
            collector.start()
            identity, start = await publisher.requested_start()
            assert start == 0
            await asyncio.wait_for(publisher.pub.recv(), 3)
            await publisher.batch(identity, 10, stored)
            await publisher.end(identity)
            await until(lambda: hit_rate(store, "a", hashes) == 1.0)
            await publisher.publish(11, clear)
            await until(lambda: collector._kv_event_counts["clear"] == 1)
            assert not has_blocks(store, "a")
            await publisher.publish(12, stored)
            await until(lambda: "resident" in remote_hashes(store, "a"))
            assert hit_rate(store, "a", hashes) == 1.0
            assert predict(store, "a", ["after-live-clear"]) == 1
            assert hit_rate(store, "other", hashes) == 1.0
            assert store.hash_schema()[1] == BLOCK_SIZE
        finally:
            collector.stop(timeout=2)
            publisher.close()

    asyncio.run(run())
