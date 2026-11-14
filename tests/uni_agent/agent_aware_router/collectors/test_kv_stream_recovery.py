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

import pytest
import zmq
from conftest import BLOCK_SIZE, kv_payload, make_stored_event

from tests.uni_agent.agent_aware_router.collectors.zmq_helpers import Publisher
from uni_agent.agent_aware_router.collectors.collector import get_collector
from uni_agent.agent_aware_router.config.collector import CollectorConfig
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), 5)


@pytest.mark.parametrize("new_sequence", [0, 200], ids=["numbering-reset", "already-past-old-cursor"])
def test_same_address_publisher_restart_rebuilds_only_its_cache(new_sequence):
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
        store = DataStore()
        old_tokens = list(range(BLOCK_SIZE))
        new_tokens = list(range(BLOCK_SIZE, 2 * BLOCK_SIZE))
        estimated_tokens = list(range(2 * BLOCK_SIZE, 3 * BLOCK_SIZE))
        old_hashes, _ = get_prefix_hashes_incremental(old_tokens, BLOCK_SIZE, 0, 0)
        new_hashes, _ = get_prefix_hashes_incremental(new_tokens, BLOCK_SIZE, 0, 0)
        estimated_hashes, _ = get_prefix_hashes_incremental(estimated_tokens, BLOCK_SIZE, 0, 0)
        old_hashes = [str(value) for value in old_hashes]
        new_hashes = [str(value) for value in new_hashes]
        estimated_hashes = [str(value) for value in estimated_hashes]
        store.add_kv_blocks("other", old_hashes)
        try:
            collector.start()
            identity, start = await publisher.requested_start()
            assert start == 0
            await publisher.batch(identity, 100, kv_payload(make_stored_event("old-remote", old_tokens)))
            await until(lambda: "old-remote" in collector._parser.remote_to_local_block_hash.get("a", {}))
            assert store.get_layer_prefix_hit_rate("a", old_hashes) == 0
            await publisher.end(identity)
            await until(lambda: store.get_layer_prefix_hit_rate("a", old_hashes) == 1)
            store.incr_block_refs("a", old_hashes)
            store.put_marker("request", "a", old_hashes)

            publisher.close()
            publisher = None
            await until(
                lambda: store.get_layer_prefix_hit_rate("a", old_hashes) == 0
                and "a" not in collector._parser.remote_to_local_block_hash
            )
            await until(lambda: store.record_dispatch_blocks("a", estimated_hashes) == 1)
            assert store.get_layer_prefix_hit_rate("a", estimated_hashes) == 1.0

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
            assert not store.kv_replica_has_blocks("a")
            await publisher.batch(identity, new_sequence, kv_payload(make_stored_event("new-remote", new_tokens)))
            await until(lambda: "new-remote" in collector._parser.remote_to_local_block_hash.get("a", {}))
            assert store.get_layer_prefix_hit_rate("a", new_hashes) == 0
            assert store.record_dispatch_blocks("a", old_hashes) == 0
            await publisher.end(identity)
            await until(lambda: store.get_layer_prefix_hit_rate("a", new_hashes) == 1)

            assert store.get_layer_prefix_hit_rate("a", old_hashes) == 0
            assert store.get_layer_prefix_hit_rate("a", estimated_hashes) == 0
            assert store.get_layer_prefix_hit_rate("other", old_hashes) == 1
            assert store.inflight_block_count("a") == 1
            assert store.get_marker("request")["replica"] == "a"
            assert set(collector._parser.remote_to_local_block_hash["a"]) == {"new-remote"}
        finally:
            collector.stop(timeout=2)
            if publisher is not None:
                publisher.close()

    asyncio.run(run())


def test_unreachable_publisher_recovers_after_retry_budget_without_restarting_collector():
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
        store = DataStore()
        hashes, _ = get_prefix_hashes_incremental(list(range(BLOCK_SIZE)), BLOCK_SIZE, 0, 0)
        hashes = [str(value) for value in hashes]
        store.set_block_size(BLOCK_SIZE)
        store.add_kv_blocks("other", hashes)
        store.add_kv_blocks("a", ["stale-before-first-connect"])
        store.incr_block_refs("a", hashes)
        store.put_marker("request", "a", hashes)
        try:
            collector.start()
            # More attempts than the initial connection and its one retry.
            await until(lambda: len(attempts) >= 3)
            assert not store.kv_replica_has_blocks("a")
            assert store.record_dispatch_blocks("a", hashes) == 1
            assert store.get_layer_prefix_hit_rate("a", hashes) == 1.0
            assert store.get_layer_prefix_hit_rate("other", hashes) == 1.0
            assert store.inflight_block_count("a") == 1
            assert store.get_marker("request")["replica"] == "a"
            await until(lambda: len(attempts) >= 5)
            assert store.get_layer_prefix_hit_rate("a", hashes) == 1.0

            publisher = Publisher(addresses)
            identity, start = await publisher.requested_start()
            assert start == 0
            assert not store.kv_replica_has_blocks("a")
            assert store.get_layer_prefix_hit_rate("a", hashes) == 0.0
            assert store.record_dispatch_blocks("a", hashes) == 0
            await publisher.batch(identity, 0, kv_payload(make_stored_event("new", list(range(BLOCK_SIZE)))))
            await until(lambda: "new" in collector._parser.remote_to_local_block_hash.get("a", {}))
            assert store.get_layer_prefix_hit_rate("a", hashes) == 0.0
            await publisher.end(identity)
            await until(lambda: store.get_layer_prefix_hit_rate("a", hashes) == 1.0)
            assert store.inflight_block_count("a") == 1
            assert store.get_marker("request")["replica"] == "a"
        finally:
            collector.stop(timeout=2)
            if publisher is not None:
                publisher.close()

    asyncio.run(run())


def test_failed_partial_replay_returns_to_fresh_dispatch_estimates():
    async def run():
        publisher = Publisher()
        collector = get_collector(
            "vllm_zmq",
            CollectorConfig(),
            kv_event_endpoints={"a": [*publisher.addresses, "zmq", "topic"]},
        )
        collector._transport._base_retry_delay = 0.2
        collector._transport._max_retry_delay = 0.2
        store = DataStore()
        partial_tokens = list(range(BLOCK_SIZE))
        estimated_tokens = list(range(BLOCK_SIZE, 2 * BLOCK_SIZE))
        partial, _ = get_prefix_hashes_incremental(partial_tokens, BLOCK_SIZE, 0, 0)
        estimated, _ = get_prefix_hashes_incremental(estimated_tokens, BLOCK_SIZE, 0, 0)
        partial = [str(value) for value in partial]
        estimated = [str(value) for value in estimated]
        store.incr_block_refs("a", ["held"])
        store.put_marker("request", "a", ["held"])
        try:
            collector.start()
            identity, _ = await publisher.requested_start()
            await publisher.batch(identity, 0, kv_payload(make_stored_event("partial", partial_tokens)))
            await until(lambda: "partial" in collector._parser.remote_to_local_block_hash.get("a", {}))
            assert store.get_layer_prefix_hit_rate("a", partial) == 0.0
            assert store.record_dispatch_blocks("a", estimated) == 0

            await publisher.batch(identity, 2, kv_payload(make_stored_event("gap", estimated_tokens)))
            await until(
                lambda: not store.kv_replica_has_blocks("a") and "a" not in collector._parser.remote_to_local_block_hash
            )
            publisher.close()
            publisher = None
            await until(lambda: store.record_dispatch_blocks("a", estimated) == 1)
            assert store.get_layer_prefix_hit_rate("a", estimated) == 1.0
            assert store.get_layer_prefix_hit_rate("a", partial) == 0.0
            assert store.inflight_block_count("a") == 1
            assert store.get_marker("request")["replica"] == "a"
        finally:
            collector.stop(timeout=2)
            if publisher is not None:
                publisher.close()

    asyncio.run(run())


def test_consumer_failure_discards_partial_cache_and_keeps_dispatch_gated():
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
        store = DataStore()
        tokens = list(range(BLOCK_SIZE))
        hashes, _ = get_prefix_hashes_incremental(tokens, BLOCK_SIZE, 0, 0)
        hashes = [str(value) for value in hashes]
        store.incr_block_refs("a", ["held"])
        store.put_marker("request", "a", ["held"])
        try:
            collector.start()
            identity, _ = await publisher.requested_start()
            await publisher.batch(identity, 0, kv_payload(make_stored_event("partial", tokens)))
            await until(lambda: collector._loop_runner.subscription_future.done())
            assert not store.kv_replica_has_blocks("a")
            assert "a" not in collector._parser.remote_to_local_block_hash
            assert store.record_dispatch_blocks("a", hashes) == 0
            assert store.get_layer_prefix_hit_rate("a", hashes) == 0.0
            assert store.inflight_block_count("a") == 1
            assert store.get_marker("request")["replica"] == "a"
        finally:
            collector.stop(timeout=2)
            publisher.close()

    asyncio.run(run())


def test_replayed_clear_preserves_requests_created_after_the_original_clear():
    async def run():
        publisher = Publisher()
        collector = get_collector("vllm_zmq", CollectorConfig(), kv_event_endpoints=publisher.endpoints)
        collector._transport._base_retry_delay = 0.01
        collector._transport._max_retry_delay = 0.02
        store = DataStore()
        tokens = list(range(BLOCK_SIZE))
        hashes, _ = get_prefix_hashes_incremental(tokens, BLOCK_SIZE, 0, 0)
        hashes = [str(value) for value in hashes]
        clear = kv_payload(["AllBlocksCleared"])
        stored = kv_payload(make_stored_event("resident", tokens))
        marker = {"replica": "a", "hashes": hashes}
        store.add_kv_blocks("other", hashes)
        try:
            collector.start()
            identity, start = await publisher.requested_start()
            assert start == 0
            await publisher.batch(identity, 10, clear)
            await publisher.batch(identity, 11, stored)
            await publisher.end(identity)
            await until(lambda: store.get_layer_prefix_hit_rate("a", hashes) == 1.0)
            assert collector._kv_event_counts["clear"] == 1
            store.incr_block_refs("a", hashes)
            store.put_marker("request-r", "a", hashes)

            # Interrupt only SUB; replay retains the original history and ROUTER.
            publisher.close_socket("pub")
            await until(lambda: store.record_dispatch_blocks("a", ["fallback"]) == 1)
            assert store.inflight_block_count("a") == 1
            assert store.get_marker("request-r") == marker
            publisher.reopen("pub")
            identity, start = await publisher.requested_start()
            assert start == 0
            await asyncio.wait_for(publisher.pub.recv(), 3)

            await publisher.batch(identity, 10, clear)
            await until(lambda: collector._kv_event_counts["clear"] == 2)
            assert store.inflight_block_count("a") == 1
            assert store.get_marker("request-r") == marker
            assert store.get_layer_prefix_hit_rate("other", hashes) == 1.0
            await publisher.batch(identity, 11, stored)
            await until(lambda: "resident" in collector._parser.remote_to_local_block_hash.get("a", {}))
            assert store.get_layer_prefix_hit_rate("a", hashes) == 0.0
            assert store.record_dispatch_blocks("a", ["during-replay"]) == 0
            await publisher.end(identity)
            await until(lambda: store.get_layer_prefix_hit_rate("a", hashes) == 1.0)
            assert store.record_dispatch_blocks("a", ["after-replay"]) == 1

            # A live clear discards residency without starting another replay gate.
            await publisher.publish(12, clear)
            await until(lambda: collector._kv_event_counts["clear"] == 3)
            assert not store.kv_replica_has_blocks("a")
            assert store.inflight_block_count("a") == 1
            assert store.get_marker("request-r") == marker
            await publisher.publish(13, stored)
            await until(lambda: "resident" in collector._parser.remote_to_local_block_hash.get("a", {}))
            assert store.get_layer_prefix_hit_rate("a", hashes) == 1.0
            assert store.record_dispatch_blocks("a", ["after-live-clear"]) == 1
            assert store.get_layer_prefix_hit_rate("other", hashes) == 1.0
            assert store.get_block_size() == BLOCK_SIZE
            assert store.pop_marker("request-r") == marker
            store.decr_block_refs("a", hashes)
            assert store.inflight_block_count("a") == 0
        finally:
            collector.stop(timeout=2)
            publisher.close()

    asyncio.run(run())
