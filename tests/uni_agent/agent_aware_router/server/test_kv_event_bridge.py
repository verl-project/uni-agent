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

"""Cache epoch isolation and replay storage limits."""

import asyncio
import socket

import msgpack
import pytest
import zmq
import zmq.asyncio

from uni_agent.agent_aware_router.collectors.collector import Collector
from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.collectors.transport.base import TransportMessage
from uni_agent.agent_aware_router.collectors.transport.lifecycle import LifecycleControlReceiver
from uni_agent.agent_aware_router.collectors.transport.zmq import ZMQTransport
from uni_agent.agent_aware_router.server.cache_lifecycle import CacheLifecycle
from uni_agent.agent_aware_router.server.kv_event_bridge import KVEventBridge
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.types.cache_lifecycle import CacheLifecycleState
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental
from uni_agent.vllm_plugins.lifecycle import RESET_BOUNDARY_KEY

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_native_subscription_start_and_rebind_preserve_ready_lifecycle():
    async def run():
        context = zmq.asyncio.Context()
        native_pub = context.socket(zmq.PUB)
        native_replay = context.socket(zmq.ROUTER)
        pub_port = native_pub.bind_to_random_port("tcp://127.0.0.1")
        replay_port = native_replay.bind_to_random_port("tcp://127.0.0.1")
        source = {
            "publisher": "zmq",
            "topic": "native-kv",
            "endpoint": f"tcp://127.0.0.1:{pub_port}",
            "replay_endpoint": f"tcp://127.0.0.1:{replay_port}",
        }
        payload = msgpack.packb([0, [{"type": "BlockStored", "token_ids": [7]}]], use_bin_type=True)

        async def replay_native():
            while True:
                identity, empty, start = await native_replay.recv_multipart()
                assert empty == b"" and int.from_bytes(start, "big") == 0
                await native_replay.send_multipart([identity, b"", b"native-kv", (7).to_bytes(8, "big"), payload])
                await native_replay.send_multipart([identity, b"", b"", (-1).to_bytes(8, "big", signed=True), b""])

        control = CacheLifecycle()
        bridge = KVEventBridge(control, "127.0.0.1")
        replay_task = asyncio.create_task(replay_native())

        async def wait_for_forwarded_batch():
            async def receive():
                while not bridge._buffer:
                    await asyncio.sleep(0.01)
                await bridge._queue.join()

            await asyncio.wait_for(receive(), 2)
            assert control.state.index_state == "ready" and control.state.reason is None
            assert len(bridge._buffer) == 1
            forwarded = msgpack.unpackb(bridge._buffer[0][-1], raw=False)
            assert forwarded["payload"] == payload and forwarded["error"] is None
            assert forwarded["state"]["instance_id"] == control.state.instance_id

        try:
            await bridge.start(source)
            await wait_for_forwarded_batch()
            for _ in range(3):
                instance = control.state.instance_id
                old_consumer, old_native = bridge._consumer, bridge._native_task
                await bridge.rebind(source)
                await wait_for_forwarded_batch()
                assert control.state.instance_id != instance
                assert old_consumer.done() and old_native.done()
                assert old_consumer not in bridge._tasks
                assert len(bridge._tasks) == 2
        finally:
            replay_task.cancel()
            await asyncio.gather(replay_task, return_exceptions=True)
            await bridge.close()
            native_pub.close(0)
            native_replay.close(0)
            context.term()

    asyncio.run(run())


@pytest.mark.parametrize("status", ["no_continuity", "unreachable"])
def test_native_transport_failure_invalidates_lifecycle_and_forwarded_batch(status):
    async def run():
        control = CacheLifecycle()
        bridge = KVEventBridge(control, "127.0.0.1")
        bridge._consumer = asyncio.create_task(bridge._consume())
        bridge._tasks.append(bridge._consumer)
        try:
            bridge._enqueue(TransportMessage(b"", status=status), "native")
            await asyncio.wait_for(bridge._queue.join(), 2)
            assert control.state.index_state == "unknown"
            assert control.state.reason == f"event_stream_{status}"
            forwarded = msgpack.unpackb(bridge._buffer[0][-1], raw=False)
            assert forwarded["error"] == f"event_stream_{status}"
            assert forwarded["payload"] == b""
        finally:
            await bridge.close()

    asyncio.run(run())


def test_native_sequence_gap_invalidates_router_cache_before_bridge_renumbering(monkeypatch, layout, stored):
    monkeypatch.setattr(KVCacheStore, "_instance", None)

    async def run():
        store = DataStore()
        store.configure_cache_registry(layout((("full_attention", 4),), unit=None))

        async def notify(sink, state):
            store.update_cache_lifecycle("a", CacheLifecycleState.from_dict(state))

        control = CacheLifecycle(notifier=notify)
        control.bind({"token": "router"})
        store.update_cache_lifecycle("a", control.state)
        bridge = KVEventBridge(control, "127.0.0.1")
        bridge._consumer = asyncio.create_task(bridge._consume())
        bridge._tasks.append(bridge._consumer)
        parser = VLLMKVParser()
        event = stored(tokens=list(range(4)), hashes=["h0"])
        hashes = [str(value) for value in get_prefix_hashes_incremental(list(range(5)), 4, 0, 0)[0]]

        async def deliver(sequence, events):
            payload = msgpack.packb([0, events])
            bridge._enqueue(TransportMessage(payload, sequence), "native")
            await asyncio.wait_for(bridge._queue.join(), 2)
            forwarded = bridge._buffer[-1]
            store.apply_kv_update(parser.parse(TransportMessage(forwarded[-1], forwarded[0]), "a"))
            return msgpack.unpackb(forwarded[-1], raw=False)

        def hit():
            return store.estimate_prefixes(["a"], hashes, 5)["a"].estimated_hit_tokens

        try:
            assert (await deliver(40, [event]))["sequence"] == 0
            assert hit() == 4
            # Native 41 could have removed h0; sequential bridge IDs must not
            # conceal the missing native events from the router.
            forwarded = await deliver(43, [event])
            assert forwarded["sequence"] == 1
            assert forwarded["error"] == "native_kv_sequence_gap:41:43"
            assert forwarded["payload"] == b""
            assert control.state.index_state == "unknown"
            assert hit() == 0
            assert store._kv._stream_errors == {"a": "native_kv_sequence_gap:41:43"}
            # A later managed Clear cannot reconstruct how many Clears were lost.
            await control.begin("reset-after-gap", 1)
            await control.finish(True)
            await deliver(44, [{"type": "AllBlocksCleared"}, event])
            assert control.state.index_state == "unknown"
            assert hit() == 0
        finally:
            await bridge.close()

    asyncio.run(run())


def test_duplicate_native_clear_is_counted_once_and_preserves_router_cache(monkeypatch, layout, stored, removed):
    monkeypatch.setattr(KVCacheStore, "_instance", None)

    async def run():
        store = DataStore()
        store.configure_cache_registry(layout((("full_attention", 4),), unit=None))

        async def notify(sink, state):
            store.update_cache_lifecycle("a", CacheLifecycleState.from_dict(state))

        control = CacheLifecycle(notifier=notify)
        control.bind({"token": "router"})
        store.update_cache_lifecycle("a", control.state)
        bridge = KVEventBridge(control, "127.0.0.1")
        bridge._consumer = asyncio.create_task(bridge._consume())
        bridge._tasks.append(bridge._consumer)
        parser = VLLMKVParser()
        event = stored(tokens=list(range(4)), hashes=["h0"])
        hashes = [str(value) for value in get_prefix_hashes_incremental(list(range(5)), 4, 0, 0)[0]]

        async def deliver(sequence, events):
            count = len(bridge._buffer)
            bridge._enqueue(TransportMessage(msgpack.packb([0, events]), sequence), "native")
            await asyncio.wait_for(bridge._queue.join(), 2)
            for forwarded in list(bridge._buffer)[count:]:
                store.apply_kv_update(parser.parse(TransportMessage(forwarded[-1], forwarded[0]), "a"))
            return len(bridge._buffer) - count

        def hit():
            return store.estimate_prefixes(["a"], hashes, 5)["a"].estimated_hit_tokens

        try:
            await control.begin("reset", 1)
            await control.finish(True)
            events = [{"type": "AllBlocksCleared"}, event]
            assert await deliver(40, events) == 1
            assert control.state.observed_clears == 1
            assert control.state.index_state == "ready" and hit() == 4
            assert await deliver(40, events) == 0
            assert await deliver(39, events) == 0
            assert control.state.expected_clears == control.state.observed_clears == 1
            assert control.state.index_state == "ready" and hit() == 4
            assert await deliver(41, [removed(0, ["h0"])]) == 1
            assert bridge._sequence == 1 and hit() == 0
        finally:
            await bridge.close()

    asyncio.run(run())


def test_matching_reset_boundary_recovers_same_engine_despite_native_sequence_gap(monkeypatch, layout, stored, removed):
    monkeypatch.setattr(KVCacheStore, "_instance", None)

    async def run():
        store = DataStore()
        store.configure_cache_registry(layout((("full_attention", 4),), unit=None))

        async def notify(sink, state):
            store.update_cache_lifecycle("a", CacheLifecycleState.from_dict(state))

        control = CacheLifecycle(notifier=notify)
        control.bind({"token": "router"})
        store.update_cache_lifecycle("a", control.state)
        bridge = KVEventBridge(control, "127.0.0.1")
        bridge._consumer = asyncio.create_task(bridge._consume())
        bridge._tasks.append(bridge._consumer)
        parser = VLLMKVParser()
        old = stored(tokens=list(range(4)), hashes=["old"])
        fresh = stored(tokens=list(range(8, 12)), hashes=["fresh"])
        clear = {"type": "AllBlocksCleared"}

        async def deliver(sequence, events, nonce=None):
            batch = [0, events, 0]
            if nonce is not None:
                batch.append({RESET_BOUNDARY_KEY: {"schema_version": 1, "nonce": nonce}})
            bridge._enqueue(TransportMessage(msgpack.packb(batch), sequence), "native")
            await asyncio.wait_for(bridge._queue.join(), 2)
            forwarded = bridge._buffer[-1]
            store.apply_kv_update(parser.parse(TransportMessage(forwarded[-1], forwarded[0]), "a"))
            return msgpack.unpackb(forwarded[-1], raw=False)

        def hit(tokens):
            hashes = [str(value) for value in get_prefix_hashes_incremental(tokens, 4, 0, 0)[0]]
            return store.estimate_prefixes(["a"], hashes, len(tokens))["a"].estimated_hit_tokens

        try:
            await deliver(40, [old])
            assert hit(list(range(5))) == 4
            instance, epoch = control.state.instance_id, control.state.epoch

            async def overflow(sequence):
                limit = bridge.queue_limit
                bridge.queue_limit = 0
                pending = len(bridge._tasks)
                try:
                    dropped = msgpack.packb([0, [removed(0, ["fresh"])]])
                    bridge._enqueue(TransportMessage(dropped, sequence), "native")
                    await asyncio.wait_for(asyncio.gather(*bridge._tasks[pending:]), 2)
                finally:
                    bridge.queue_limit = limit

            await overflow(41)
            assert control.state.index_state == "unknown" and hit(list(range(5))) == 0
            await control.prepare_recovery("checked-reset")
            # Old Clears, including a different attempt's marker, cannot establish
            # the checked reset's ordering boundary or restore old residency.
            await deliver(42, [clear, old], nonce="obsolete-attempt")
            assert hit(list(range(5))) == 0
            assert control.state.index_state != "ready"
            forwarded = await deliver(44, [old, clear, fresh], nonce="checked-reset")
            await control.wait_for_recovery(0.5)
            assert forwarded["error"] is None
            assert msgpack.unpackb(forwarded["payload"], raw=False)[1] == [clear, fresh]
            assert control.state.instance_id == instance and control.state.epoch == epoch + 1
            assert control.state.expected_clears == control.state.observed_clears == 1
            assert control.state.index_state == "ready"
            assert hit(list(range(5))) == 0 and hit(list(range(8, 13))) == 4
            await deliver(45, [removed(0, ["fresh"])])
            assert hit(list(range(8, 13))) == 0
            await deliver(46, [fresh])
            assert hit(list(range(8, 13))) == 4
            # Recovery must re-arm overflow invalidation on the same bridge;
            # no later native frame is needed to discover this lost Removed.
            await overflow(47)
            assert control.state.index_state == "unknown"
            assert hit(list(range(8, 13))) == 0
        finally:
            await bridge.close()

    asyncio.run(run())


@pytest.mark.parametrize("limit", ["bytes", "messages"])
@pytest.mark.parametrize("lost_event", ["removed", "clear"])
def test_native_queue_overflow_invalidates_cache_when_removed_or_clear_is_dropped(
    monkeypatch, layout, stored, removed, limit, lost_event
):
    monkeypatch.setattr(KVCacheStore, "_instance", None)

    async def run():
        store = DataStore()
        store.configure_cache_registry(layout((("full_attention", 4),), unit=None))
        notifications = []

        async def notify(sink, state):
            notifications.append(state)
            store.update_cache_lifecycle("a", CacheLifecycleState.from_dict(state))

        control = CacheLifecycle(notifier=notify)
        control.bind({"token": "router"})
        store.update_cache_lifecycle("a", control.state)
        store.update_cache_lifecycle("b", control.state)
        queued = msgpack.packb([0, []])
        bridge = KVEventBridge(control, "127.0.0.1", queue_bytes=len(queued) if limit == "bytes" else 1024**2)
        parser = VLLMKVParser()
        initial = msgpack.packb([0, [stored(tokens=list(range(4)), hashes=["h0"])]])
        try:
            await bridge.forward(initial)
            envelope = TransportMessage(bridge._buffer[-1][-1], 0)
            for node in ("a", "b"):
                assert store.apply_kv_update(parser.parse(envelope, node))
            assert store.per_replica_block_counts() == {"a": 1, "b": 1}
            # Fill the real queue before its consumer runs, then lose an event
            # which would invalidate the cached checkpoint at the native source.
            for _ in range(bridge._queue.maxsize if limit == "messages" else 1):
                bridge._enqueue(TransportMessage(queued, 40), "native")
            event = removed(0, ["h0"]) if lost_event == "removed" else {"type": "AllBlocksCleared"}
            dropped = TransportMessage(msgpack.packb([1, [event]]), 41)
            bridge._enqueue(dropped, "native")
            bridge._enqueue(dropped, "native")
            assert bridge._overflow_reported
            await asyncio.wait_for(asyncio.gather(*bridge._tasks), 2)
            assert control.state.index_state == "unknown"
            assert [state["reason"] for state in notifications] == ["bridge_queue_overflow"]
            assert store.per_replica_block_counts() == {"b": 1}
            bridge._consumer = asyncio.create_task(bridge._consume())
            bridge._tasks.append(bridge._consumer)
            await asyncio.wait_for(bridge._queue.join(), 2)
            assert control.state.index_state == "unknown"
            assert store.per_replica_block_counts() == {"b": 1}
        finally:
            await bridge.close()

    asyncio.run(run())


def test_repeated_overflow_and_checked_recovery_retain_only_live_bridge_tasks(monkeypatch, layout, stored, removed):
    monkeypatch.setattr(KVCacheStore, "_instance", None)

    async def run():
        store = DataStore()
        store.configure_cache_registry(layout((("full_attention", 4),), unit=None))

        async def notify(sink, state):
            store.update_cache_lifecycle("a", CacheLifecycleState.from_dict(state))

        control = CacheLifecycle(notifier=notify)
        control.bind({"token": "router"})
        store.update_cache_lifecycle("a", control.state)
        bridge = KVEventBridge(control, "127.0.0.1")
        bridge._consumer = asyncio.create_task(bridge._consume())
        resident = {bridge._consumer, asyncio.create_task(bridge._serve_replay())}
        bridge._tasks.extend(resident)
        parser = VLLMKVParser()
        event = stored(tokens=list(range(4)), hashes=["h0"])
        hashes = [str(value) for value in get_prefix_hashes_incremental(list(range(5)), 4, 0, 0)[0]]

        async def deliver(sequence, batch):
            bridge._enqueue(TransportMessage(msgpack.packb(batch), sequence), "native")
            await asyncio.wait_for(bridge._queue.join(), 2)
            forwarded = bridge._buffer[-1]
            store.apply_kv_update(parser.parse(TransportMessage(forwarded[-1], forwarded[0]), "a"))

        def hit():
            return store.estimate_prefixes(["a"], hashes, 5)["a"].estimated_hit_tokens

        try:
            await deliver(0, [0, [event], 0])
            assert hit() == 4
            for attempt in range(10):
                bridge.queue_limit = 0
                dropped = msgpack.packb([0, [removed(0, ["h0"])]])
                bridge._enqueue(TransportMessage(dropped, 2 * attempt + 1), "native")
                invalidations = set(bridge._tasks) - resident
                assert len(invalidations) == 1
                await asyncio.wait_for(asyncio.gather(*invalidations), 2)
                assert control.state.index_state == "unknown" and hit() == 0
                bridge.queue_limit = 1024**2
                nonce = f"checked-reset-{attempt}"
                await control.prepare_recovery(nonce)
                await deliver(
                    2 * attempt + 2,
                    [
                        0,
                        [{"type": "AllBlocksCleared"}, event],
                        0,
                        {RESET_BOUNDARY_KEY: {"schema_version": 1, "nonce": nonce}},
                    ],
                )
                await control.wait_for_recovery(0.5)
                assert control.state.index_state == "ready" and hit() == 4
                # Finished invalidations must not accumulate across recoveries;
                # both permanent workers continue serving the same bridge.
                assert set(bridge._tasks) == resident
                assert all(not task.done() for task in resident)
        finally:
            await bridge.close()
        assert not bridge._tasks and all(task.done() for task in resident)

    asyncio.run(run())


def test_replay_never_returns_old_epoch_and_replay_storage_is_bounded():
    async def run():
        control = CacheLifecycle()
        bridge = KVEventBridge(control, "127.0.0.1", replay_steps=3, replay_bytes=4096)
        replay_task = asyncio.create_task(bridge._serve_replay())
        ctx = zmq.asyncio.Context()
        client = ctx.socket(zmq.DEALER)
        client.connect(bridge.sources()["0"]["replay_endpoint"])
        try:
            old = msgpack.packb([0, [{"type": "BlockStored", "token_ids": [7]}]])
            await bridge.forward(old)
            await control.begin("reset", 1)
            await control.finish(True)
            await bridge.forward(None)
            await client.send_multipart([b"", (0).to_bytes(8, "big")])
            frames = await asyncio.wait_for(client.recv_multipart(), 2)
            message = msgpack.unpackb(frames[-1], raw=False)
            assert message["state"]["epoch"] == 1 and message["payload"] == b""
            end = await asyncio.wait_for(client.recv_multipart(), 2)
            assert not end[-1]
            for _ in range(8):
                await bridge.forward(b"x")
            assert len(bridge._buffer) == 3
            for _ in range(2):
                await bridge.forward(b"x" * 2048)
            assert len(bridge._buffer) == 1
            assert bridge._buffer_bytes <= 4096
        finally:
            replay_task.cancel()
            await asyncio.gather(replay_task, return_exceptions=True)
            await bridge.close()
            client.close(0)
            ctx.term()

    asyncio.run(run())


@pytest.mark.parametrize("history", ["truncated", "caught-up", "new-epoch", "new-instance"])
def test_empty_replay_end_only_marks_eviction_in_current_generation(history):
    """Replay END status describes the requested window, independently of cache policy."""

    async def run():
        control = CacheLifecycle()
        # Even the newest envelope exceeds this budget, leaving only END.
        bridge = KVEventBridge(control, "127.0.0.1", replay_bytes=1)
        replay_task = asyncio.create_task(bridge._serve_replay())
        source = bridge.sources()["0"]
        transport = ZMQTransport(
            {
                "a": [
                    source["endpoint"].removeprefix("tcp://"),
                    source["replay_endpoint"].removeprefix("tcp://"),
                    "zmq",
                    source["topic"],
                ]
            }
        )
        received = []
        transport._deliver("a", 0, b"initial", lambda message, node: None)
        try:
            await bridge.forward(msgpack.packb([0, []]))
            if history != "caught-up":
                await bridge.forward(msgpack.packb([1, []]))
            if history == "new-epoch":
                await control.begin("reset", 1)
            elif history == "new-instance":
                await control.new_instance()
            assert not bridge._buffer
            await transport._connect_zmq_for("a", source["endpoint"], source["replay_endpoint"])
            assert await asyncio.wait_for(
                transport._replay_historical_data_for("a", lambda message, node: received.append(message)), 2
            )
            expected = [TransportMessage(b"", status="no_continuity")] if history == "truncated" else []
            assert received == expected
            # END status is control metadata, never a native event sequence.
            assert transport.get_cursors()["a"][-1] == 0
        finally:
            transport.stop()
            replay_task.cancel()
            await asyncio.gather(replay_task, return_exceptions=True)
            await bridge.close()

    asyncio.run(run())


def test_rebuilt_collector_stays_cold_until_current_clear_then_recovers(monkeypatch, layout, stored):
    monkeypatch.setattr(KVCacheStore, "_instance", None)

    async def run():
        replay_finished = asyncio.Queue()
        replay_method = ZMQTransport._replay_historical_data_for

        async def replay_and_notify(transport, node, handler):
            complete = await replay_method(transport, node, handler)
            replay_finished.put_nowait(complete)
            return complete

        monkeypatch.setattr(ZMQTransport, "_replay_historical_data_for", replay_and_notify)
        store = DataStore()
        store.configure_cache_registry(layout())
        hashes = [str(value) for value in get_prefix_hashes_incremental(list(range(17)), 2, 0, 0)[0]]
        receiver = LifecycleControlReceiver(
            lambda node, state: store.update_cache_lifecycle(node, CacheLifecycleState.from_dict(state)), "127.0.0.1"
        )
        control = CacheLifecycle()
        store.update_cache_lifecycle("a", control.state)
        control.bind(receiver.descriptor("a"))
        bridge = KVEventBridge(control, "127.0.0.1")
        replay = asyncio.create_task(bridge._serve_replay())
        source = bridge.sources()["0"]

        def new_collector():
            transport = ZMQTransport(
                {
                    "a": [
                        source["endpoint"].removeprefix("tcp://"),
                        source["replay_endpoint"].removeprefix("tcp://"),
                        "zmq",
                        source["topic"],
                    ]
                },
            )
            collector = Collector(transport, VLLMKVParser())
            collector.start()
            return collector

        def hit():
            return store.estimate_prefixes(["a"], hashes, 17)["a"].estimated_hit_tokens

        async def wait_for_hit(expected):
            for _ in range(200):
                if hit() == expected:
                    return
                await asyncio.sleep(0.01)
            raise AssertionError(f"expected hit {expected}, observed {hit()}")

        collector = new_collector()
        try:
            events = [stored(), stored(group=1, kind="mamba", block=8)]
            native = msgpack.packb([0, events], use_bin_type=True)
            await bridge.forward(await control.accept(native))
            await wait_for_hit(16)
            assert await asyncio.wait_for(replay_finished.get(), 2)
            await control.begin("reset", 1)
            await control.finish(True)
            assert hit() == 0
            await asyncio.to_thread(collector.stop)
            collector = new_collector()
            # An empty current-epoch replay must not expose the old buffer.
            assert await asyncio.wait_for(replay_finished.get(), 2)
            assert hit() == 0
            native = msgpack.packb([1, [{"type": "AllBlocksCleared"}, *events]], use_bin_type=True)
            await bridge.forward(await control.accept(native))
            await wait_for_hit(16)
        finally:
            await asyncio.to_thread(collector.stop)
            replay.cancel()
            await asyncio.gather(replay, return_exceptions=True)
            await bridge.close()
            receiver.close()

    asyncio.run(run())


@pytest.mark.parametrize("host", ["::1", "[::1]"])
def test_ipv6_bridge_sources_can_be_replayed(host):
    try:
        with socket.socket(socket.AF_INET6) as probe:
            probe.bind(("::1", 0))
    except OSError:
        pytest.skip("IPv6 loopback is unavailable")

    async def run():
        control = CacheLifecycle()
        bridge = KVEventBridge(control, host)
        replay = asyncio.create_task(bridge._serve_replay())
        source = bridge.sources()["0"]
        assert source["endpoint"].startswith("tcp://[::1]:")
        assert source["replay_endpoint"].startswith("tcp://[::1]:")
        transport = ZMQTransport({"a": [source["endpoint"][6:], source["replay_endpoint"][6:], "zmq", source["topic"]]})
        payload = msgpack.packb([0, []])
        received = []
        try:
            await bridge.forward(payload)
            assert bridge._pub.getsockopt(zmq.IPV6) == bridge._replay.getsockopt(zmq.IPV6) == 1
            assert await transport._connect_zmq_for("a", source["endpoint"], source["replay_endpoint"])
            assert await asyncio.wait_for(
                transport._replay_historical_data_for("a", lambda message, node: received.append(message)), 2
            )
            assert len(received) == 1
            assert msgpack.unpackb(received[0].payload, raw=False)["payload"] == payload
        finally:
            transport.stop()
            replay.cancel()
            await asyncio.gather(replay, return_exceptions=True)
            await bridge.close()

    asyncio.run(run())
