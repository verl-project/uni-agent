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

"""KV event replay and lifecycle notification protocols."""

import asyncio

import msgpack
import pytest
import zmq
import zmq.asyncio

from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.collectors.transport.lifecycle import LifecycleControlReceiver, notify_lifecycle
from uni_agent.agent_aware_router.collectors.transport.zmq import ZMQTransport
from uni_agent.agent_aware_router.store import DataStore
from uni_agent.agent_aware_router.types.cache_lifecycle import CacheLifecycleState, KVEventFrame

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.mark.parametrize("gap", [False, True], ids=["complete-replay", "missing-event"])
def test_subscription_retry_preserves_cache_only_with_contiguous_replay(layout, stored, removed, gap):
    async def run():
        ctx = zmq.asyncio.Context()
        pub, replay = ctx.socket(zmq.XPUB), ctx.socket(zmq.ROUTER)
        pub.setsockopt(zmq.XPUB_VERBOSE, 1)
        pub_port = pub.bind_to_random_port("tcp://127.0.0.1")
        replay_port = replay.bind_to_random_port("tcp://127.0.0.1")
        transport = ZMQTransport(
            {"a": [f"127.0.0.1:{pub_port}", f"127.0.0.1:{replay_port}", "zmq", "kv"]},
            deliver_frames=True,
            base_retry_delay=0,
        )
        store = DataStore()
        store.configure_cache_registry(layout((("full_attention", 4),), unit=None))
        parser = VLLMKVParser(group_aware=True)
        delivered = asyncio.Queue()

        def handler(frame, node):
            store.apply_kv_update(parser.parse(frame, node))
            delivered.put_nowait(frame)

        async def respond(expected, batches):
            identity, delimiter, start = await replay.recv_multipart()
            assert delimiter == b"" and int.from_bytes(start, "big") == expected
            for sequence, events in batches:
                await replay.send_multipart(
                    [identity, b"", b"kv", sequence.to_bytes(8, "big"), msgpack.packb([0, events])]
                )
            await replay.send_multipart([identity, b"", b"", (-1).to_bytes(8, "big", signed=True), b""])

        async def received(sequence):
            while (await delivered.get()).sequence != sequence:
                pass

        subscription = asyncio.create_task(transport.subscribe(handler))
        try:
            assert await asyncio.wait_for(pub.recv(), 3) == b"\x01kv"
            await asyncio.wait_for(respond(0, [(0, [stored(tokens=list(range(4)), hashes=["h0"])])]), 3)
            await asyncio.wait_for(received(0), 3)
            assert store.per_replica_block_counts() == {"a": 1}
            await pub.send_multipart([b"kv", b"bad-sequence", b"invalid"])
            batches = [] if gap else [(1, [stored(tokens=list(range(4, 8)), hashes=["h1"])])]
            batches.append((2, [removed(0, ["h0"])]))
            await asyncio.wait_for(respond(1, batches), 3)
            await asyncio.wait_for(received(2), 3)
            # Wait for the new SUB socket rather than relying on a fixed sleep.
            while await asyncio.wait_for(pub.recv(), 3) != b"\x01kv":
                pass
            await pub.send_multipart(
                [b"kv", (3).to_bytes(8, "big"), msgpack.packb([0, [stored(tokens=list(range(8, 12)), hashes=["h2"])]])]
            )
            await asyncio.wait_for(received(3), 3)
            if gap:
                assert store._kv._stream_errors == {"a": "kv_sequence_gap:1:2"}
                assert store.per_replica_block_counts() == {}
            else:
                assert not store._kv._stream_errors
                assert store.per_replica_block_counts() == {"a": 2}
        finally:
            transport.stop()
            await subscription
            pub.close(0)
            replay.close(0)
            ctx.term()

    asyncio.run(run())


def test_replay_timeout_reports_failure_and_replaces_socket(monkeypatch):
    async def run():
        transport = ZMQTransport({"a": ["127.0.0.1:1", "127.0.0.1:2", "zmq", "kv"]}, deliver_frames=True)
        await transport._connect_zmq_for("a", "tcp://127.0.0.1:1", "tcp://127.0.0.1:2")
        previous = transport._endpoint_sockets["a"].replay_socket
        wait_for = asyncio.wait_for
        monkeypatch.setattr(asyncio, "wait_for", lambda awaitable, timeout: wait_for(awaitable, 0.01))
        frames = []
        try:
            assert not await transport._replay_historical_data_for("a", lambda frame, node: frames.append(frame))
            assert [frame.error for frame in frames] == ["kv_replay_incomplete"]
            assert previous.closed
            assert transport._endpoint_sockets["a"].replay_socket is not previous
        finally:
            transport.stop()

    asyncio.run(run())


def test_dealer_reads_multiple_batches_and_end_marker_and_resumes_at_cursor():
    async def run():
        ctx = zmq.asyncio.Context()
        pub, replay = ctx.socket(zmq.PUB), ctx.socket(zmq.ROUTER)
        pub_port = pub.bind_to_random_port("tcp://127.0.0.1")
        replay_port = replay.bind_to_random_port("tcp://127.0.0.1")
        endpoints = {"a": [f"127.0.0.1:{pub_port}", f"127.0.0.1:{replay_port}", "zmq", "kv"]}
        transport = ZMQTransport(endpoints, deliver_frames=True)
        received = []
        await transport._connect_zmq_for("a", f"tcp://127.0.0.1:{pub_port}", f"tcp://127.0.0.1:{replay_port}")

        async def serve(expected_start, sequences):
            identity, delimiter, start = await replay.recv_multipart()
            assert delimiter == b"" and int.from_bytes(start, "big") == expected_start
            for sequence in sequences:
                await replay.send_multipart(
                    [identity, b"", b"kv", sequence.to_bytes(8, "big"), msgpack.packb([sequence, []])]
                )
            await replay.send_multipart([identity, b"", b"", (-1).to_bytes(8, "big", signed=True), b""])

        try:
            worker = asyncio.create_task(serve(0, [5, 6, 7]))
            assert await asyncio.wait_for(
                transport._replay_historical_data_for("a", lambda frame, node: received.append(frame)), 2
            )
            await worker
            assert [frame.sequence for frame in received] == [5, 6, 7]
            transport.stop()
            transport = ZMQTransport(endpoints, deliver_frames=True, cursors=transport.get_cursors())
            await transport._connect_zmq_for("a", f"tcp://127.0.0.1:{pub_port}", f"tcp://127.0.0.1:{replay_port}")
            worker = asyncio.create_task(serve(8, [7, 8]))
            assert await asyncio.wait_for(
                transport._replay_historical_data_for("a", lambda frame, node: received.append(frame)), 2
            )
            await worker
            assert [frame.sequence for frame in received] == [5, 6, 7, 8]
            assert not any(frame.error for frame in received)
            transport.stop()
            endpoints["a"][3] = "new-stream"
            transport = ZMQTransport(endpoints, deliver_frames=True, cursors=transport.get_cursors())
            await transport._connect_zmq_for("a", f"tcp://127.0.0.1:{pub_port}", f"tcp://127.0.0.1:{replay_port}")
            worker = asyncio.create_task(serve(0, [12]))
            assert await asyncio.wait_for(
                transport._replay_historical_data_for("a", lambda frame, node: received.append(frame)), 2
            )
            await worker
            assert received[-1].sequence == 12 and not received[-1].error
        finally:
            transport.stop()
            pub.close(0)
            replay.close(0)
            ctx.term()

    asyncio.run(run())


@pytest.mark.parametrize("replay", [False, True], ids=["sub-only", "with-replay"])
def test_missing_sequence_and_duplicate_are_not_silently_applied(replay):
    transport = ZMQTransport({"a": ["127.0.0.1:1", "127.0.0.1:2" if replay else "", "zmq", "kv"]}, deliver_frames=True)
    frames = []
    handler = lambda frame, node: frames.append(frame)
    transport._deliver("a", 40, b"first", handler)
    transport._deliver("a", 40, b"duplicate", handler)
    transport._deliver("a", 41, b"next", handler)
    transport._deliver("a", 43, b"gap", handler)
    assert [frame.payload for frame in frames] == [b"first", b"next", b"", b"gap"]
    assert frames[2].error == "kv_sequence_gap:42:43"


@pytest.mark.parametrize("group_aware", [False, True], ids=["legacy", "groups"])
def test_fresh_sub_only_source_resets_old_residency_and_accepts_nonzero_sequence(group_aware, layout, stored, removed):
    async def run():
        store = DataStore()
        if group_aware:
            store.configure_cache_registry(layout((("full_attention", 4),), unit=None))
        parser = VLLMKVParser(group_aware=group_aware)
        initial = msgpack.packb([0, [stored(tokens=list(range(4)), hashes=["old"])]])
        store.apply_kv_update(parser.parse(KVEventFrame(initial, 100), "a"))
        other = msgpack.packb([0, [stored(tokens=list(range(100, 104)), hashes=["other"])]])
        store.apply_kv_update(parser.parse(KVEventFrame(other, 110), "b"))
        assert store.per_replica_block_counts() == {"a": 1, "b": 1}
        ctx = zmq.asyncio.Context()
        pub = ctx.socket(zmq.XPUB)
        port = pub.bind_to_random_port("tcp://127.0.0.1")
        transport = ZMQTransport({"a": [f"127.0.0.1:{port}", "", "zmq", "kv"]}, deliver_frames=True)
        delivered = asyncio.Queue()

        def handler(frame, node):
            store.apply_kv_update(parser.parse(frame, node))
            delivered.put_nowait(frame)

        task = asyncio.create_task(transport.subscribe(handler))
        try:
            assert await asyncio.wait_for(pub.recv(), 3) == b"\x01kv"
            assert (await asyncio.wait_for(delivered.get(), 3)).reset_stream
            assert store.per_replica_block_counts() == {"b": 1}
            assert "a" not in store._kv._event_sequences
            for sequence, events, count in [
                (5, [stored(tokens=list(range(4, 8)), hashes=["new"])], 1),
                (6, [removed(0, ["new"])], 0),
            ]:
                await pub.send_multipart([b"kv", sequence.to_bytes(8, "big"), msgpack.packb([0, events])])
                assert (await asyncio.wait_for(delivered.get(), 3)).sequence == sequence
                assert store.per_replica_block_counts().get("a", 0) == count
                assert store.per_replica_block_counts()["b"] == 1
            assert not store._kv._stream_errors
        finally:
            transport.stop()
            await task
            pub.close(0)
            ctx.term()

    asyncio.run(run())


def test_control_ack_and_capability_check():
    applied = []
    receiver = LifecycleControlReceiver(
        lambda sid, state: applied.append((sid, state)) or {"applied": True}, "127.0.0.1"
    )

    async def run():
        sink = receiver.descriptor("a")
        assert (await notify_lifecycle(sink, CacheLifecycleState("engine").to_dict()))["applied"]
        with pytest.raises(RuntimeError, match="capability"):
            await notify_lifecycle({**sink, "token": "wrong"}, CacheLifecycleState("engine").to_dict())

    try:
        asyncio.run(run())
        assert len(applied) == 1 and applied[0][0] == "a"
    finally:
        receiver.close()
