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

"""Protocol status and replay delivery applied to the hybrid cache consumer."""

import asyncio

import msgpack
import pytest
import zmq
import zmq.asyncio

from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.collectors.transport.base import TransportMessage
from uni_agent.agent_aware_router.collectors.transport.zmq import ZMQTransport
from uni_agent.agent_aware_router.store.data_store import DataStore

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
            base_retry_delay=0,
        )
        store = DataStore()
        store.configure_cache_registry(layout((("full_attention", 4),), unit=None))
        parser = VLLMKVParser()
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


@pytest.mark.parametrize("metadata_available", [False, True], ids=["inferred-full", "native-layout"])
def test_fresh_sub_only_source_resets_old_residency_and_accepts_nonzero_sequence(
    metadata_available, layout, stored, removed
):
    async def run():
        store = DataStore()
        if metadata_available:
            store.configure_cache_registry(layout((("full_attention", 4),), unit=None))
        parser = VLLMKVParser()

        def old_or_native(event):
            if not metadata_available:
                event.pop("group_idx", None)
                event.pop("kv_cache_spec_kind", None)
            return event

        initial = msgpack.packb([0, [old_or_native(stored(tokens=list(range(4)), hashes=["old"]))]])
        store.apply_kv_update(parser.parse(TransportMessage(initial, 100), "a"))
        other = msgpack.packb([0, [old_or_native(stored(tokens=list(range(100, 104)), hashes=["other"]))]])
        store.apply_kv_update(parser.parse(TransportMessage(other, 110), "b"))
        assert store.per_replica_block_counts() == {"a": 1, "b": 1}
        ctx = zmq.asyncio.Context()
        pub = ctx.socket(zmq.XPUB)
        port = pub.bind_to_random_port("tcp://127.0.0.1")
        transport = ZMQTransport({"a": [f"127.0.0.1:{port}", "", "zmq", "kv"]})
        delivered = asyncio.Queue()

        def handler(frame, node):
            store.apply_kv_update(parser.parse(frame, node))
            delivered.put_nowait(frame)

        task = asyncio.create_task(transport.subscribe(handler))
        try:
            assert await asyncio.wait_for(pub.recv(), 3) == b"\x01kv"
            assert (await asyncio.wait_for(delivered.get(), 3)).status == "new_stream"
            assert store.per_replica_block_counts() == {"b": 1}
            assert "a" not in store._kv._event_sequences
            for sequence, events, count in [
                (5, [old_or_native(stored(tokens=list(range(4, 8)), hashes=["new"]))], 1),
                (6, [old_or_native(removed(0, ["new"]))], 0),
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
