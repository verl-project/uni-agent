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

import msgpack
import pytest
import zmq
import zmq.asyncio

from uni_agent.agent_aware_router.collectors.collector import Collector
from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.collectors.transport.lifecycle import LifecycleControlReceiver
from uni_agent.agent_aware_router.collectors.transport.zmq import ZMQTransport
from uni_agent.agent_aware_router.server.cache_lifecycle import CacheLifecycle
from uni_agent.agent_aware_router.server.kv_event_bridge import KVEventBridge
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.store.kv_cache_store import KVCacheStore
from uni_agent.agent_aware_router.types.cache_lifecycle import CacheLifecycleState
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


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


def test_rebuilt_collector_stays_cold_until_current_clear_then_recovers(monkeypatch, layout, stored):
    monkeypatch.setattr(KVCacheStore, "_instance", None)

    async def run():
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
                deliver_frames=True,
            )
            collector = Collector(
                transport, VLLMKVParser(group_aware=True, lifecycle_reader=store.cache_lifecycle_state)
            )
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
            await control.begin("reset", 1)
            await control.finish(True)
            assert hit() == 0
            await asyncio.to_thread(collector.stop)
            collector = new_collector()
            # An empty current-epoch replay must not expose the old buffer.
            await asyncio.sleep(0.1)
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
