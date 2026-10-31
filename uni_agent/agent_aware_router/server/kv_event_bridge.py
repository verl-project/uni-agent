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

"""Forward native KV events with cache epochs and bounded replay."""

import asyncio
import logging
from collections import deque

import msgpack
import zmq
import zmq.asyncio

from ..collectors.transport.zmq import ZMQTransport
from ..types.cache_lifecycle import BRIDGE_SCHEMA_KEY

logger = logging.getLogger(__name__)


class KVEventBridge:
    def __init__(self, lifecycle, host, *, replay_steps=10000, replay_bytes=64 * 1024**2, queue_bytes=16 * 1024**2):
        self.lifecycle = lifecycle
        self.host = host
        self.replay_steps, self.replay_bytes, self.queue_limit = replay_steps, replay_bytes, queue_bytes
        self._buffer = deque()
        self._buffer_bytes = 0
        self._queue_bytes = 0
        self._queue = asyncio.Queue(maxsize=1024)
        self._sequence = -1
        self._context = zmq.asyncio.Context()
        self._pub = self._context.socket(zmq.PUB)
        self._replay = self._context.socket(zmq.ROUTER)
        self._pub.setsockopt(zmq.LINGER, 0)
        self._replay.setsockopt(zmq.LINGER, 0)
        self._pub_port = self._pub.bind_to_random_port("tcp://0.0.0.0")
        self._replay_port = self._replay.bind_to_random_port("tcp://0.0.0.0")
        self._tasks = []
        self._native_task = None
        self._native = None
        self._overflow_reported = False

    def sources(self):
        return {
            "0": {
                "publisher": "zmq",
                "topic": "uni-agent-kv",
                "endpoint": f"tcp://{self.host}:{self._pub_port}",
                "replay_endpoint": f"tcp://{self.host}:{self._replay_port}",
            }
        }

    async def start(self, native_source):
        self._consumer = asyncio.create_task(self._consume())
        self._tasks = [self._consumer, asyncio.create_task(self._serve_replay())]
        await self.attach(native_source)

    async def attach(self, native_source):
        if self._native is not None:
            self._native.stop()
        if self._native_task is not None:
            self._native_task.cancel()
            await asyncio.gather(self._native_task, return_exceptions=True)
        if native_source is None:
            await self.lifecycle.stream_gap("native_events_disabled")
            return
        endpoint = native_source["endpoint"].removeprefix("tcp://")
        replay = (native_source.get("replay_endpoint") or "").removeprefix("tcp://")
        self._native = ZMQTransport({"native": [endpoint, replay, "zmq", native_source["topic"]]}, deliver_frames=True)
        self._native_task = asyncio.create_task(self._native.subscribe(self._enqueue))
        # Socket setup and subscription run before launch returns to callers.
        await asyncio.sleep(0)

    async def rebind(self, native_source):
        self._consumer.cancel()
        await asyncio.gather(self._consumer, return_exceptions=True)
        if self._native is not None:
            self._native.stop()
        if self._native_task is not None:
            self._native_task.cancel()
            await asyncio.gather(self._native_task, return_exceptions=True)
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()
        self._queue_bytes = 0
        self._buffer.clear()
        self._buffer_bytes = 0
        self._overflow_reported = False
        await self.lifecycle.new_instance()
        self._consumer = asyncio.create_task(self._consume())
        self._tasks.append(self._consumer)
        await self.attach(native_source)

    def _enqueue(self, frame, node):
        size = len(frame.payload)
        if self._queue.full() or self._queue_bytes + size > self.queue_limit:
            if not self._overflow_reported:
                self._overflow_reported = True
                task = asyncio.create_task(self.lifecycle.stream_gap("bridge_queue_overflow"))
                self._tasks.append(task)
            return
        self._queue_bytes += size
        self._queue.put_nowait(frame)

    async def _consume(self):
        while True:
            frame = await self._queue.get()
            self._queue_bytes -= len(frame.payload)
            try:
                if frame.error:
                    await self.lifecycle.stream_gap(frame.error)
                    payload = None
                else:
                    payload = await self.lifecycle.accept(frame.payload)
                await self.forward(payload, frame.error)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Cache event bridge failed; disabling cache estimates")
                try:
                    await self.lifecycle.stream_gap("bridge_processing_failed")
                except Exception:
                    # Notification failure has already closed request admission.
                    logger.exception("Could not acknowledge bridge invalidation")
            finally:
                self._queue.task_done()

    async def forward(self, payload, error=None):
        self._sequence += 1
        encoded = msgpack.packb(
            {
                BRIDGE_SCHEMA_KEY: 1,
                "state": self.lifecycle.state.to_dict(),
                "sequence": self._sequence,
                "payload": payload or b"",
                "error": error,
            },
            use_bin_type=True,
        )
        state = self.lifecycle.state
        self._buffer.append((self._sequence, state.instance_id, state.epoch, encoded))
        self._buffer_bytes += len(encoded)
        while self._buffer and (len(self._buffer) > self.replay_steps or self._buffer_bytes > self.replay_bytes):
            _, _, _, old = self._buffer.popleft()
            self._buffer_bytes -= len(old)
        await self._pub.send_multipart([b"uni-agent-kv", self._sequence.to_bytes(8, "big"), encoded])

    async def _serve_replay(self):
        while True:
            frames = await self._replay.recv_multipart()
            if len(frames) != 3 or frames[1] or len(frames[2]) != 8:
                continue
            identity, _, start_bytes = frames
            start = int.from_bytes(start_bytes, "big")
            epoch, instance = self.lifecycle.state.epoch, self.lifecycle.state.instance_id
            # Snapshot the current instance and epoch before sending replies.
            retained = []
            for sequence, batch_instance, batch_epoch, payload in self._buffer:
                if sequence < start:
                    continue
                if batch_instance == instance and batch_epoch == epoch:
                    retained.append((sequence, payload))
            for sequence, payload in retained:
                await self._replay.send_multipart(
                    [identity, b"", b"uni-agent-kv", sequence.to_bytes(8, "big"), payload]
                )
            await self._replay.send_multipart([identity, b"", b"", (-1).to_bytes(8, "big", signed=True), b""])

    async def close(self):
        if self._native is not None:
            self._native.stop()
        tasks = self._tasks + ([self._native_task] if self._native_task else [])
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._pub.close(0)
        self._replay.close(0)
        self._context.term()
