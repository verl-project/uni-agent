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
from ..utils.endpoints import format_host_port, is_ipv6_host

logger = logging.getLogger(__name__)


class KVEventBridge:
    def __init__(self, lifecycle, host, *, replay_steps=10000, replay_bytes=64 * 1024**2, queue_bytes=16 * 1024**2):
        self.lifecycle = lifecycle
        self.host = host
        self.replay_steps, self.replay_bytes, self.queue_limit = replay_steps, replay_bytes, queue_bytes
        self._buffer = deque()
        self._buffer_bytes = 0
        self._last_evicted = None
        self._queue_bytes = 0
        self._queue = asyncio.Queue(maxsize=1024)
        self._sequence = -1
        self._context = zmq.asyncio.Context()
        self._pub = self._context.socket(zmq.PUB)
        self._replay = self._context.socket(zmq.ROUTER)
        ipv6 = is_ipv6_host(host)
        for socket in (self._pub, self._replay):
            socket.setsockopt(zmq.LINGER, 0)
            if ipv6:
                socket.setsockopt(zmq.IPV6, 1)
        bind_address = "tcp://[::]" if ipv6 else "tcp://0.0.0.0"
        self._pub_port = self._pub.bind_to_random_port(bind_address)
        self._replay_port = self._replay.bind_to_random_port(bind_address)
        self._tasks = []
        self._native_task = None
        self._native = None
        self._native_sequence = None
        self._overflow_reported = False

    def sources(self):
        return {
            "0": {
                "publisher": "zmq",
                "topic": "uni-agent-kv",
                "endpoint": f"tcp://{format_host_port(self.host, self._pub_port)}",
                "replay_endpoint": f"tcp://{format_host_port(self.host, self._replay_port)}",
            }
        }

    async def start(self, native_source):
        self._consumer = asyncio.create_task(self._consume())
        self._tasks = [self._consumer, asyncio.create_task(self._serve_replay())]
        await self.attach(native_source)

    async def attach(self, native_source):
        if native_source is None:
            await self.lifecycle.stream_gap("native_events_disabled")
            return
        endpoint = native_source["endpoint"].removeprefix("tcp://")
        replay = (native_source.get("replay_endpoint") or "").removeprefix("tcp://")
        self._native = ZMQTransport({"native": [endpoint, replay, "zmq", native_source["topic"]]})
        self._native_task = asyncio.create_task(self._native.subscribe(self._enqueue))
        # Socket setup and subscription run before launch returns to callers.
        await asyncio.sleep(0)

    async def _stop_native(self):
        if self._native is not None:
            self._native.stop()
        if self._native_task is not None:
            self._native_task.cancel()
            await asyncio.gather(self._native_task, return_exceptions=True)
        self._native = self._native_task = None

    async def rebind(self, native_source, *, cache_metadata=None):
        self._consumer.cancel()
        await asyncio.gather(self._consumer, return_exceptions=True)
        await self._stop_native()
        finished = {task for task in self._tasks if task.done()}
        await asyncio.gather(*finished, return_exceptions=True)
        self._tasks = [task for task in self._tasks if task not in finished]
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()
        self._queue_bytes = 0
        self._buffer.clear()
        self._buffer_bytes = 0
        self._last_evicted = None
        self._overflow_reported = False
        self._native_sequence = None
        await self.lifecycle.new_instance(cache_metadata=cache_metadata)
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
                task.add_done_callback(self._invalidation_finished)
            return
        self._queue_bytes += size
        self._queue.put_nowait(frame)

    def _invalidation_finished(self, task):
        if task in self._tasks:
            self._tasks.remove(task)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error(
                    "Could not invalidate cache estimates after bridge queue overflow",
                    exc_info=(type(error), error, error.__traceback__),
                )

    async def _consume(self):
        while True:
            frame = await self._queue.get()
            self._queue_bytes -= len(frame.payload)
            try:
                if frame.status == "new_stream":
                    # Initial attach and rebind already start with a cold
                    # lifecycle; the transport marker contains no native batch.
                    self._native_sequence = None
                    continue
                error = f"event_stream_{frame.status}" if frame.status != "ok" else None
                native_gap = False
                if error is None and frame.sequence is not None:
                    last = self._native_sequence
                    if last is not None and frame.sequence <= last:
                        continue
                    self._native_sequence = frame.sequence
                    if last is not None and frame.sequence != last + 1:
                        # The bridge assigns its own delivery sequence, so native
                        # loss must invalidate the lifecycle before re-numbering.
                        error = f"native_kv_sequence_gap:{last + 1}:{frame.sequence}"
                        native_gap = True
                if error:
                    await self.lifecycle.stream_gap(error)
                    # A producer-confirmed reset boundary can establish a new
                    # epoch even when this batch follows missing native events.
                    payload = await self.lifecycle.accept(frame.payload) if native_gap else None
                    if payload is not None:
                        error = None
                else:
                    payload = await self.lifecycle.accept(frame.payload)
                if payload is not None and self.lifecycle.state.index_state == "ready":
                    self._overflow_reported = False
                await self.forward(payload, error)
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
            sequence, instance, epoch, old = self._buffer.popleft()
            self._last_evicted = (instance, epoch, sequence)
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
            # An empty window can also mean all missing events were evicted.
            # Only loss in the current generation invalidates this replay.
            truncated = (
                not retained
                and self._last_evicted is not None
                and self._last_evicted[:2] == (instance, epoch)
                and start <= self._last_evicted[2]
            )
            await self._replay.send_multipart(
                [
                    identity,
                    b"",
                    b"",
                    (-1).to_bytes(8, "big", signed=True),
                    b"kv_replay_truncated" if truncated else b"",
                ]
            )

    async def close(self):
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await self._stop_native()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._pub.close(0)
        self._replay.close(0)
        self._context.term()
