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

"""Independent TCP publishers and subscription cleanup for collector tests."""

import asyncio
from contextlib import asynccontextmanager, contextmanager

import zmq
import zmq.asyncio

END_SEQ = (-1).to_bytes(8, "big", signed=True)


class Publisher:
    def __init__(self, addresses=None, *, replay=True):
        self.context = zmq.asyncio.Context()
        self.pub = self.context.socket(zmq.XPUB)
        self.router = self.context.socket(zmq.ROUTER) if replay else None
        try:
            self.pub.setsockopt(zmq.XPUB_VERBOSE, 1)
            if addresses is None:
                self.pub_address = f"127.0.0.1:{self.pub.bind_to_random_port('tcp://127.0.0.1')}"
                self.replay_address = (
                    f"127.0.0.1:{self.router.bind_to_random_port('tcp://127.0.0.1')}" if replay else ""
                )
            else:
                self.pub_address, self.replay_address = addresses
                self.pub.bind("tcp://" + self.pub_address)
                if self.router is not None:
                    self.router.bind("tcp://" + self.replay_address)
        except Exception:
            self.close()
            raise
        self.addresses = (self.pub_address, self.replay_address)
        self.endpoints = {"a": [*self.addresses, "zmq", "topic"]}

    def close_socket(self, name):
        getattr(self, name).close(0)

    def reopen(self, name):
        socket = self.context.socket(zmq.XPUB if name == "pub" else zmq.ROUTER)
        if name == "pub":
            socket.setsockopt(zmq.XPUB_VERBOSE, 1)
        try:
            socket.bind("tcp://" + (self.pub_address if name == "pub" else self.replay_address))
        except zmq.ZMQError:
            socket.close(0)
            raise
        setattr(self, name, socket)

    async def publish(self, sequence, payload):
        await self.pub.send_multipart([b"topic", sequence.to_bytes(8, "big"), payload])

    async def requested_start(self):
        identity, delimiter, start = await asyncio.wait_for(self.router.recv_multipart(), 5)
        assert delimiter == b""
        return identity, int.from_bytes(start, "big")

    async def batch(self, identity, sequence, payload):
        await self.router.send_multipart([identity, b"", sequence.to_bytes(8, "big"), payload])

    async def end(self, identity):
        await self.router.send_multipart([identity, b"", END_SEQ, b""])

    def close(self):
        self.pub.close(0)
        if self.router is not None:
            self.router.close(0)
        self.context.term()


@contextmanager
def publisher(replay=True, addresses=None):
    source = Publisher(addresses, replay=replay)
    try:
        yield source
    finally:
        source.close()


def observe(transport, handler=None):
    received = asyncio.Queue()
    if handler is None:
        handler = lambda payload, replica: received.put_nowait((replica, payload))
    task = asyncio.create_task(transport.subscribe(handler))
    return received, task


async def stop(transport, task):
    transport.stop()
    await asyncio.wait_for(task, 3)


@asynccontextmanager
async def subscription(transport, handler=None):
    received, task = observe(transport, handler)
    try:
        yield received, task
    finally:
        await stop(transport, task)
