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

"""Acknowledged cache lifecycle notifications."""

import asyncio
import hmac
import threading
import uuid

import msgpack
import zmq
import zmq.asyncio


class LifecycleControlReceiver:
    """Receive lifecycle updates on a dedicated thread."""

    def __init__(self, handler, advertise_host: str, timeout: float = 5.0):
        self.handler = handler
        self.timeout = timeout
        self.token = uuid.uuid4().hex
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="cache-lifecycle-control", daemon=True)
        self._error = None
        self._thread.start()
        if not self._ready.wait(timeout):
            self.close()
            raise TimeoutError("cache lifecycle receiver startup timed out")
        if self._error is not None:
            raise self._error
        self.endpoint = f"tcp://{advertise_host}:{self._port}"

    def descriptor(self, server_id: str) -> dict:
        return {"endpoint": self.endpoint, "token": self.token, "server_id": server_id}

    def _run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            self._port = socket.bind_to_random_port("tcp://0.0.0.0")
            self._ready.set()
            while not self._stop.is_set():
                if not socket.poll(50):
                    continue
                try:
                    request = msgpack.unpackb(socket.recv(), raw=False)
                    supplied = request.get("token", "")
                    if not isinstance(supplied, str) or not hmac.compare_digest(supplied, self.token):
                        raise ValueError("invalid lifecycle capability")
                    result = self.handler(request["server_id"], request["state"])
                    response = {"ok": True, "result": result}
                except Exception as exc:
                    response = {"ok": False, "error": str(exc)}
                socket.send(msgpack.packb(response, use_bin_type=True))
        except Exception as exc:
            self._error = exc
            self._ready.set()
        finally:
            socket.close(0)
            context.term()

    def close(self):
        self._stop.set()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=self.timeout)


async def notify_lifecycle(sink: dict, state: dict, timeout: float = 5.0):
    # A new socket after a timeout keeps REQ's send/receive state well-defined.
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(sink["endpoint"])
        request = {"token": sink["token"], "server_id": sink["server_id"], "state": state}
        await socket.send(msgpack.packb(request, use_bin_type=True))
        reply = msgpack.unpackb(await asyncio.wait_for(socket.recv(), timeout), raw=False)
        if not reply.get("ok"):
            raise RuntimeError(f"cache lifecycle acknowledgement failed: {reply.get('error')}")
        return reply.get("result")
    finally:
        socket.close(0)
        context.term()
