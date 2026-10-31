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

"""ZMQTransport — ZMQ replay + sub dual socket transport.

Connects to per-endpoint ZMQ addresses (sub + replay), subscribes to
live events, and delivers raw payloads to the handler callback.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable

import zmq
import zmq.asyncio

from ...collectors.transport.base import Transport
from ...types.cache_lifecycle import KVEventFrame

logger = logging.getLogger(__name__)


@dataclass
class _EndpointSocketSet:
    """Per-endpoint ZMQ socket bundle (internal — not exported)."""

    node_id: str
    context: zmq.asyncio.Context
    sub_socket: zmq.asyncio.Socket
    replay_socket: zmq.asyncio.Socket | None
    closed: bool = False


class ZMQTransport(Transport):
    """ZMQ transport — replay + sub dual socket per endpoint.

    Each endpoint gets its own ZMQ context, socket pair, and
    background coroutine — all endpoints subscribe concurrently.

    Args:
        endpoints: ``{node_id: [sub_ip:port, replay_ip:port, publisher, topic]}``
        base_retry_delay: Initial retry delay in seconds.
        max_retry_delay: Maximum retry delay cap in seconds.
        max_retry_attempts: Maximum number of retries per endpoint.
        retry_backoff_factor: Exponential backoff multiplier.
    """

    def __init__(
        self,
        endpoints: dict[str, list[str]],
        base_retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
        max_retry_attempts: int = 5,
        retry_backoff_factor: float = 2.0,
        deliver_frames: bool = False,
        cursors: dict[str, tuple[str, str, str, int]] | None = None,
    ) -> None:
        self._deliver_frames = deliver_frames
        self._last_sequences: dict[str, int] = {}
        self._base_retry_delay = base_retry_delay
        self._max_retry_delay = max_retry_delay
        self._max_retry_attempts = max_retry_attempts
        self._retry_backoff_factor = retry_backoff_factor

        # endpoints: {node_id: [sub_endpoint, replay_endpoint, publisher, topic]}
        # replay_endpoint may be empty ("") for sub-only mode (e.g. standalone
        # collector that has no Ray RPC to discover the replay socket).
        self._sub_endpoints: dict[str, str] = {}
        self._replay_endpoints: dict[str, str] = {}
        self._topics: dict[str, str] = {}
        for node_id, addrs in endpoints.items():
            if len(addrs) < 4:
                raise ValueError(
                    f"endpoint '{node_id}' needs 4 elements [sub, replay, publisher, topic], got {len(addrs)}"
                )
            if addrs[2] != "zmq":
                raise ValueError(f"endpoint '{node_id}' publisher must be 'zmq', got '{addrs[2]}'")
            self._sub_endpoints[node_id] = f"tcp://{addrs[0]}"
            self._replay_endpoints[node_id] = f"tcp://{addrs[1]}" if addrs[1] else ""
            self._topics[node_id] = addrs[3]

        for node_id, (sub, replay, topic, sequence) in (cursors or {}).items():
            if (
                self._sub_endpoints.get(node_id) == sub
                and self._replay_endpoints[node_id] == replay
                and self._topics[node_id] == topic
            ):
                self._last_sequences[node_id] = sequence

        self._stopped = False
        self._endpoint_sockets: dict[str, _EndpointSocketSet] = {}
        self._retry_counts: dict[str, int] = {}
        self._sub_tasks: dict[str, asyncio.Task] = {}

    def get_cursors(self) -> dict[str, tuple[str, str, str, int]]:
        """Snapshot progress together with the event source it belongs to."""
        return {
            node: (self._sub_endpoints[node], self._replay_endpoints[node], self._topics[node], sequence)
            for node, sequence in self._last_sequences.items()
        }

    async def subscribe(self, handler: Callable[[bytes | str | KVEventFrame, str], None]) -> None:
        """Spawn per-endpoint subscription tasks, deliver payloads to handler."""
        self._stopped = False
        sub_tasks = []
        for node_id in self._sub_endpoints:
            sub_addr = self._sub_endpoints[node_id]
            replay_addr = self._replay_endpoints[node_id]
            t = asyncio.create_task(self._subscribe_for_endpoint(node_id, sub_addr, replay_addr, handler))
            self._sub_tasks[node_id] = t
            sub_tasks.append(t)
        try:
            await asyncio.gather(*sub_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for t in self._sub_tasks.values():
                t.cancel()
        finally:
            self._close_all_zmq_sockets()

    def stop(self) -> None:
        """Signal stop, cancel tasks, close ZMQ sockets. No loop dependency.

        This method only include:
          1. sets the stop flag so subscribe loops exit,
          2. cancels tasks (synchronous, no loop ref needed),
          3. closes sockets/contexts (idempotent via ``closed`` guard).
        """
        self._stopped = True
        for task in self._sub_tasks.values():
            task.cancel()
        self._sub_tasks.clear()
        self._close_all_zmq_sockets()

    # ── ZMQ connection management ───────────────────────────────────────

    async def _connect_zmq_for(
        self,
        node_id: str,
        sub_addr: str,
        replay_addr: str,
    ) -> bool:
        """Create ZMQ context and replay + sub dual socket for a single endpoint.

        If ``replay_addr`` is empty (sub-only mode), the replay DEALER socket is
        skipped — used by the standalone collector which has no replay endpoint.
        """
        try:
            ctx = zmq.asyncio.Context()

            sub_socket = ctx.socket(zmq.SUB)
            sub_socket.connect(sub_addr)
            sub_socket.setsockopt_string(zmq.SUBSCRIBE, self._topics[node_id])

            replay_socket = None
            if replay_addr:
                replay_socket = ctx.socket(zmq.DEALER)
                replay_socket.connect(replay_addr)

            self._endpoint_sockets[node_id] = _EndpointSocketSet(
                node_id=node_id,
                context=ctx,
                sub_socket=sub_socket,
                replay_socket=replay_socket,
            )
            return True

        except zmq.ZMQError as exc:
            logger.warning(f"ZMQ connection error for node {node_id}: {exc}")
            self._close_zmq_sockets_for(node_id)
            return False

    def _close_zmq_sockets_for(self, node_id: str) -> None:
        """Safely close ZMQ sockets and context for a single endpoint."""
        sockets = self._endpoint_sockets.pop(node_id, None)
        if sockets is None or sockets.closed:
            return
        sockets.closed = True
        sockets.sub_socket.close(linger=0)
        if sockets.replay_socket is not None:
            sockets.replay_socket.close(linger=0)
        sockets.context.term()

    def _close_all_zmq_sockets(self) -> None:
        """Close all per-endpoint ZMQ sockets and contexts."""
        for node_id in list(self._endpoint_sockets.keys()):
            self._close_zmq_sockets_for(node_id)

    async def _reconnect_with_backoff_for(
        self,
        node_id: str,
        sub_addr: str,
        replay_addr: str,
    ) -> bool:
        """Exponential backoff reconnect for a single endpoint."""
        retry_count = self._retry_counts.get(node_id, 0)
        while retry_count < self._max_retry_attempts:
            delay = min(
                self._base_retry_delay * (self._retry_backoff_factor**retry_count),
                self._max_retry_delay,
            )
            await asyncio.sleep(delay)
            retry_count += 1
            self._retry_counts[node_id] = retry_count

            if await self._connect_zmq_for(node_id, sub_addr, replay_addr):
                return True

        return False

    # ── Per-endpoint subscription ────────────────────────────────────────

    def _deliver(
        self,
        node_id: str,
        sequence: int,
        payload: bytes,
        handler: Callable[[bytes | str | KVEventFrame, str], None],
    ) -> None:
        last = self._last_sequences.get(node_id)
        if last is not None and sequence <= last:
            return  # Live SUB and replay overlap; each batch applies once.
        if last is not None and sequence != last + 1:
            expected = last + 1
            error = f"kv_sequence_gap:{expected}:{sequence}"
            logger.warning("KV event gap for %s: expected=%s received=%s", node_id, expected, sequence)
            if self._deliver_frames:
                handler(KVEventFrame(b"", error=error), node_id)
        self._last_sequences[node_id] = sequence
        handler(KVEventFrame(payload, sequence) if self._deliver_frames else payload, node_id)

    async def _subscribe_for_endpoint(
        self,
        node_id: str,
        sub_addr: str,
        replay_addr: str,
        handler: Callable[[bytes | str | KVEventFrame, str], None],
    ) -> None:
        try:
            if self._deliver_frames and node_id not in self._last_sequences:
                handler(KVEventFrame(b"", reset_stream=True), node_id)
            if not await self._connect_zmq_for(node_id, sub_addr, replay_addr):
                if not await self._reconnect_with_backoff_for(node_id, sub_addr, replay_addr):
                    if self._deliver_frames:
                        handler(KVEventFrame(b"", error="kv_subscription_failed"), node_id)
                    return
            if replay_addr:
                await self._replay_historical_data_for(node_id, handler)
            while not self._stopped:
                sockets = self._endpoint_sockets.get(node_id)
                if sockets is None:
                    return
                try:
                    parts = await sockets.sub_socket.recv_multipart()
                    if len(parts) < 3 or len(parts[-2]) != 8:
                        raise ValueError("invalid KV event sequence frame")
                    sequence = int.from_bytes(parts[-2], "big")
                    last = self._last_sequences.get(node_id)
                    expected = 0 if last is None else last + 1
                    if sequence > expected and replay_addr:
                        await self._replay_historical_data_for(node_id, handler)
                    self._deliver(node_id, sequence, parts[-1], handler)
                    self._retry_counts[node_id] = 0
                except (zmq.ZMQError, ValueError) as exc:
                    logger.warning("KV subscription error for %s: %s", node_id, exc)
                    if self._deliver_frames and not replay_addr:
                        handler(KVEventFrame(b"", error="kv_subscription_failed"), node_id)
                    self._close_zmq_sockets_for(node_id)
                    if not await self._reconnect_with_backoff_for(node_id, sub_addr, replay_addr):
                        if self._deliver_frames and replay_addr:
                            handler(KVEventFrame(b"", error="kv_subscription_failed"), node_id)
                        break
                    if replay_addr:
                        await self._replay_historical_data_for(node_id, handler)
        except asyncio.CancelledError:
            pass
        finally:
            self._close_zmq_sockets_for(node_id)

    async def _replay_historical_data_for(
        self,
        node_id: str,
        handler: Callable[[bytes | str | KVEventFrame, str], None],
    ) -> bool:
        """Read streamed ROUTER replies through END_SEQ using DEALER."""
        sockets = self._endpoint_sockets.get(node_id)
        if sockets is None or sockets.replay_socket is None:
            return False
        try:
            start = self._last_sequences.get(node_id, -1) + 1
            await sockets.replay_socket.send_multipart([b"", start.to_bytes(8, "big")])
            while not self._stopped:
                frames = await asyncio.wait_for(sockets.replay_socket.recv_multipart(), timeout=5.0)
                if len(frames) < 3 or len(frames[-2]) != 8:
                    raise ValueError("invalid KV replay frame")
                if frames[-2] == (-1).to_bytes(8, "big", signed=True) and not frames[-1]:
                    return True
                if not frames[-1]:
                    raise ValueError("unexpected empty KV replay payload")
                self._deliver(node_id, int.from_bytes(frames[-2], "big"), frames[-1], handler)
        except (asyncio.TimeoutError, zmq.ZMQError, ValueError) as exc:
            logger.warning("KV replay incomplete for %s: %s", node_id, exc)
            if self._deliver_frames:
                handler(KVEventFrame(b"", error="kv_replay_incomplete"), node_id)
            # A timed-out streamed reply can arrive later. Retire the socket so
            # it cannot be mistaken for the next replay request's response.
            sockets.replay_socket.close(0)
            replacement = sockets.context.socket(zmq.DEALER)
            replacement.connect(self._replay_endpoints[node_id])
            sockets.replay_socket = replacement
        return False
