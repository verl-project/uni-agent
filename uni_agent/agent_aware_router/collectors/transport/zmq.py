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
live events, and delivers uninterpreted messages to the handler callback.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable

import zmq
import zmq.asyncio

from ...collectors.transport.base import Transport, TransportMessage
from ...utils.endpoints import format_host_port, is_ipv6_host

logger = logging.getLogger(__name__)


def _tcp_endpoint(address: str) -> str:
    host, _, port = address.rpartition(":")
    if is_ipv6_host(host):
        address = format_host_port(host, port)
    return f"tcp://{address}"


def _socket(context, kind, endpoint):
    socket = context.socket(kind)
    host = endpoint.removeprefix("tcp://").rpartition(":")[0]
    if is_ipv6_host(host):
        socket.setsockopt(zmq.IPV6, 1)
    return socket


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
        cursors: Delivered replay positions, bound to their source addresses and topic.
    """

    def __init__(
        self,
        endpoints: dict[str, list[str]],
        base_retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
        max_retry_attempts: int = 5,
        retry_backoff_factor: float = 2.0,
        cursors: dict[str, tuple[str, str, str, int]] | None = None,
    ) -> None:
        self._replay_cursors: dict[str, int] = {}
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
            self._sub_endpoints[node_id] = _tcp_endpoint(addrs[0])
            self._replay_endpoints[node_id] = _tcp_endpoint(addrs[1]) if addrs[1] else ""
            self._topics[node_id] = addrs[3]

        for node_id, (sub, replay, topic, sequence) in (cursors or {}).items():
            if (
                self._sub_endpoints.get(node_id) == sub
                and self._replay_endpoints[node_id] == replay
                and self._topics[node_id] == topic
            ):
                self._replay_cursors[node_id] = sequence

        self._stopped = False
        self._endpoint_sockets: dict[str, _EndpointSocketSet] = {}
        self._retry_counts: dict[str, int] = {}
        self._sub_tasks: dict[str, asyncio.Task] = {}

    def get_cursors(self) -> dict[str, tuple[str, str, str, int]]:
        """Snapshot delivered progress together with the event source it belongs to."""
        return {
            node: (self._sub_endpoints[node], self._replay_endpoints[node], self._topics[node], sequence)
            for node, sequence in self._replay_cursors.items()
        }

    async def subscribe(self, handler: Callable[[TransportMessage, str], None]) -> None:
        """Subscribe until every endpoint task finishes or the subscription stops.

        A handler exception is fatal only to its endpoint: log its node and
        traceback, close its resources, and do not retry it as a network fault.
        Other endpoints continue. Completion represents the aggregate lifecycle,
        not per-endpoint health. Repair a failed consumer, then explicitly stop
        and restart the subscription to resume that endpoint.
        """
        self._stopped = False
        # A new session gets fresh attempts while retaining delivered replay positions.
        self._retry_counts.clear()
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

            sub_socket = _socket(ctx, zmq.SUB, sub_addr)
            sub_socket.connect(sub_addr)
            sub_socket.setsockopt_string(zmq.SUBSCRIBE, self._topics[node_id])

            replay_socket = None
            if replay_addr:
                replay_socket = _socket(ctx, zmq.DEALER, replay_addr)
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
        handler: Callable[[TransportMessage, str], None],
    ) -> None:
        """Deliver every frame; the cursor only selects future replay ranges."""
        handler(TransportMessage(payload, sequence), node_id)
        self._replay_cursors[node_id] = max(sequence, self._replay_cursors.get(node_id, sequence))

    async def _subscribe_for_endpoint(
        self,
        node_id: str,
        sub_addr: str,
        replay_addr: str,
        handler: Callable[[TransportMessage, str], None],
    ) -> None:
        try:
            if node_id not in self._replay_cursors:
                handler(TransportMessage(b"", status="new_stream"), node_id)
            if not await self._connect_zmq_for(node_id, sub_addr, replay_addr):
                if not await self._reconnect_with_backoff_for(node_id, sub_addr, replay_addr):
                    handler(TransportMessage(b"", status="unreachable"), node_id)
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
                        raise ValueError("invalid subscription sequence frame")
                    sequence = int.from_bytes(parts[-2], "big")
                except (zmq.ZMQError, ValueError) as exc:
                    logger.warning("ZMQ subscription error for %s: %s", node_id, exc)
                    if not replay_addr:
                        handler(TransportMessage(b"", status="no_continuity"), node_id)
                    self._close_zmq_sockets_for(node_id)
                    if not await self._reconnect_with_backoff_for(node_id, sub_addr, replay_addr):
                        handler(TransportMessage(b"", status="unreachable"), node_id)
                        break
                    if replay_addr:
                        await self._replay_historical_data_for(node_id, handler)
                    continue
                self._deliver(node_id, sequence, parts[-1], handler)
                self._retry_counts[node_id] = 0
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("ZMQ endpoint task failed for %s", node_id)
            raise
        finally:
            self._close_zmq_sockets_for(node_id)

    async def _replay_historical_data_for(
        self,
        node_id: str,
        handler: Callable[[TransportMessage, str], None],
    ) -> bool:
        """Read every streamed ROUTER reply through END_SEQ using DEALER."""
        sockets = self._endpoint_sockets.get(node_id)
        if sockets is None or sockets.replay_socket is None:
            return False
        resume_pending = node_id in self._replay_cursors
        try:
            start = self._replay_cursors.get(node_id, -1) + 1
            await sockets.replay_socket.send_multipart([b"", start.to_bytes(8, "big")])
        except (zmq.ZMQError, ValueError) as exc:
            return self._replay_failed(node_id, sockets, handler, exc)
        while not self._stopped:
            try:
                frames = await asyncio.wait_for(sockets.replay_socket.recv_multipart(), timeout=5.0)
                if len(frames) < 3 or len(frames[-2]) != 8:
                    raise ValueError("invalid replay sequence frame")
                replay_ended = frames[-2] == (-1).to_bytes(8, "big", signed=True)
                if replay_ended:
                    if frames[-1] not in {b"", b"kv_replay_truncated"}:
                        raise ValueError("unexpected payload on replay end marker")
                else:
                    if not frames[-1]:
                        raise ValueError("unexpected empty replay payload")
                    sequence = int.from_bytes(frames[-2], "big")
            except (asyncio.TimeoutError, zmq.ZMQError, ValueError) as exc:
                return self._replay_failed(node_id, sockets, handler, exc)
            if replay_ended:
                if frames[-1]:
                    handler(TransportMessage(b"", status="no_continuity"), node_id)
                return True
            if resume_pending and sequence >= start:
                resume_pending = False
                if sequence > start:
                    handler(TransportMessage(b"", status="no_continuity"), node_id)
            self._deliver(node_id, sequence, frames[-1], handler)
        return False

    def _replay_failed(self, node_id, sockets, handler, error) -> bool:
        if self._stopped:
            return False
        logger.warning("Replay incomplete for %s: %s", node_id, error)
        # Retire the old identity before another request: late streamed replies
        # must not become the next request's response.
        sockets.replay_socket.close(0)
        replacement = _socket(sockets.context, zmq.DEALER, self._replay_endpoints[node_id])
        replacement.connect(self._replay_endpoints[node_id])
        sockets.replay_socket = replacement
        handler(TransportMessage(b"", status="no_continuity"), node_id)
        return False
