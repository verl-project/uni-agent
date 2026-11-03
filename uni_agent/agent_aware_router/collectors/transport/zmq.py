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
from dataclasses import dataclass, field
from typing import Callable

import zmq
import zmq.asyncio
from zmq.utils.monitor import recv_monitor_message

from ...collectors.transport.base import Transport

logger = logging.getLogger(__name__)


class _StreamReset(Exception):
    """Connection loss or incomplete delivery that requires a fresh session."""


class _MalformedFrame(_StreamReset):
    """Invalid wire framing, counted separately from recoverable interruptions."""


@dataclass
class _EndpointSocketSet:
    """Sockets and monitors owned by one replica subscription session."""

    replica_id: str
    context: zmq.asyncio.Context
    sub_socket: zmq.asyncio.Socket
    sub_monitor: zmq.asyncio.Socket
    replay_socket: zmq.asyncio.Socket | None = None
    replay_monitor: zmq.asyncio.Socket | None = None
    closed: bool = False
    pending_live: dict[int, bytes] = field(default_factory=dict)
    pending_bytes: int = 0


class ZMQTransport(Transport):
    """ZMQ transport — replay + sub dual socket per endpoint.

    Each endpoint gets its own ZMQ context, socket pair, and
    background coroutine — all endpoints subscribe concurrently.

    Connection loss invalidates the session. Recovery clears its cursor and
    starts replay from zero; sequence regression alone is treated as overlap.
    Replay is a bounded history, not a complete cache snapshot.

    Args:
        endpoints: ``{replica_id: [sub_ip:port, replay_ip:port, publisher, topic]}``
        base_retry_delay: Initial retry delay in seconds.
        max_retry_delay: Maximum retry delay cap in seconds.
        max_retry_attempts: Retries per recovery round, and retries for invalid
            framing before the endpoint stops. Recovery rounds cool down and
            continue until stopped.
        retry_backoff_factor: Exponential backoff multiplier.
        on_stream_reset: Gate and clear the replica's view before rebuilding,
            and keep it gated after a fatal consumer or protocol failure.
        on_stream_ready: Enable the rebuilt view after replay, or connection in
            subscription-only mode.
        on_stream_unavailable: Clear stale stream state and permit dispatch
            estimates while disconnected. If omitted, reset remains the fallback.
    """

    _CONNECTION_TIMEOUT = 5.0
    _REPLAY_TIMEOUT = 5.0
    _RECEIVE_BATCH_SIZE = 64
    _MAX_PENDING_MESSAGES = 1024
    _MAX_PENDING_BYTES = 16 * 1024 * 1024
    _MIN_RECOVERY_COOLDOWN = 0.1

    def __init__(
        self,
        endpoints: dict[str, list[str]],
        base_retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
        max_retry_attempts: int = 5,
        retry_backoff_factor: float = 2.0,
        on_stream_reset: Callable[[str], None] | None = None,
        on_stream_ready: Callable[[str], None] | None = None,
        on_stream_unavailable: Callable[[str], None] | None = None,
    ) -> None:
        self._replay_cursors: dict[str, int] = {}
        self._base_retry_delay = base_retry_delay
        self._max_retry_delay = max_retry_delay
        self._max_retry_attempts = max_retry_attempts
        self._retry_backoff_factor = retry_backoff_factor
        self._on_stream_reset = on_stream_reset
        self._on_stream_ready = on_stream_ready
        self._on_stream_unavailable = on_stream_unavailable

        # endpoints: {replica_id: [sub_endpoint, replay_endpoint, publisher, topic]}
        # replay_endpoint may be empty ("") for sub-only mode (e.g. standalone
        # collector that has no Ray RPC to discover the replay socket).
        self._sub_endpoints: dict[str, str] = {}
        self._replay_endpoints: dict[str, str] = {}
        self._topics: dict[str, str] = {}
        for replica_id, addrs in endpoints.items():
            if len(addrs) < 4:
                raise ValueError(
                    f"endpoint '{replica_id}' needs 4 elements [sub, replay, publisher, topic], got {len(addrs)}"
                )
            if addrs[2] != "zmq":
                raise ValueError(f"endpoint '{replica_id}' publisher must be 'zmq', got '{addrs[2]}'")
            self._sub_endpoints[replica_id] = f"tcp://{addrs[0]}"
            self._replay_endpoints[replica_id] = f"tcp://{addrs[1]}" if addrs[1] else ""
            self._topics[replica_id] = addrs[3]

        self._stopped = False
        self._endpoint_sockets: dict[str, _EndpointSocketSet] = {}
        self._retry_counts: dict[str, int] = {}
        self._malformed_counts: dict[str, int] = {}
        self._stream_states: dict[str, str] = {}
        self._sub_tasks: dict[str, asyncio.Task] = {}

    async def subscribe(self, handler: Callable[[bytes | str, str], None]) -> None:
        """Run endpoints independently; consumer exceptions stop only that endpoint."""
        self._stopped = False
        self._retry_counts.clear()
        self._malformed_counts.clear()
        self._stream_states.clear()
        tasks = {
            replica_id: asyncio.create_task(self._subscribe_for_endpoint(replica_id, handler))
            for replica_id in self._sub_endpoints
        }
        self._sub_tasks.update(tasks)
        try:
            await asyncio.gather(*tasks.values(), return_exceptions=True)
        finally:
            for task in tasks.values():
                task.cancel()
            try:
                await asyncio.gather(*tasks.values(), return_exceptions=True)
            finally:
                for replica_id, task in tasks.items():
                    if self._sub_tasks.get(replica_id) is task:
                        del self._sub_tasks[replica_id]

    def stop(self) -> None:
        """Cancel the current subscription and close its sockets and monitors."""
        self._stopped = True
        for task in self._sub_tasks.values():
            task.cancel()
        self._sub_tasks.clear()
        for sockets in list(self._endpoint_sockets.values()):
            self._close_zmq_sockets_for(sockets)
        self._replay_cursors.clear()

    # ── ZMQ connection management ───────────────────────────────────────

    async def _connect_zmq_for(self, replica_id: str) -> _EndpointSocketSet | None:
        """Install monitors before connecting; return sockets owned by this session."""
        ctx = zmq.asyncio.Context()
        opened: list[zmq.asyncio.Socket] = []
        try:
            sub_socket = ctx.socket(zmq.SUB)
            opened.append(sub_socket)
            sub_monitor = sub_socket.get_monitor_socket()
            opened.append(sub_monitor)
            sockets = _EndpointSocketSet(replica_id, ctx, sub_socket, sub_monitor)
            sub_socket.setsockopt_string(zmq.SUBSCRIBE, self._topics[replica_id])
            sub_socket.connect(self._sub_endpoints[replica_id])
            replay_addr = self._replay_endpoints[replica_id]
            if replay_addr:
                sockets.replay_socket = ctx.socket(zmq.DEALER)
                opened.append(sockets.replay_socket)
                sockets.replay_monitor = sockets.replay_socket.get_monitor_socket()
                opened.append(sockets.replay_monitor)
                sockets.replay_socket.connect(replay_addr)
            self._endpoint_sockets[replica_id] = sockets
            return sockets
        except Exception as exc:
            for socket in reversed(opened):
                socket.close(linger=0)
            ctx.term()
            if not isinstance(exc, zmq.ZMQError):
                raise
            logger.warning(f"ZMQ connection error for node {replica_id}: {exc}")
            return None

    def _close_zmq_sockets_for(self, sockets: _EndpointSocketSet) -> None:
        """Close this object only; a retired session cannot close its successor."""
        if self._endpoint_sockets.get(sockets.replica_id) is sockets:
            del self._endpoint_sockets[sockets.replica_id]
        if sockets.closed:
            return
        cleanup_failed = False
        for socket, monitor in (
            (sockets.sub_socket, sockets.sub_monitor),
            (sockets.replay_socket, sockets.replay_monitor),
        ):
            if socket is not None and not socket.closed:
                try:
                    socket.disable_monitor()
                except Exception as exc:
                    cleanup_failed = True
                    logger.debug("ZMQ monitor shutdown failed for %s: %s", sockets.replica_id, exc)
                try:
                    socket.close(linger=0)
                except Exception as exc:
                    cleanup_failed = True
                    logger.debug("ZMQ socket close failed for %s: %s", sockets.replica_id, exc)
            if monitor is not None:
                try:
                    monitor.close(linger=0)
                except Exception as exc:
                    cleanup_failed = True
                    logger.debug("ZMQ monitor close failed for %s: %s", sockets.replica_id, exc)
        if cleanup_failed:
            sockets.context.destroy(linger=0)
        else:
            sockets.context.term()
        sockets.pending_live.clear()
        sockets.pending_bytes = 0
        sockets.closed = True

    # ── Per-endpoint subscription ────────────────────────────────────────

    def _transition_stream(self, replica_id: str, state: str) -> None:
        """Notify availability changes once; fatal always retries safe invalidation."""
        previous = self._stream_states.get(replica_id)
        if state == previous and state != "fatal":
            if state != "unavailable" or self._on_stream_unavailable is not None:
                return
        callback = self._on_stream_reset
        if state == "unavailable" and self._on_stream_unavailable is not None:
            callback = self._on_stream_unavailable
        elif state == "ready":
            callback = self._on_stream_ready
        elif state == "rebuilding" and previous == "unavailable" and self._on_stream_unavailable is None:
            # Legacy callers already gated the view through the reset fallback.
            callback = None
        self._stream_states[replica_id] = state
        if callback is not None:
            callback(replica_id)
        if state != previous and state in ("unavailable", "rebuilding"):
            logger.info("ZMQ replica %s %s", replica_id, state)

    async def _subscribe_for_endpoint(self, replica_id: str, handler: Callable[[bytes, str], None]) -> None:
        sockets = None
        try:
            if self._stopped or self._sub_tasks.get(replica_id) is not asyncio.current_task():
                return
            self._replay_cursors.pop(replica_id, None)
            self._transition_stream(replica_id, "unavailable")
            while not self._stopped and self._sub_tasks.get(replica_id) is asyncio.current_task():
                malformed = False
                sockets = await self._connect_zmq_for(replica_id)
                if sockets is not None:
                    try:
                        await self._consume_session(sockets, handler)
                    except _StreamReset as exc:
                        malformed = isinstance(exc, _MalformedFrame)
                        logger.warning("ZMQ stream interrupted for %s: %s", replica_id, exc)
                    finally:
                        self._close_zmq_sockets_for(sockets)
                if self._stopped or self._sub_tasks.get(replica_id) is not asyncio.current_task():
                    return
                self._replay_cursors.pop(replica_id, None)
                if malformed:
                    failures = self._malformed_counts.get(replica_id, 0) + 1
                    self._malformed_counts[replica_id] = failures
                    if failures > self._max_retry_attempts:
                        self._transition_stream(replica_id, "fatal")
                        logger.warning("ZMQ malformed-frame retries exhausted for replica %s", replica_id)
                        return
                self._transition_stream(replica_id, "unavailable")
                await self._wait_to_retry(replica_id, malformed)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("ZMQ endpoint task failed for %s", replica_id)
            if self._sub_tasks.get(replica_id) is asyncio.current_task():
                self._replay_cursors.pop(replica_id, None)
                try:
                    self._transition_stream(replica_id, "fatal")
                except Exception:
                    logger.exception("ZMQ cache invalidation failed for %s", replica_id)
            raise
        finally:
            if sockets is not None:
                self._close_zmq_sockets_for(sockets)

    async def _wait_to_retry(self, replica_id: str, malformed: bool) -> None:
        """Bound recovery bursts without letting healthy replay hide invalid framing."""
        if malformed:
            retry_count = self._malformed_counts[replica_id] - 1
        else:
            retry_count = self._retry_counts.get(replica_id, 0)
            if retry_count >= self._max_retry_attempts:
                delay = max(self._max_retry_delay, self._MIN_RECOVERY_COOLDOWN)
                logger.warning("ZMQ recovery retries exhausted for %s; cooling down for %ss", replica_id, delay)
                self._retry_counts[replica_id] = 0
                await asyncio.sleep(delay)
                return
            self._retry_counts[replica_id] = retry_count + 1
        delay = min(
            self._base_retry_delay * (self._retry_backoff_factor**retry_count),
            self._max_retry_delay,
        )
        await asyncio.sleep(delay)

    async def _consume_session(self, sockets: _EndpointSocketSet, handler: Callable[[bytes, str], None]) -> None:
        """Connect, rebuild from replay and buffered live data, then consume live."""
        poller = zmq.asyncio.Poller()
        poller.register(sockets.sub_monitor, zmq.POLLIN)
        poller.register(sockets.sub_socket, zmq.POLLIN)
        if sockets.replay_monitor is not None:
            poller.register(sockets.replay_monitor, zmq.POLLIN)
        await self._wait_connected(sockets, poller)
        if self._stopped or self._endpoint_sockets.get(sockets.replica_id) is not sockets:
            raise asyncio.CancelledError
        self._transition_stream(sockets.replica_id, "rebuilding")
        if self._stopped or self._endpoint_sockets.get(sockets.replica_id) is not sockets:
            raise asyncio.CancelledError
        if sockets.replay_socket is not None:
            await self._consume_replay(sockets, poller, handler)
        await self._merge_pending_live(sockets, poller, handler)
        self._transition_stream(sockets.replica_id, "ready")
        logger.info("ZMQ replica %s ready", sockets.replica_id)
        await self._consume_live(sockets, poller, handler)

    async def _wait_connected(self, sockets: _EndpointSocketSet, poller: zmq.asyncio.Poller) -> None:
        """Keep draining SUB while waiting for both connection handshakes."""
        pending = {sockets.sub_monitor}
        if sockets.replay_monitor is not None:
            pending.add(sockets.replay_monitor)
        deadline = asyncio.get_running_loop().time() + self._CONNECTION_TIMEOUT
        while pending:
            events = await self._poll_session(sockets, poller, deadline, pending)
            if sockets.sub_socket in events:
                await self._buffer_live_frames(sockets)
            await asyncio.sleep(0)

    async def _poll_session(
        self,
        sockets: _EndpointSocketSet,
        poller: zmq.asyncio.Poller,
        deadline: float | None = None,
        pending_connections: set[zmq.asyncio.Socket] | None = None,
        *,
        wait: bool = True,
    ) -> dict:
        """Poll owned sockets, prioritize monitors, and enforce the phase deadline."""
        loop = asyncio.get_running_loop()
        timeout = None if deadline is None else max(1, int((deadline - loop.time()) * 1000))
        try:
            events = dict(await poller.poll(timeout if wait else 0))
        except zmq.ZMQError as exc:
            raise _StreamReset(f"poll failed: {exc}") from exc
        if self._stopped or self._endpoint_sockets.get(sockets.replica_id) is not sockets:
            raise asyncio.CancelledError
        for monitor in (sockets.sub_monitor, sockets.replay_monitor):
            if monitor is None or monitor not in events:
                continue
            for _ in range(self._RECEIVE_BATCH_SIZE):
                try:
                    event = await recv_monitor_message(monitor, flags=zmq.DONTWAIT)
                except zmq.Again:
                    break
                except zmq.ZMQError as exc:
                    raise _StreamReset(f"monitor failed: {exc}") from exc
                if event["event"] == zmq.EVENT_DISCONNECTED:
                    raise _StreamReset("connection interrupted")
                if event["event"] == zmq.EVENT_HANDSHAKE_SUCCEEDED and pending_connections is not None:
                    pending_connections.discard(monitor)
        if deadline is not None and loop.time() >= deadline:
            raise _StreamReset("connection or replay timed out")
        return events

    async def _receive_frame(self, socket: zmq.asyncio.Socket, *, replay: bool = False) -> tuple[int, bytes] | None:
        """Read and validate one wire frame; -1 denotes a valid replay end marker."""
        try:
            frames = await socket.recv_multipart(flags=zmq.DONTWAIT)
        except zmq.Again:
            return None
        except zmq.ZMQError as exc:
            raise _StreamReset(f"receive failed: {exc}") from exc
        if len(frames) < 3 or len(frames[-2]) != 8:
            raise _MalformedFrame("invalid sequence frame")
        if frames[-2] == (-1).to_bytes(8, "big", signed=True):
            if not replay or frames[-1]:
                raise _MalformedFrame("invalid replay end marker")
            return -1, b""
        if not frames[-1]:
            raise _MalformedFrame("empty payload")
        return int.from_bytes(frames[-2], "big"), frames[-1]

    async def _buffer_live_frames(self, sockets: _EndpointSocketSet) -> None:
        """Read a bounded batch without applying live updates ahead of replay."""
        for _ in range(self._RECEIVE_BATCH_SIZE):
            frame = await self._receive_frame(sockets.sub_socket)
            if frame is None:
                break
            sequence, payload = frame
            cursor = self._replay_cursors.get(sockets.replica_id)
            if (cursor is not None and sequence <= cursor) or sequence in sockets.pending_live:
                continue
            if (
                len(sockets.pending_live) >= self._MAX_PENDING_MESSAGES
                or sockets.pending_bytes + len(payload) > self._MAX_PENDING_BYTES
            ):
                raise _StreamReset("pending live buffer overflow")
            sockets.pending_live[sequence] = payload
            sockets.pending_bytes += len(payload)

    async def _consume_replay(
        self, sockets: _EndpointSocketSet, poller: zmq.asyncio.Poller, handler: Callable[[bytes, str], None]
    ) -> None:
        """Consume replay fairly with SUB, discarding overlap as replay advances."""
        try:
            await sockets.replay_socket.send_multipart([b"", (0).to_bytes(8, "big")])
        except zmq.ZMQError as exc:
            raise _StreamReset(f"replay request failed: {exc}") from exc
        poller.register(sockets.replay_socket, zmq.POLLIN)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._REPLAY_TIMEOUT
        while True:
            events = await self._poll_session(sockets, poller, deadline)
            if sockets.replay_socket in events:
                for _ in range(self._RECEIVE_BATCH_SIZE):
                    frame = await self._receive_frame(sockets.replay_socket, replay=True)
                    if frame is None:
                        break
                    sequence, payload = frame
                    if sequence == -1:
                        self._retire_replay(sockets, poller)
                        return
                    deadline = loop.time() + self._REPLAY_TIMEOUT
                    cursor = self._replay_cursors.get(sockets.replica_id)
                    if cursor is not None:
                        if sequence <= cursor:
                            continue
                        if sequence != cursor + 1:
                            raise _StreamReset(f"replay sequence gap: {cursor} -> {sequence}")
                    handler(payload, sockets.replica_id)
                    if self._stopped or self._endpoint_sockets.get(sockets.replica_id) is not sockets:
                        raise asyncio.CancelledError
                    self._replay_cursors[sockets.replica_id] = sequence
                    for covered in list(sockets.pending_live):
                        if covered <= sequence:
                            sockets.pending_bytes -= len(sockets.pending_live.pop(covered))
            if sockets.sub_socket in events:
                await self._buffer_live_frames(sockets)
            await asyncio.sleep(0)

    def _retire_replay(self, sockets: _EndpointSocketSet, poller: zmq.asyncio.Poller) -> None:
        """Retire the completed replay connection before enabling the live view."""
        try:
            poller.unregister(sockets.replay_socket)
            poller.unregister(sockets.replay_monitor)
            try:
                sockets.replay_socket.disable_monitor()
            finally:
                sockets.replay_socket.close(linger=0)
                sockets.replay_monitor.close(linger=0)
        except zmq.ZMQError as exc:
            raise _StreamReset(f"replay connection cleanup failed: {exc}") from exc
        sockets.replay_socket = None
        sockets.replay_monitor = None

    async def _merge_pending_live(
        self, sockets: _EndpointSocketSet, poller: zmq.asyncio.Poller, handler: Callable[[bytes, str], None]
    ) -> None:
        """Merge a finite buffered watermark; later arrivals belong to live delivery."""
        boundary = max(sockets.pending_live, default=None)
        while boundary is not None:
            events = await self._poll_session(sockets, poller, wait=False)
            sequences = sorted(sequence for sequence in sockets.pending_live if sequence <= boundary)
            for sequence in sequences[: self._RECEIVE_BATCH_SIZE]:
                payload = sockets.pending_live.pop(sequence)
                sockets.pending_bytes -= len(payload)
                cursor = self._replay_cursors.get(sockets.replica_id)
                if cursor is not None:
                    if sequence <= cursor:
                        continue
                    if sequence != cursor + 1:
                        raise _StreamReset(f"buffered live sequence gap: {cursor} -> {sequence}")
                handler(payload, sockets.replica_id)
                if self._stopped or self._endpoint_sockets.get(sockets.replica_id) is not sockets:
                    raise asyncio.CancelledError
                self._replay_cursors[sockets.replica_id] = sequence
                self._retry_counts[sockets.replica_id] = 0
                self._malformed_counts[sockets.replica_id] = 0
            if self._replay_cursors.get(sockets.replica_id, -1) >= boundary:
                return
            if sockets.sub_socket in events:
                await self._buffer_live_frames(sockets)
            await asyncio.sleep(0)

    async def _consume_live(
        self, sockets: _EndpointSocketSet, poller: zmq.asyncio.Poller, handler: Callable[[bytes, str], None]
    ) -> None:
        """Deliver ordered live frames, yielding after each finite batch."""
        while True:
            if sockets.pending_live:
                await self._merge_pending_live(sockets, poller, handler)
                await asyncio.sleep(0)
                continue
            events = await self._poll_session(sockets, poller)
            if sockets.sub_socket not in events:
                continue
            for _ in range(self._RECEIVE_BATCH_SIZE):
                frame = await self._receive_frame(sockets.sub_socket)
                if frame is None:
                    break
                sequence, payload = frame
                cursor = self._replay_cursors.get(sockets.replica_id)
                if cursor is not None:
                    if sequence <= cursor:
                        continue
                    if sequence != cursor + 1:
                        raise _StreamReset(f"live sequence gap: {cursor} -> {sequence}")
                handler(payload, sockets.replica_id)
                if self._stopped or self._endpoint_sockets.get(sockets.replica_id) is not sockets:
                    raise asyncio.CancelledError
                self._replay_cursors[sockets.replica_id] = sequence
                self._retry_counts[sockets.replica_id] = 0
                self._malformed_counts[sockets.replica_id] = 0
            await asyncio.sleep(0)
