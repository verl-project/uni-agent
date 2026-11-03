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

"""Real TCP coverage for replay, connection loss, and isolated ZMQ sessions."""

import asyncio
import logging

import pytest
import zmq
import zmq.asyncio

from tests.uni_agent.agent_aware_router.collectors.zmq_helpers import (
    END_SEQ,
)
from tests.uni_agent.agent_aware_router.collectors.zmq_helpers import (
    observe as _observe,
)
from tests.uni_agent.agent_aware_router.collectors.zmq_helpers import (
    publisher as _source,
)
from tests.uni_agent.agent_aware_router.collectors.zmq_helpers import (
    stop as _stop,
)
from tests.uni_agent.agent_aware_router.collectors.zmq_helpers import (
    subscription as _subscription,
)
from uni_agent.agent_aware_router.collectors.transport.zmq import ZMQTransport, _MalformedFrame

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


async def _next(queue):
    return await asyncio.wait_for(queue.get(), 3)


async def _subscribed(pub):
    while await asyncio.wait_for(pub.recv(), 3) != b"\x01topic":
        pass


async def _request(router):
    identity, delimiter, start = await asyncio.wait_for(router.recv_multipart(), 3)
    assert delimiter == b"" and int.from_bytes(start, "big") == 0
    return identity


async def _reply(router, identity, batches, *, end=True):
    for sequence, payload in batches:
        await router.send_multipart([identity, b"", b"topic", sequence.to_bytes(8, "big"), payload])
    if end:
        await router.send_multipart([identity, b"", b"", END_SEQ, b""])


async def _respond(router, batches):
    identity = await _request(router)
    await _reply(router, identity, batches)
    return identity


def _transport(endpoints, **kwargs):
    return ZMQTransport(endpoints, base_retry_delay=0, **kwargs)


def test_replay_suffix_accepts_nonzero_baseline_and_deduplicates_older_frames():
    async def run():
        with _source() as source:
            ready = asyncio.Queue()
            resets = []
            transport = _transport(source.endpoints, on_stream_reset=resets.append, on_stream_ready=ready.put_nowait)
            async with _subscription(transport) as (received, task):
                await _subscribed(source.pub)
                await _respond(source.router, [(40, b"forty"), (40, b"duplicate"), (39, b"older"), (41, b"next")])
                assert await _next(ready) == "a"
                assert await _next(received) == ("a", b"forty")
                assert await _next(received) == ("a", b"next")
                await source.publish(40, b"queued older")
                await source.publish(41, b"queued duplicate")
                await source.publish(42, b"live")
                assert await _next(received) == ("a", b"live")
                assert transport._replay_cursors == {"a": 42} and resets == ["a"]

    asyncio.run(run())


@pytest.mark.parametrize("restart_sequence", [0, 80], ids=["sequence-reset", "already-above-old-cursor"])
def test_same_address_publisher_restart_invalidates_before_replay(restart_sequence):
    async def run():
        with _source() as source:
            resets, ready, received = asyncio.Queue(), asyncio.Queue(), asyncio.Queue()
            cache = {b"stale"}
            trusted = set()

            def reset(replica):
                cache.clear()
                trusted.discard(replica)
                resets.put_nowait(replica)

            def mark_ready(replica):
                trusted.add(replica)
                ready.put_nowait(replica)

            def handler(payload, replica):
                cache.add(payload)
                received.put_nowait((replica, payload))

            transport = _transport(source.endpoints, on_stream_reset=reset, on_stream_ready=mark_ready)
            async with _subscription(transport, handler) as (_, task):
                assert await _next(resets) == "a" and not cache
                await _subscribed(source.pub)
                old = transport._endpoint_sockets["a"]
                replay_socket, replay_monitor = old.replay_socket, old.replay_monitor
                await _respond(source.router, [(40, b"old generation")])
                await _next(ready)
                assert await _next(received) == ("a", b"old generation")
                old = transport._endpoint_sockets["a"]
                source.close_socket("pub")
                source.close_socket("router")
                assert await _next(resets) == "a"
                assert not cache and not trusted and "a" not in transport._replay_cursors
                source.reopen("pub")
                source.reopen("router")
                await _subscribed(source.pub)
                await _respond(source.router, [(restart_sequence, b"new generation")])
                assert await _next(ready) == "a"
                assert await _next(received) == ("a", b"new generation")
                assert cache == {b"new generation"} and trusted == {"a"}
                assert transport._replay_cursors == {"a": restart_sequence}
                current = transport._endpoint_sockets["a"]
                transport._close_zmq_sockets_for(old)
                assert old.closed and old.sub_monitor.closed and replay_monitor.closed and replay_socket.closed
                assert transport._endpoint_sockets["a"] is current and not current.closed
                await source.publish(restart_sequence + 1, b"still new")
                assert await _next(received) == ("a", b"still new")

    asyncio.run(run())


@pytest.mark.parametrize(
    "disconnected_socket", ["pub", "router"], ids=["only-sub-disconnects", "only-replay-disconnects"]
)
def test_sub_or_incomplete_replay_disconnect_retires_the_whole_session(disconnected_socket):
    async def run():
        with _source() as source:
            resets, ready = asyncio.Queue(), asyncio.Queue()
            transport = _transport(
                source.endpoints, on_stream_reset=resets.put_nowait, on_stream_ready=ready.put_nowait
            )
            async with _subscription(transport) as (received, task):
                await _next(resets)
                await _subscribed(source.pub)
                identity = await _request(source.router)
                old = transport._endpoint_sockets["a"]
                replay_socket, replay_monitor = old.replay_socket, old.replay_monitor
                await _reply(source.router, identity, [(10, b"old")], end=disconnected_socket == "pub")
                if disconnected_socket == "pub":
                    await _next(ready)
                await _next(received)
                source.close_socket(disconnected_socket)
                assert await _next(resets) == "a"
                source.reopen(disconnected_socket)
                await _subscribed(source.pub)
                await _respond(source.router, [(0, b"fresh")])
                await _next(ready)
                assert await _next(received) == ("a", b"fresh")
                assert old.closed and old.sub_socket.closed and replay_socket.closed
                assert old.sub_monitor.closed and replay_monitor.closed and old.context.closed

    asyncio.run(run())


def test_stop_then_publisher_restart_does_not_reuse_the_previous_cursor():
    async def run():
        with _source() as source:
            resets, ready = asyncio.Queue(), asyncio.Queue()
            transport = _transport(
                source.endpoints, on_stream_reset=resets.put_nowait, on_stream_ready=ready.put_nowait
            )
            received, task = _observe(transport)
            second = None
            try:
                await _next(resets)
                await _subscribed(source.pub)
                await _respond(source.router, [(40, b"before stop")])
                await _next(ready)
                await _next(received)
                await _stop(transport, task)
                assert not transport._replay_cursors
                source.close_socket("pub")
                source.close_socket("router")
                # The stopped transport has no monitor that could observe this restart.
                source.reopen("pub")
                source.reopen("router")
                received, second = _observe(transport)
                assert await _next(resets) == "a"
                await _subscribed(source.pub)
                await _respond(source.router, [(0, b"after restart")])
                await _next(ready)
                assert await _next(received) == ("a", b"after restart")
                assert transport._replay_cursors == {"a": 0}
            finally:
                await _stop(transport, second if second is not None else task)

    asyncio.run(run())


def test_old_subscription_cleanup_cannot_close_a_new_subscription():
    async def run():
        with _source(False) as source:
            ready = asyncio.Queue()
            transport = _transport(source.endpoints, on_stream_ready=ready.put_nowait)
            received, first = _observe(transport)
            second = None
            try:
                await _subscribed(source.pub)
                await _next(ready)
                old = transport._endpoint_sockets["a"]
                transport.stop()
                # Start before awaiting the old subscribe coroutine's finally block.
                received, second = _observe(transport)
                await _subscribed(source.pub)
                await _next(ready)
                await first
                current = transport._endpoint_sockets["a"]
                transport._close_zmq_sockets_for(old)
                await source.publish(0, b"new session")
                assert await _next(received) == ("a", b"new session")
                assert transport._endpoint_sockets["a"] is current and not current.closed
            finally:
                await _stop(transport, second if second is not None else first)

    asyncio.run(run())


def test_empty_replay_allows_an_arbitrary_first_live_baseline():
    async def run():
        with _source() as source:
            ready = asyncio.Queue()
            transport = _transport(source.endpoints, on_stream_ready=ready.put_nowait)
            async with _subscription(transport) as (received, task):
                await _subscribed(source.pub)
                await _respond(source.router, [])
                assert await _next(ready) == "a" and not transport._replay_cursors
                await source.publish(900, b"first")
                await source.publish(901, b"next")
                assert await _next(received) == ("a", b"first")
                assert await _next(received) == ("a", b"next")

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["replay", "live"])
def test_sequence_gaps_invalidate_partial_state_and_replay_from_zero(phase):
    async def run():
        with _source() as source:
            resets, ready = asyncio.Queue(), asyncio.Queue()
            cache = set()

            def reset(replica):
                cache.clear()
                resets.put_nowait(replica)

            transport = _transport(source.endpoints, on_stream_reset=reset, on_stream_ready=ready.put_nowait)
            async with _subscription(transport, lambda payload, _: cache.add(payload)) as (_, task):
                await _next(resets)
                await _subscribed(source.pub)
                identity = await _request(source.router)
                await _reply(source.router, identity, [(4, b"stale")], end=phase == "live")
                if phase == "live":
                    await _next(ready)
                    await source.publish(6, b"gap")
                else:
                    await _reply(source.router, identity, [(6, b"gap")])
                assert await _next(resets) == "a" and not cache
                assert ready.empty()
                await _subscribed(source.pub)
                await _respond(source.router, [(0, b"rebuilt")])
                await _next(ready)
                assert cache == {b"rebuilt"} and transport._replay_cursors == {"a": 0}

    asyncio.run(run())


@pytest.mark.parametrize(
    "bad_reply",
    [[b"short", b"payload"], [(0).to_bytes(8, "big"), b""], [END_SEQ, b"unexpected"], [b""]],
)
def test_malformed_replay_never_marks_ready_and_retries_with_a_fresh_identity(bad_reply):
    async def run():
        with _source() as source:
            resets, ready = asyncio.Queue(), asyncio.Queue()
            transport = _transport(
                source.endpoints, on_stream_reset=resets.put_nowait, on_stream_ready=ready.put_nowait
            )
            async with _subscription(transport) as (received, task):
                await _next(resets)
                await _subscribed(source.pub)
                old_identity = await _request(source.router)
                old = transport._endpoint_sockets["a"]
                await source.router.send_multipart([old_identity, b"", *bad_reply])
                assert await _next(resets) == "a" and ready.empty() and not transport._replay_cursors
                await _subscribed(source.pub)
                identity = await _request(source.router)
                assert identity != old_identity and old.closed
                await _reply(source.router, identity, [(0, b"fresh")])
                await _next(ready)
                assert await _next(received) == ("a", b"fresh")

    asyncio.run(run())


def test_replay_timeout_retires_identity_and_excludes_late_replies():
    async def run():
        with _source() as source:
            resets, ready = asyncio.Queue(), asyncio.Queue()
            transport = _transport(
                source.endpoints, on_stream_reset=resets.put_nowait, on_stream_ready=ready.put_nowait
            )
            transport._REPLAY_TIMEOUT = 0.05
            async with _subscription(transport) as (received, task):
                await _next(resets)
                await _subscribed(source.pub)
                old_identity = await _request(source.router)
                old = transport._endpoint_sockets["a"]
                assert await _next(resets) == "a" and ready.empty()
                transport._REPLAY_TIMEOUT = 3
                await _subscribed(source.pub)
                identity = await _request(source.router)
                assert identity != old_identity and old.closed
                await _reply(source.router, old_identity, [(99, b"late")])
                await _reply(source.router, identity, [(0, b"fresh")])
                await _next(ready)
                assert await _next(received) == ("a", b"fresh")
                assert transport._replay_cursors == {"a": 0}

    asyncio.run(run())


def test_initially_unreachable_connection_recovers_after_the_retry_budget_cools_down():
    async def run():
        with _source(False) as source:
            source.close_socket("pub")
            resets, ready = asyncio.Queue(), asyncio.Queue()
            transport = _transport(
                source.endpoints,
                max_retry_attempts=1,
                max_retry_delay=0,
                on_stream_reset=resets.put_nowait,
                on_stream_ready=ready.put_nowait,
            )
            transport._CONNECTION_TIMEOUT = 0.03
            async with _subscription(transport) as (received, task):
                for _ in range(3):
                    await _next(resets)
                assert not task.done(), "a recoverable outage must not permanently stop the endpoint"
                assert ready.empty() and not transport._replay_cursors
                source.reopen("pub")
                await _subscribed(source.pub)
                await _next(ready)
                await source.publish(0, b"available again")
                assert await _next(received) == ("a", b"available again")

    asyncio.run(run())


@pytest.mark.parametrize("replay", [False, True], ids=["subscription-only", "successful-replay-each-time"])
def test_bad_live_frames_exhaust_retries_even_after_successful_handshake_or_replay(replay):
    async def run():
        with _source(replay) as source:
            resets, ready = asyncio.Queue(), asyncio.Queue()
            transport = _transport(
                source.endpoints,
                max_retry_attempts=1,
                on_stream_reset=resets.put_nowait,
                on_stream_ready=ready.put_nowait,
            )
            async with _subscription(transport) as (received, task):
                await _next(resets)
                for _ in range(2):
                    await _subscribed(source.pub)
                    if replay:
                        await _respond(source.router, [(5, b"replayed")])
                        assert await _next(received) == ("a", b"replayed")
                    await _next(ready)
                    await source.pub.send_multipart([b"topic", b"bad", b"payload"])
                    assert await _next(resets) == "a"
                await asyncio.wait_for(task, 3)
                assert not transport._replay_cursors and not transport._endpoint_sockets

    asyncio.run(run())


def test_valid_live_data_restores_retry_budget_and_sub_only_restart_recovers():
    async def run():
        with _source(False) as source:
            resets, ready = asyncio.Queue(), asyncio.Queue()
            transport = _transport(
                source.endpoints,
                max_retry_attempts=1,
                on_stream_reset=resets.put_nowait,
                on_stream_ready=ready.put_nowait,
            )
            async with _subscription(transport) as (received, task):
                await _next(resets)
                for sequence in (40, 0, 100):
                    await _subscribed(source.pub)
                    await _next(ready)
                    await source.publish(sequence, b"valid")
                    assert await _next(received) == ("a", b"valid")
                    assert transport._retry_counts == {"a": 0}
                    source.close_socket("pub")
                    assert await _next(resets) == "a" and "a" not in transport._replay_cursors
                    source.reopen("pub")
                assert not task.done()

    asyncio.run(run())


def test_consumer_error_invalidates_partial_writes_and_does_not_reconnect(caplog, monkeypatch):
    async def run():
        with _source(False) as bad, _source(False) as good:
            cache = {"a": set(), "b": set()}
            ready = asyncio.Queue()
            received = asyncio.Queue()
            resets = []

            def reset(replica):
                cache[replica].clear()
                resets.append(replica)

            transport = _transport(
                {"a": bad.endpoints["a"], "b": good.endpoints["a"]},
                on_stream_reset=reset,
                on_stream_ready=ready.put_nowait,
            )

            def handler(payload, replica):
                cache[replica].add(payload)
                if replica == "a":
                    raise ValueError("consumer rejected the message after writing")
                received.put_nowait((replica, payload))

            async with _subscription(transport, handler) as (_, task):
                await _subscribed(bad.pub)
                await _subscribed(good.pub)
                assert {await _next(ready), await _next(ready)} == {"a", "b"}
                sources = dict(transport._endpoint_sockets)
                await bad.publish(0, b"rejected")
                with pytest.raises(ValueError, match="consumer rejected"):
                    await asyncio.wait_for(transport._sub_tasks["a"], 3)
                assert not cache["a"] and "a" not in transport._replay_cursors
                assert sources["a"].closed and not sources["b"].closed and not task.done()
                await good.publish(0, b"still alive")
                assert await _next(received) == ("b", b"still alive")
                assert resets.count("a") == 2 and resets.count("b") == 1
            assert not transport._endpoint_sockets and not transport._sub_tasks
            assert all(source.context.closed and source.sub_monitor.closed for source in sources.values())

    module_logger = logging.getLogger(ZMQTransport.__module__)
    monkeypatch.setattr(module_logger, "propagate", False)
    module_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.ERROR, logger=ZMQTransport.__module__):
            asyncio.run(run())
    finally:
        module_logger.removeHandler(caplog.handler)
    records = [record for record in caplog.records if record.name == ZMQTransport.__module__]
    assert len(records) == 1 and "ZMQ endpoint task failed for a" in records[0].getMessage()
    assert records[0].exc_info[0] is ValueError


def test_setup_failure_closes_the_partially_created_socket_and_context(monkeypatch):
    async def run():
        context = zmq.asyncio.Context()
        opened = []
        original_socket = context.socket

        def socket(*args, **kwargs):
            created = original_socket(*args, **kwargs)
            opened.append(created)
            return created

        def fail_monitor(*_args, **_kwargs):
            raise zmq.ZMQError(zmq.EINVAL)

        monkeypatch.setattr(context, "socket", socket)
        monkeypatch.setattr(zmq.asyncio, "Context", lambda: context)
        monkeypatch.setattr(zmq.asyncio.Socket, "get_monitor_socket", fail_monitor)
        transport = _transport({"a": ["127.0.0.1:1", "", "zmq", "topic"]})
        assert await transport._connect_zmq_for("a") is None
        assert context.closed and len(opened) == 1 and opened[0].closed
        assert not transport._endpoint_sockets

    asyncio.run(run())


def test_monitor_shutdown_error_still_closes_both_socket_pairs(monkeypatch):
    async def run():
        with _source() as source:
            ready = asyncio.Queue()
            transport = _transport(source.endpoints, on_stream_ready=ready.put_nowait)
            _, task = _observe(transport)
            try:
                await _subscribed(source.pub)
                sockets = transport._endpoint_sockets["a"]
                replay_socket, replay_monitor = sockets.replay_socket, sockets.replay_monitor
                await _respond(source.router, [])
                await _next(ready)
                disable = zmq.asyncio.Socket.disable_monitor

                def fail_one_monitor(socket):
                    if socket is sockets.sub_socket:
                        raise zmq.ZMQError(zmq.EINVAL)
                    return disable(socket)

                monkeypatch.setattr(zmq.asyncio.Socket, "disable_monitor", fail_one_monitor)
                await _stop(transport, task)
                assert sockets.closed and sockets.context.closed
                assert all(
                    socket.closed
                    for socket in (
                        sockets.sub_socket,
                        sockets.sub_monitor,
                        replay_socket,
                        replay_monitor,
                    )
                )
            finally:
                await _stop(transport, task)

    asyncio.run(run())


def test_retired_endpoint_cannot_reset_the_current_replica_state():
    async def run():
        resets = []
        transport = _transport({"a": ["127.0.0.1:1", "", "zmq", "topic"]}, on_stream_reset=resets.append)
        owner = asyncio.create_task(asyncio.Event().wait())
        transport._sub_tasks["a"] = owner
        transport._replay_cursors["a"] = 42
        try:
            stale = asyncio.create_task(transport._subscribe_for_endpoint("a", lambda *_: None))
            await stale
            assert not resets and transport._replay_cursors == {"a": 42}
            assert transport._sub_tasks == {"a": owner}
        finally:
            transport.stop()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(run())


def test_replay_consumer_error_invalidates_and_stops_instead_of_retrying():
    async def run():
        with _source() as source:
            resets, ready = [], []
            cache = set()

            def reset(replica):
                cache.clear()
                resets.append(replica)

            def handler(payload, _replica):
                cache.add(payload)
                raise ValueError("replay consumer failed after writing")

            transport = _transport(source.endpoints, on_stream_reset=reset, on_stream_ready=ready.append)
            async with _subscription(transport, handler) as (_, task):
                await _subscribed(source.pub)
                endpoint = transport._sub_tasks["a"]
                await _respond(source.router, [(0, b"partial write")])
                with pytest.raises(ValueError, match="replay consumer failed"):
                    await asyncio.wait_for(endpoint, 3)
                await asyncio.wait_for(task, 3)
                assert not ready and not cache and resets == ["a", "a"]
                assert not transport._replay_cursors and not transport._retry_counts
                assert not transport._endpoint_sockets

    asyncio.run(run())


@pytest.fixture
def sub_frames_read(monkeypatch):
    """Observe real SUB reads without changing their socket behavior."""
    frames = asyncio.Queue()
    recv_multipart = zmq.asyncio.Socket.recv_multipart

    async def recv(socket, *args, **kwargs):
        is_sub = socket.type == zmq.SUB
        result = await recv_multipart(socket, *args, **kwargs)
        if is_sub:
            frames.put_nowait(result)
        return result

    monkeypatch.setattr(zmq.asyncio.Socket, "recv_multipart", recv)
    return frames


@pytest.mark.parametrize("phase", ["connecting", "replay"])
def test_sub_is_read_before_replay_finishes_and_merged_before_ready(phase, sub_frames_read):
    async def run():
        with _source() as source:
            if phase == "connecting":
                source.close_socket("router")
            snapshots, received = asyncio.Queue(), []
            transport = _transport(source.endpoints, on_stream_ready=lambda _: snapshots.put_nowait(list(received)))
            async with _subscription(transport, lambda payload, _: received.append(payload)) as (_, task):
                await _subscribed(source.pub)
                identity = await _request(source.router) if phase == "replay" else None
                for sequence in range(1, 9):
                    await source.publish(sequence, str(sequence).encode())
                    assert (await _next(sub_frames_read))[-1] == str(sequence).encode()
                assert received == [] and snapshots.empty()
                if phase == "connecting":
                    source.reopen("router")
                    identity = await _request(source.router)
                await _reply(source.router, identity, [(0, b"0")])
                assert await _next(snapshots) == [str(sequence).encode() for sequence in range(9)]
                assert transport._replay_cursors == {"a": 8}

    asyncio.run(run())


@pytest.mark.parametrize("bound", ["messages", "bytes"])
def test_pending_live_buffer_overflow_invalidates_and_rebuilds(bound, sub_frames_read):
    async def run():
        with _source() as source:
            resets, ready = asyncio.Queue(), asyncio.Queue()
            cache = set()

            def reset(replica):
                cache.clear()
                resets.put_nowait(replica)

            transport = _transport(source.endpoints, on_stream_reset=reset, on_stream_ready=ready.put_nowait)
            transport._MAX_PENDING_MESSAGES = 1 if bound == "messages" else 100
            transport._MAX_PENDING_BYTES = 5 if bound == "bytes" else 1024
            async with _subscription(transport, lambda payload, _: cache.add(payload)) as (_, task):
                await _next(resets)
                await _subscribed(source.pub)
                identity = await _request(source.router)
                await _reply(source.router, identity, [(0, b"root")], end=False)
                await source.publish(1, b"one")
                await _next(sub_frames_read)
                await source.publish(2, b"two")
                await _next(sub_frames_read)
                assert await _next(resets) == "a" and not cache and ready.empty()
                await _subscribed(source.pub)
                await _respond(source.router, [(0, b"rebuilt")])
                await _next(ready)
                assert cache == {b"rebuilt"}

    asyncio.run(run())


def test_replay_fills_pending_gaps_and_reclaims_overlap_buffer_space(sub_frames_read):
    async def run():
        with _source() as source:
            resets, ready, received = [], asyncio.Queue(), asyncio.Queue()
            transport = _transport(source.endpoints, on_stream_reset=resets.append, on_stream_ready=ready.put_nowait)
            transport._MAX_PENDING_MESSAGES = 2
            transport._MAX_PENDING_BYTES = 4
            async with _subscription(transport, lambda payload, _: received.put_nowait(payload)) as (_, task):
                await _subscribed(source.pub)
                identity = await _request(source.router)
                await source.publish(0, b"0")
                assert (await _next(sub_frames_read))[-1] == b"0"
                await _reply(source.router, identity, [(0, b"0")], end=False)
                assert await _next(received) == b"0"
                for sequence in (2, 4):
                    await source.publish(sequence, str(sequence).encode())
                    await _next(sub_frames_read)
                await _reply(source.router, identity, [(1, b"1"), (2, b"2")], end=False)
                assert [await _next(received), await _next(received)] == [b"1", b"2"]
                await source.publish(5, b"5")
                await _next(sub_frames_read)
                await _reply(source.router, identity, [(3, b"3"), (4, b"4")])
                await _next(ready)
                assert [await _next(received), await _next(received), await _next(received)] == [b"3", b"4", b"5"]
                assert resets == ["a"] and transport._replay_cursors == {"a": 5}

    asyncio.run(run())


def test_completed_replay_retires_dealer_before_ready_and_keeps_healthy_sub():
    async def run():
        with _source() as source:
            resets, ready = [], asyncio.Queue()
            sockets = None
            replay_socket = replay_monitor = None

            def mark_ready(replica):
                ready.put_nowait((replica, replay_socket.closed, replay_monitor.closed))

            transport = _transport(source.endpoints, on_stream_reset=resets.append, on_stream_ready=mark_ready)
            async with _subscription(transport) as (received, task):
                await _subscribed(source.pub)
                identity = await _request(source.router)
                sockets = transport._endpoint_sockets["a"]
                replay_socket, replay_monitor = sockets.replay_socket, sockets.replay_monitor
                await _reply(source.router, identity, [])
                assert await _next(ready) == ("a", True, True)
                assert sockets.replay_socket is None and sockets.replay_monitor is None
                source.close_socket("router")
                await source.publish(0, b"live stays healthy")
                assert await _next(received) == ("a", b"live stays healthy")
                assert resets == ["a"] and transport._endpoint_sockets["a"] is sockets

    asyncio.run(run())


def test_recoverable_gaps_cool_down_and_resume_without_a_new_subscription():
    async def run():
        with _source() as source:
            resets, ready = asyncio.Queue(), asyncio.Queue()
            transport = _transport(
                source.endpoints,
                max_retry_attempts=1,
                max_retry_delay=0,
                on_stream_reset=resets.put_nowait,
                on_stream_ready=ready.put_nowait,
            )
            async with _subscription(transport) as (received, task):
                await _next(resets)
                for _ in range(2):
                    await _subscribed(source.pub)
                    await _respond(source.router, [(0, b"replay succeeded")])
                    await _next(ready)
                    await _next(received)
                    await source.publish(2, b"gap")
                    await _next(resets)
                assert not task.done(), "recoverable gaps must not permanently stop the endpoint"
                await _subscribed(source.pub)
                await _respond(source.router, [(0, b"healthy replay")])
                await _next(ready)
                assert await _next(received) == ("a", b"healthy replay")
                await source.publish(1, b"healthy live")
                assert await _next(received) == ("a", b"healthy live")
                assert transport._retry_counts == {"a": 0}

    asyncio.run(run())


def test_recoverable_cooldown_does_not_refill_the_malformed_frame_budget():
    async def run():
        with _source() as source:
            resets, ready = asyncio.Queue(), asyncio.Queue()
            transport = _transport(
                source.endpoints,
                max_retry_attempts=1,
                max_retry_delay=0,
                on_stream_reset=resets.put_nowait,
                on_stream_ready=ready.put_nowait,
            )
            async with _subscription(transport) as (received, task):
                await _next(resets)
                for malformed in (True, False, False, True):
                    await _subscribed(source.pub)
                    await _respond(source.router, [(0, b"healthy replay")])
                    await _next(ready)
                    await _next(received)
                    if malformed:
                        await source.pub.send_multipart([b"topic", b"bad", b"payload"])
                    else:
                        await source.publish(2, b"gap")
                    await _next(resets)
                    if not malformed:
                        assert not task.done()
                await asyncio.wait_for(task, 3)
                assert not transport._endpoint_sockets and not transport._replay_cursors

    asyncio.run(run())


def test_continuous_live_traffic_does_not_hold_replay_readiness_open(sub_frames_read):
    async def run():
        with _source() as source:
            ready, received = asyncio.Queue(), asyncio.Queue()
            transport = _transport(source.endpoints, on_stream_ready=ready.put_nowait)
            transport._RECEIVE_BATCH_SIZE = 1
            task = asyncio.create_task(transport.subscribe(lambda payload, _: received.put_nowait(payload)))
            producer = None

            async def publish_continuously():
                sequence = 9
                while True:
                    for _ in range(16):
                        await source.publish(sequence, str(sequence).encode())
                        sequence += 1
                    await asyncio.sleep(0)

            try:
                await _subscribed(source.pub)
                identity = await _request(source.router)
                await _reply(source.router, identity, [(0, b"0")], end=False)
                assert await _next(received) == b"0"
                for sequence in range(1, 9):
                    await source.publish(sequence, str(sequence).encode())
                    await _next(sub_frames_read)
                producer = asyncio.create_task(publish_continuously())
                await _reply(source.router, identity, [])
                assert await _next(ready) == "a"
                assert [await _next(received) for _ in range(8)] == [str(sequence).encode() for sequence in range(1, 9)]
                assert not producer.done()
            finally:
                if producer is not None:
                    producer.cancel()
                    await asyncio.gather(producer, return_exceptions=True)
                await _stop(transport, task)

    asyncio.run(run())


def test_valid_live_progress_resets_the_malformed_frame_streak():
    async def run():
        with _source(False) as source:
            resets, ready = asyncio.Queue(), asyncio.Queue()
            transport = _transport(
                source.endpoints,
                max_retry_attempts=1,
                on_stream_reset=resets.put_nowait,
                on_stream_ready=ready.put_nowait,
            )
            async with _subscription(transport) as (received, task):
                await _next(resets)
                for baseline in (0, 100, 200):
                    await _subscribed(source.pub)
                    await _next(ready)
                    await source.publish(baseline, b"healthy live")
                    assert await _next(received) == ("a", b"healthy live")
                    if baseline != 200:
                        await source.pub.send_multipart([b"topic", b"bad", b"payload"])
                        await _next(resets)
                assert not task.done()

    asyncio.run(run())


def test_buffered_live_progress_resets_retries_before_a_later_merge_error():
    async def run():
        with _source(False) as source:
            transport = _transport(source.endpoints, max_retry_attempts=1)
            transport._RECEIVE_BATCH_SIZE = 1
            sockets = await transport._connect_zmq_for("a")
            poller = zmq.asyncio.Poller()
            poller.register(sockets.sub_socket, zmq.POLLIN)
            poller.register(sockets.sub_monitor, zmq.POLLIN)
            try:
                await _subscribed(source.pub)
                sockets.pending_live.update({1: b"one", 2: b"two"})
                sockets.pending_bytes = 6
                transport._replay_cursors["a"] = 0
                transport._retry_counts["a"] = 1
                transport._malformed_counts["a"] = 1
                await source.pub.send_multipart([b"topic", b"bad", b"payload"])
                assert await sockets.sub_socket.poll(3000, zmq.POLLIN)
                received = []

                with pytest.raises(_MalformedFrame, match="invalid sequence frame"):
                    await transport._merge_pending_live(sockets, poller, lambda payload, _: received.append(payload))

                assert received == [b"one"] and transport._replay_cursors["a"] == 1
                assert transport._malformed_counts["a"] == 0
                assert transport._retry_counts["a"] == 0
            finally:
                transport.stop()

    asyncio.run(run())


def test_replay_retirement_zmq_error_rebuilds_instead_of_stopping(monkeypatch):
    async def run():
        with _source() as source:
            resets, ready = asyncio.Queue(), asyncio.Queue()
            transport = _transport(
                source.endpoints, on_stream_reset=resets.put_nowait, on_stream_ready=ready.put_nowait
            )
            async with _subscription(transport) as (received, task):
                disable = zmq.asyncio.Socket.disable_monitor
                failed = False

                def fail_once(socket):
                    nonlocal failed
                    if socket.type == zmq.DEALER and not failed:
                        failed = True
                        raise zmq.ZMQError(zmq.EINVAL)
                    return disable(socket)

                await _next(resets)
                await _subscribed(source.pub)
                identity = await _request(source.router)
                old = transport._endpoint_sockets["a"]
                replay_socket, replay_monitor = old.replay_socket, old.replay_monitor
                monkeypatch.setattr(zmq.asyncio.Socket, "disable_monitor", fail_once)
                await _reply(source.router, identity, [])
                await _next(resets)
                assert failed and ready.empty() and replay_socket.closed and replay_monitor.closed
                await _subscribed(source.pub)
                await _respond(source.router, [(0, b"recovered")])
                await _next(ready)
                assert await _next(received) == ("a", b"recovered")
                assert old.closed and old.context.closed and not task.done()

    asyncio.run(run())


def _stream_observer():
    view = {"resident": {b"stale"}, "gated": False, "events": [], "changes": asyncio.Queue()}

    def unavailable(replica):
        assert replica == "a"
        view["resident"].clear()
        view["gated"] = False
        view["events"].append("unavailable")
        view["changes"].put_nowait("unavailable")

    def reset(replica):
        assert replica == "a"
        view["resident"].clear()
        view["gated"] = True
        view["events"].append("reset")
        view["changes"].put_nowait("reset")

    def ready(replica):
        assert replica == "a"
        view["gated"] = False
        view["events"].append("ready")
        view["changes"].put_nowait("ready")

    return view, {"on_stream_unavailable": unavailable, "on_stream_reset": reset, "on_stream_ready": ready}


def test_connect_failures_preserve_dispatch_estimates_until_handshake_rebuilds(monkeypatch):
    async def run():
        with _source(False) as source:
            source.close_socket("pub")
            view, callbacks = _stream_observer()
            transport = _transport(source.endpoints, max_retry_attempts=0, max_retry_delay=0, **callbacks)
            transport._CONNECTION_TIMEOUT = 0.02
            attempts = asyncio.Queue()
            connect = transport._connect_zmq_for

            async def counted_connect(replica):
                sockets = await connect(replica)
                attempts.put_nowait(replica)
                return sockets

            monkeypatch.setattr(transport, "_connect_zmq_for", counted_connect)
            async with _subscription(transport) as (received, task):
                assert await _next(view["changes"]) == "unavailable"
                assert not view["resident"] and not view["gated"]
                view["resident"].add(b"new dispatch")
                for _ in range(3):
                    await _next(attempts)
                assert view["events"] == ["unavailable"]
                assert view["resident"] == {b"new dispatch"} and not view["gated"]
                source.reopen("pub")
                await _subscribed(source.pub)
                assert await _next(view["changes"]) == "reset"
                assert await _next(view["changes"]) == "ready"
                assert not view["resident"]
                await source.publish(0, b"live")
                assert await _next(received) == ("a", b"live")

    asyncio.run(run())


def test_handshake_gates_and_clears_fallback_before_any_replay_handler():
    async def run():
        with _source() as source:
            view, callbacks = _stream_observer()
            received = asyncio.Queue()
            transport = _transport(source.endpoints, **callbacks)

            def handler(payload, replica):
                assert view["gated"], "replay must not run against an available fallback view"
                view["resident"].add(payload)
                view["events"].append("data")
                received.put_nowait((replica, payload))

            async with _subscription(transport, handler) as (_, task):
                assert await _next(view["changes"]) == "unavailable"
                await _subscribed(source.pub)
                identity = await _request(source.router)
                assert await _next(view["changes"]) == "reset"
                assert view["gated"] and not view["resident"]
                await _reply(source.router, identity, [(0, b"native replay")], end=False)
                assert await _next(received) == ("a", b"native replay")
                assert view["events"] == ["unavailable", "reset", "data"]
                await _reply(source.router, identity, [])
                assert await _next(view["changes"]) == "ready"
                assert view["resident"] == {b"native replay"} and not view["gated"]

    asyncio.run(run())


def test_recoverable_failure_closes_owned_session_before_opening_fallback(monkeypatch):
    async def run():
        with _source() as source:
            view, callbacks = _stream_observer()
            unavailable = callbacks["on_stream_unavailable"]
            old = None

            def assert_closed_then_unavailable(replica):
                if old is not None:
                    assert old.closed and old.context.closed
                    assert replica not in transport._replay_cursors
                unavailable(replica)

            callbacks["on_stream_unavailable"] = assert_closed_then_unavailable
            transport = _transport(source.endpoints, max_retry_delay=0, max_retry_attempts=0, **callbacks)
            transport._CONNECTION_TIMEOUT = 0.03
            attempts = asyncio.Queue()
            connect = transport._connect_zmq_for

            async def counted_connect(replica):
                sockets = await connect(replica)
                attempts.put_nowait(replica)
                return sockets

            monkeypatch.setattr(transport, "_connect_zmq_for", counted_connect)
            async with _subscription(transport) as (received, task):
                assert await _next(view["changes"]) == "unavailable"
                await _subscribed(source.pub)
                await _respond(source.router, [(0, b"native")])
                assert await _next(view["changes"]) == "reset"
                assert await _next(view["changes"]) == "ready"
                await _next(received)
                await _next(attempts)
                old = transport._endpoint_sockets["a"]
                source.close_socket("router")
                await source.publish(2, b"gap")
                assert await _next(view["changes"]) == "unavailable"
                view["resident"].add(b"dispatch during outage")
                await _next(attempts)
                await _next(attempts)
                assert view["events"] == ["unavailable", "reset", "ready", "unavailable"]
                assert view["resident"] == {b"dispatch during outage"} and not view["gated"]
                source.reopen("router")
                assert await _next(view["changes"]) == "reset"
                await _respond(source.router, [(0, b"rebuilt")])
                assert await _next(view["changes"]) == "ready"
                assert await _next(received) == ("a", b"rebuilt")

    asyncio.run(run())


def test_terminal_malformed_failure_gates_without_an_unavailable_window():
    async def run():
        with _source(False) as source:
            view, callbacks = _stream_observer()
            transport = _transport(source.endpoints, max_retry_attempts=1, **callbacks)
            async with _subscription(transport) as (_, task):
                assert await _next(view["changes"]) == "unavailable"
                for attempt in range(2):
                    await _subscribed(source.pub)
                    assert await _next(view["changes"]) == "reset"
                    assert await _next(view["changes"]) == "ready"
                    view["resident"].add(b"partial")
                    await source.pub.send_multipart([b"topic", b"bad", b"payload"])
                    assert await _next(view["changes"]) == ("unavailable" if attempt == 0 else "reset")
                await asyncio.wait_for(task, 3)
                assert view["gated"] and not view["resident"]
                assert view["events"] == ["unavailable", "reset", "ready", "unavailable", "reset", "ready", "reset"]

    asyncio.run(run())


@pytest.mark.parametrize(
    "fault, legacy",
    [
        pytest.param("consumer", False, id="consumer"),
        pytest.param("reset", False, id="reset"),
        pytest.param("ready", False, id="ready"),
        pytest.param("unavailable", False, id="unavailable"),
        pytest.param("reset", True, id="legacy-reset"),
        pytest.param("ready", True, id="legacy-ready"),
    ],
)
def test_fatal_callbacks_gate_without_reopening_fallback(fault, legacy):
    async def run():
        with _source(False) as source:
            view, callbacks = _stream_observer()

            def failing_callback(name, callback):
                def run(replica):
                    callback(replica)
                    if name == fault:
                        raise ValueError(f"{name} callback failed")

                return run

            for name in ("reset", "ready", "unavailable"):
                key = f"on_stream_{name}"
                callbacks[key] = failing_callback(name, callbacks[key])
            if legacy:
                callbacks.pop("on_stream_unavailable")
            transport = _transport(source.endpoints, **callbacks)

            def handler(payload, replica):
                view["resident"].add(payload)
                view["events"].append("data")
                raise ValueError("consumer failed after a partial write")

            async with _subscription(transport, handler) as (_, task):
                if fault == "consumer":
                    await _subscribed(source.pub)
                    for event in ("unavailable", "reset", "ready"):
                        assert await _next(view["changes"]) == event
                    await source.publish(0, b"partial")
                await asyncio.wait_for(task, 3)
                assert view["gated"] and not view["resident"]
                assert view["events"].count("unavailable") == (0 if legacy else 1)
                assert view["events"][-1] == "reset"
                assert not transport._endpoint_sockets and not transport._replay_cursors
                if legacy:
                    assert not transport._retry_counts

    asyncio.run(run())


@pytest.mark.parametrize("replacement", [False, True], ids=["own-task", "replacement-task"])
def test_cancelled_cleanup_gather_removes_only_its_own_task_records(monkeypatch, replacement):
    """Inject cancellation at cleanup gather; ordinary 3.12 cancellation may miss this path."""

    async def run():
        with _source(False) as source:
            ready = asyncio.Queue()
            transport = _transport(source.endpoints, on_stream_ready=ready.put_nowait)
            original_gather = asyncio.gather
            subscription = None
            successor = None
            gather_calls = 0

            def interrupted_cleanup(*tasks, **kwargs):
                nonlocal gather_calls, successor
                if asyncio.current_task() is subscription:
                    gather_calls += 1
                    if gather_calls == 2:
                        if replacement:
                            successor = asyncio.create_task(asyncio.Event().wait())
                            transport._sub_tasks["a"] = successor
                        cancelled = asyncio.get_running_loop().create_future()
                        cancelled.cancel()
                        return cancelled
                return original_gather(*tasks, **kwargs)

            monkeypatch.setattr(asyncio, "gather", interrupted_cleanup)
            subscription = asyncio.create_task(transport.subscribe(lambda payload, replica: None))
            try:
                await _subscribed(source.pub)
                assert await _next(ready) == "a"
                sockets = transport._endpoint_sockets["a"]
                endpoint = transport._sub_tasks["a"]
                subscription.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(subscription, 3)
                assert gather_calls == 2 and endpoint.done()
                assert sockets.closed and sockets.sub_socket.closed and sockets.sub_monitor.closed
                assert sockets.context.closed and not transport._endpoint_sockets
                assert transport._sub_tasks == ({"a": successor} if replacement else {})
            finally:
                transport.stop()
                if successor is not None:
                    successor.cancel()
                await original_gather(
                    subscription, *([successor] if successor is not None else []), return_exceptions=True
                )

    asyncio.run(run())
