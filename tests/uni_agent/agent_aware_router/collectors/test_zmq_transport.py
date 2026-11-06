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

"""ZMQ framing, ordered network facts, and replay without consumer policy."""

import asyncio
import logging
from contextlib import contextmanager

import pytest
import zmq
import zmq.asyncio

from uni_agent.agent_aware_router.collectors.transport.base import TransportMessage
from uni_agent.agent_aware_router.collectors.transport.zmq import ZMQTransport

pytestmark = [pytest.mark.cpu, pytest.mark.level0]
END_SEQ = (-1).to_bytes(8, "big", signed=True)


@contextmanager
def _source(replay=True):
    context = zmq.asyncio.Context()
    pub = context.socket(zmq.XPUB)
    pub.setsockopt(zmq.XPUB_VERBOSE, 1)
    pub_port = pub.bind_to_random_port("tcp://127.0.0.1")
    router = context.socket(zmq.ROUTER) if replay else None
    replay_address = f"127.0.0.1:{router.bind_to_random_port('tcp://127.0.0.1')}" if router else ""
    try:
        yield pub, router, {"a": [f"127.0.0.1:{pub_port}", replay_address, "zmq", "topic"]}
    finally:
        pub.close(0)
        if router is not None:
            router.close(0)
        context.term()


async def _subscribed(pub):
    while await asyncio.wait_for(pub.recv(), 3) != b"\x01topic":
        pass


async def _request(router, expected):
    identity, delimiter, start = await asyncio.wait_for(router.recv_multipart(), 3)
    assert delimiter == b"" and int.from_bytes(start, "big") == expected
    return identity


async def _respond(router, expected, batches):
    identity = await _request(router, expected)
    for sequence, payload in batches:
        await router.send_multipart([identity, b"", sequence.to_bytes(8, "big"), payload])
    await router.send_multipart([identity, b"", END_SEQ, b""])


def _transport(endpoints, **kwargs):
    return ZMQTransport(endpoints, base_retry_delay=0, **kwargs)


def test_all_sequences_are_delivered_and_only_replay_cursor_is_monotonic():
    transport = _transport({"a": ["127.0.0.1:1", "", "zmq", "topic"]})
    received = []
    for sequence, payload in [(40, b"first"), (40, b"duplicate"), (39, b"older"), (43, b"later")]:
        transport._deliver("a", sequence, payload, lambda message, node: received.append((node, message)))
    assert received == [
        ("a", TransportMessage(b"first", 40)),
        ("a", TransportMessage(b"duplicate", 40)),
        ("a", TransportMessage(b"older", 39)),
        ("a", TransportMessage(b"later", 43)),
    ]
    assert transport._replay_cursors == {"a": 43}


def test_dealer_drains_multiple_replies_and_reconnect_resumes_without_dropping_duplicates():
    async def run():
        with _source() as (_, router, endpoints):
            transport = _transport(endpoints)
            received = []
            callback = lambda message, node: received.append(message)
            addresses = endpoints["a"]
            await transport._connect_zmq_for("a", "tcp://" + addresses[0], "tcp://" + addresses[1])
            try:
                responder = asyncio.create_task(_respond(router, 0, [(5, b"five"), (6, b"six"), (7, b"seven")]))
                assert await asyncio.wait_for(transport._replay_historical_data_for("a", callback), 3)
                await responder
                transport._close_zmq_sockets_for("a")
                await transport._connect_zmq_for("a", "tcp://" + addresses[0], "tcp://" + addresses[1])
                responder = asyncio.create_task(_respond(router, 8, [(7, b"duplicate"), (8, b"eight")]))
                assert await asyncio.wait_for(transport._replay_historical_data_for("a", callback), 3)
                await responder
                assert received == [
                    TransportMessage(b"five", 5),
                    TransportMessage(b"six", 6),
                    TransportMessage(b"seven", 7),
                    TransportMessage(b"duplicate", 7),
                    TransportMessage(b"eight", 8),
                ]
                assert transport._replay_cursors == {"a": 8}
            finally:
                transport.stop()

    asyncio.run(run())


@pytest.mark.parametrize("cursor", [0, 20])
@pytest.mark.parametrize("duplicate", [False, True], ids=["new-frames", "duplicate-first"])
def test_replay_gap_is_reported_before_the_first_new_payload(cursor, duplicate):
    async def run():
        with _source() as (_, router, endpoints):
            transport = _transport(endpoints)
            transport._replay_cursors["a"] = cursor
            addresses = endpoints["a"]
            await transport._connect_zmq_for("a", "tcp://" + addresses[0], "tcp://" + addresses[1])
            received = []
            first = cursor + 5
            batches = [(first, b"first retained"), (first + 1, b"next retained")]
            expected = [
                TransportMessage(b"", status="no_continuity"),
                TransportMessage(b"first retained", first),
                TransportMessage(b"next retained", first + 1),
            ]
            if duplicate:
                batches.insert(0, (cursor, b"duplicate"))
                expected.insert(0, TransportMessage(b"duplicate", cursor))
            try:
                responder = asyncio.create_task(_respond(router, cursor + 1, batches))
                assert await asyncio.wait_for(
                    transport._replay_historical_data_for("a", lambda message, node: received.append(message)), 3
                )
                await responder
                assert received == expected
                assert transport._replay_cursors == {"a": first + 1}
            finally:
                transport.stop()

    asyncio.run(run())


@pytest.mark.parametrize("replay", [False, True], ids=["subscription-only", "with-replay"])
def test_new_stream_precedes_nonzero_data_and_live_jump_does_not_trigger_replay(replay):
    async def run():
        with _source(replay) as (pub, router, endpoints):
            transport = _transport(endpoints)
            received = asyncio.Queue()
            task = asyncio.create_task(transport.subscribe(lambda message, node: received.put_nowait(message)))
            try:
                await _subscribed(pub)
                assert await asyncio.wait_for(received.get(), 3) == TransportMessage(b"", status="new_stream")
                if replay:
                    await _respond(router, 0, [(40, b"first")])
                else:
                    await pub.send_multipart([b"topic", (40).to_bytes(8, "big"), b"first"])
                assert await asyncio.wait_for(received.get(), 3) == TransportMessage(b"first", 40)
                await pub.send_multipart([b"topic", (43).to_bytes(8, "big"), b"later"])
                # No reply is served: an automatic replay on the jump would block delivery.
                assert await asyncio.wait_for(received.get(), 3) == TransportMessage(b"later", 43)
                assert transport._replay_cursors == {"a": 43}
            finally:
                transport.stop()
                await task

    asyncio.run(run())


@pytest.mark.parametrize("replay", [False, True], ids=["subscription-only", "with-replay"])
def test_reconnect_reports_only_network_facts_and_keeps_existing_stream(replay):
    async def run():
        with _source(replay) as (pub, router, endpoints):
            transport = _transport(endpoints)
            received = asyncio.Queue()
            task = asyncio.create_task(transport.subscribe(lambda message, node: received.put_nowait(message)))
            try:
                await _subscribed(pub)
                assert await asyncio.wait_for(received.get(), 3) == TransportMessage(b"", status="new_stream")
                if replay:
                    await _respond(router, 0, [(40, b"first")])
                else:
                    await pub.send_multipart([b"topic", (40).to_bytes(8, "big"), b"first"])
                assert await asyncio.wait_for(received.get(), 3) == TransportMessage(b"first", 40)
                await pub.send_multipart([b"topic", b"bad-sequence", b"invalid"])
                if replay:
                    await _respond(router, 41, [(42, b"replayed")])
                    # Report the gap before data; the consumer owns its cache policy.
                    assert await asyncio.wait_for(received.get(), 3) == TransportMessage(b"", status="no_continuity")
                    assert await asyncio.wait_for(received.get(), 3) == TransportMessage(b"replayed", 42)
                else:
                    assert await asyncio.wait_for(received.get(), 3) == TransportMessage(b"", status="no_continuity")
                await _subscribed(pub)
                await pub.send_multipart([b"topic", (43).to_bytes(8, "big"), b"live"])
                assert await asyncio.wait_for(received.get(), 3) == TransportMessage(b"live", 43)
            finally:
                transport.stop()
                await task

    asyncio.run(run())


def test_replay_timeout_retires_identity_and_excludes_late_replies(monkeypatch):
    async def run():
        with _source() as (_, router, endpoints):
            transport = _transport(endpoints)
            addresses = endpoints["a"]
            await transport._connect_zmq_for("a", "tcp://" + addresses[0], "tcp://" + addresses[1])
            original = transport._endpoint_sockets["a"].replay_socket
            received = []
            callback = lambda message, node: received.append(message)
            wait_for = asyncio.wait_for
            monkeypatch.setattr(
                asyncio, "wait_for", lambda value, timeout: wait_for(value, 0.05 if timeout == 5 else timeout)
            )
            try:
                replay = asyncio.create_task(transport._replay_historical_data_for("a", callback))
                old_identity = await _request(router, 0)
                assert not await asyncio.wait_for(replay, 3)
                assert original.closed and transport._endpoint_sockets["a"].replay_socket is not original
                assert received == [TransportMessage(b"", status="no_continuity")]
                monkeypatch.setattr(asyncio, "wait_for", wait_for)
                replay = asyncio.create_task(transport._replay_historical_data_for("a", callback))
                new_identity = await _request(router, 0)
                assert new_identity != old_identity
                await router.send_multipart([old_identity, b"", (99).to_bytes(8, "big"), b"late"])
                await router.send_multipart([old_identity, b"", END_SEQ, b""])
                await router.send_multipart([new_identity, b"", (0).to_bytes(8, "big"), b"fresh"])
                await router.send_multipart([new_identity, b"", END_SEQ, b""])
                assert await asyncio.wait_for(replay, 3)
                assert received == [TransportMessage(b"", status="no_continuity"), TransportMessage(b"fresh", 0)]
            finally:
                transport.stop()

    asyncio.run(run())


@pytest.mark.parametrize(
    "bad_reply",
    [
        [b"short", b"payload"],
        [(0).to_bytes(8, "big"), b""],
        [END_SEQ, b"unexpected"],
        [b""],
    ],
)
def test_malformed_replay_reports_no_continuity(bad_reply):
    async def run():
        with _source() as (_, router, endpoints):
            transport = _transport(endpoints)
            addresses = endpoints["a"]
            await transport._connect_zmq_for("a", "tcp://" + addresses[0], "tcp://" + addresses[1])
            received = []
            try:
                replay = asyncio.create_task(
                    transport._replay_historical_data_for("a", lambda message, node: received.append(message))
                )
                identity = await _request(router, 0)
                await router.send_multipart([identity, b"", *bad_reply])
                assert not await asyncio.wait_for(replay, 3)
                assert received == [TransportMessage(b"", status="no_continuity")]
                assert not transport._replay_cursors
            finally:
                transport.stop()

    asyncio.run(run())


def test_connection_exhaustion_announces_unreachable_after_new_stream(monkeypatch):
    async def run():
        transport = _transport({"a": ["127.0.0.1:1", "", "zmq", "topic"]}, max_retry_attempts=2)
        attempts = []

        async def unavailable(*args):
            attempts.append(args)
            return False

        monkeypatch.setattr(transport, "_connect_zmq_for", unavailable)
        received = []
        await transport.subscribe(lambda message, node: received.append(message))
        assert len(attempts) == 3
        assert received == [TransportMessage(b"", status="new_stream"), TransportMessage(b"", status="unreachable")]

    asyncio.run(run())


def test_bad_subscription_frames_exhaust_retries_despite_successful_socket_connect():
    async def run():
        with _source(False) as (pub, _, endpoints):
            transport = _transport(endpoints, max_retry_attempts=2)
            received = asyncio.Queue()
            task = asyncio.create_task(transport.subscribe(lambda message, node: received.put_nowait(message)))
            try:
                await _subscribed(pub)
                assert (await asyncio.wait_for(received.get(), 3)).status == "new_stream"
                for attempt in range(3):
                    await pub.send_multipart([b"topic", b"bad", b"payload"])
                    assert (await asyncio.wait_for(received.get(), 3)).status == "no_continuity"
                    if attempt < 2:
                        await _subscribed(pub)
                assert (await asyncio.wait_for(received.get(), 3)).status == "unreachable"
                await asyncio.wait_for(task, 3)
            finally:
                transport.stop()
                await task

    asyncio.run(run())


def test_valid_subscription_data_restores_retry_budget():
    async def run():
        with _source(False) as (pub, _, endpoints):
            transport = _transport(endpoints, max_retry_attempts=1)
            received = asyncio.Queue()
            task = asyncio.create_task(transport.subscribe(lambda message, node: received.put_nowait(message)))
            try:
                await _subscribed(pub)
                assert (await asyncio.wait_for(received.get(), 3)).status == "new_stream"
                for sequence in (5, 6):
                    await pub.send_multipart([b"topic", b"bad", b"payload"])
                    assert (await asyncio.wait_for(received.get(), 3)).status == "no_continuity"
                    await _subscribed(pub)
                    await pub.send_multipart([b"topic", sequence.to_bytes(8, "big"), b"valid"])
                    assert await asyncio.wait_for(received.get(), 3) == TransportMessage(b"valid", sequence)
            finally:
                transport.stop()
                await task

    asyncio.run(run())


def test_restart_restores_exhausted_retry_budget_without_resetting_replay_cursor(monkeypatch):
    async def run():
        with _source(False) as (pub, _, endpoints):
            transport = _transport(endpoints, max_retry_attempts=1)
            received = asyncio.Queue()
            handler = lambda message, node: received.put_nowait(message)
            first = asyncio.create_task(transport.subscribe(handler))
            restarted = None
            try:
                await _subscribed(pub)
                assert (await asyncio.wait_for(received.get(), 3)).status == "new_stream"
                await pub.send_multipart([b"topic", (40).to_bytes(8, "big"), b"first session"])
                assert await asyncio.wait_for(received.get(), 3) == TransportMessage(b"first session", 40)
                for attempt in range(2):
                    await pub.send_multipart([b"topic", b"bad", b"payload"])
                    assert (await asyncio.wait_for(received.get(), 3)).status == "no_continuity"
                    if attempt == 0:
                        await _subscribed(pub)
                assert (await asyncio.wait_for(received.get(), 3)).status == "unreachable"
                await asyncio.wait_for(first, 3)
                assert transport._retry_counts == {"a": 1}
                transport.stop()
                assert transport._replay_cursors == {"a": 40}

                connect = transport._connect_zmq_for
                attempts = []

                async def fail_first_connect(*args):
                    attempts.append(args)
                    return False if len(attempts) == 1 else await connect(*args)

                monkeypatch.setattr(transport, "_connect_zmq_for", fail_first_connect)
                restarted = asyncio.create_task(transport.subscribe(handler))
                await _subscribed(pub)
                assert len(attempts) == 2
                await pub.send_multipart([b"topic", (41).to_bytes(8, "big"), b"second session"])
                # No new_stream or unreachable fact may precede this resumed data.
                assert await asyncio.wait_for(received.get(), 3) == TransportMessage(b"second session", 41)
                assert transport._replay_cursors == {"a": 41}
            finally:
                transport.stop()
                await first
                if restarted is not None:
                    await restarted

    asyncio.run(run())


def test_consumer_error_is_visible_and_does_not_reconnect(caplog, monkeypatch):
    async def run():
        with _source(False) as (bad_pub, _, bad_source), _source(False) as (good_pub, _, good_source):
            endpoints = {"a": bad_source["a"], "b": good_source["a"]}
            transport = _transport(endpoints)
            connect = transport._connect_zmq_for
            connections = {"a": 0, "b": 0}
            received = asyncio.Queue()

            async def counted_connect(node_id, *args):
                connections[node_id] += 1
                return await connect(node_id, *args)

            def handler(message, node):
                if node == "a" and message.status == "ok":
                    raise ValueError("consumer rejected the message")
                received.put_nowait((node, message))

            monkeypatch.setattr(transport, "_connect_zmq_for", counted_connect)
            task = asyncio.create_task(transport.subscribe(handler))
            try:
                await _subscribed(bad_pub)
                await _subscribed(good_pub)
                initial = dict([await asyncio.wait_for(received.get(), 3) for _ in range(2)])
                assert initial == {
                    "a": TransportMessage(b"", status="new_stream"),
                    "b": TransportMessage(b"", status="new_stream"),
                }
                sources = dict(transport._endpoint_sockets)
                await bad_pub.send_multipart([b"topic", (0).to_bytes(8, "big"), b"rejected"])
                with pytest.raises(ValueError, match="consumer rejected the message"):
                    await asyncio.wait_for(transport._sub_tasks["a"], 3)
                assert "a" not in transport._endpoint_sockets and sources["a"].closed
                assert not task.done() and not sources["b"].closed
                await good_pub.send_multipart([b"topic", (0).to_bytes(8, "big"), b"still alive"])
                assert await asyncio.wait_for(received.get(), 3) == ("b", TransportMessage(b"still alive", 0))
                assert connections == {"a": 1, "b": 1}
                assert not task.done()
            finally:
                transport.stop()
                await task
            assert not transport._endpoint_sockets and not transport._sub_tasks
            assert all(
                source.closed and source.sub_socket.closed and source.context.closed for source in sources.values()
            )

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
    assert "consumer rejected the message" in str(records[0].exc_info[1])
