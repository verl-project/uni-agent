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

"""Collector resource ownership and its neutral input boundary."""

import asyncio
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from uni_agent.agent_aware_router.collectors import collector as collector_module
from uni_agent.agent_aware_router.collectors.collector import Collector
from uni_agent.agent_aware_router.collectors.parse.base import Parser
from uni_agent.agent_aware_router.collectors.transport.base import Transport, TransportMessage
from uni_agent.agent_aware_router.collectors.transport.callback import StatisticEvent

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


class IdleTransport(Transport[object]):
    def __init__(self):
        self.stop_calls = 0

    async def subscribe(self, handler):
        await asyncio.Event().wait()

    def stop(self):
        self.stop_calls += 1


class NoopParser(Parser[object]):
    def parse(self, message: TransportMessage[object]):
        return None


@pytest.mark.parametrize("loop_running", [False, True], ids=["delayed-loop", "running-loop"])
def test_startup_timeout_cleans_loop_and_thread_before_restart(monkeypatch, loop_running):
    transport = IdleTransport()
    collector = Collector(transport, NoopParser())
    first_loop = collector._loop
    entered, release = Event(), Event()
    threads = []
    stop = transport.stop

    class StartupThread(Thread):
        def run(self):
            entered.set()
            if not loop_running:
                release.wait(3)
            super().run()

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            threads.append(self)

    class TimedOutReadiness(Event):
        def wait(self, timeout=None):
            assert timeout == 5
            assert entered.wait(1)
            if loop_running:
                assert super().wait(1)
            return False

    def stop_and_release():
        stop()
        release.set()

    monkeypatch.setattr(transport, "stop", stop_and_release)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(collector_module, "threading", SimpleNamespace(Event=TimedOutReadiness, Thread=StartupThread))
            with pytest.raises(TimeoutError, match="startup timed out"):
                collector.start()
            assert not threads[0].is_alive()
            assert collector._loop_thread is None and first_loop.is_closed()
            assert collector._future is None and collector._stop_future is None
            assert transport.stop_calls == 1
        collector.start()
        assert collector._loop is not first_loop and collector._loop.is_running()
    finally:
        release.set()
        collector.stop(timeout=1)


@pytest.mark.parametrize("cleanup_error", [TimeoutError("thread did not stop"), RuntimeError("cleanup failed")])
def test_startup_timeout_keeps_cleanup_failure_as_its_cause(monkeypatch, cleanup_error):
    collector = Collector(IdleTransport(), NoopParser())

    class TimedOutReadiness(Event):
        def wait(self, timeout=None):
            assert super().wait(1)
            return False

    def fail_stop():
        raise cleanup_error

    try:
        with monkeypatch.context() as patch:
            patch.setattr(collector_module, "threading", SimpleNamespace(Event=TimedOutReadiness, Thread=Thread))
            patch.setattr(collector, "stop", fail_stop)
            with pytest.raises(TimeoutError, match="collector event loop startup timed out") as error:
                collector.start()
            assert error.value.__cause__ is cleanup_error
            assert collector._loop_thread is not None and collector._loop_thread.is_alive()
        collector.stop(timeout=1)
        assert collector._loop_thread is None and collector._loop.is_closed()
    finally:
        collector.stop(timeout=1)


def test_stop_timeout_retains_pending_cleanup_until_retry():
    transport = IdleTransport()
    collector = Collector(transport, NoopParser())
    collector.start()
    thread = collector._loop_thread
    entered, release = Event(), Event()

    def block_loop():
        entered.set()
        release.wait(3)

    collector._loop.call_soon_threadsafe(block_loop)
    try:
        assert entered.wait(1)
        with pytest.raises(TimeoutError, match="cleanup timed out"):
            collector.stop(timeout=0.01)
        pending = collector._stop_future
        assert pending is not None and not pending.done()
        assert collector._loop_thread is thread and thread.is_alive()
        with pytest.raises(RuntimeError, match="stop the collector"):
            collector.start()
        with pytest.raises(TimeoutError, match="cleanup timed out"):
            collector.stop(timeout=0.01)
        assert collector._stop_future is pending
    finally:
        release.set()
        collector.stop(timeout=1)
    assert transport.stop_calls == 1
    assert not thread.is_alive() and collector._loop.is_closed()
    assert collector._loop_thread is None and collector._stop_future is None


def test_stop_joins_thread_before_its_loop_has_started(monkeypatch):
    transport = IdleTransport()
    collector = Collector(transport, NoopParser())
    entered, release = Event(), Event()

    def delayed_loop():
        entered.set()
        release.wait(3)
        collector._loop.run_forever()

    thread = Thread(target=delayed_loop, daemon=True)
    collector._loop_thread = thread
    thread.start()
    monkeypatch.setattr(transport, "stop", release.set)
    try:
        assert entered.wait(1)
        assert not collector._loop.is_running()
        collector.stop(timeout=1)
        assert not thread.is_alive() and collector._loop.is_closed()
    finally:
        release.set()
        collector.stop(timeout=1)


def test_start_rejects_duplicate_and_restarts_after_complete_cleanup():
    collector = Collector(IdleTransport(), NoopParser())
    try:
        collector.start()
        first_loop = collector._loop
        assert first_loop.is_running()
        with pytest.raises(RuntimeError, match="stop the collector"):
            collector.start()
        collector.stop(timeout=1)
        assert first_loop.is_closed()
        collector.start()
        assert collector._loop is not first_loop and collector._loop.is_running()
    finally:
        collector.stop(timeout=1)


def test_collector_forwards_messages_with_their_source_and_protocol_metadata():
    messages = [
        TransportMessage(b"ordered", "replica-a", sequence=7),
        TransportMessage(b"", "replica-b", sequence=8, status="no_continuity"),
        TransportMessage("metrics", "replica-c"),
        TransportMessage(StatisticEvent("on_acquire"), ""),
    ]
    received = []

    class PayloadTransport(IdleTransport):
        is_async = False

        async def subscribe(self, handler):
            for message in messages:
                handler(message)

    class RecordingParser(NoopParser):
        def parse(self, message: TransportMessage[object]):
            received.append(message)

    collector = Collector(PayloadTransport(), RecordingParser())
    try:
        collector.start()
        assert len(received) == len(messages)
        assert all(actual is expected for actual, expected in zip(received, messages, strict=True))
    finally:
        collector.stop(timeout=1)
