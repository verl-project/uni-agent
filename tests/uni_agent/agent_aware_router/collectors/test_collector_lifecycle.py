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

from uni_agent.agent_aware_router.collectors import _loop_runner as loop_runner_module
from uni_agent.agent_aware_router.collectors.collector import Collector
from uni_agent.agent_aware_router.collectors.parse.base import Parser
from uni_agent.agent_aware_router.collectors.transport.base import Transport

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


class IdleTransport(Transport):
    def __init__(self):
        self.stop_calls = 0

    async def subscribe(self, handler):
        await asyncio.Event().wait()

    def stop(self):
        self.stop_calls += 1


class NoopParser(Parser):
    def parse(self, raw_data, replica_id):
        return None


@pytest.mark.parametrize("loop_running", [False, True], ids=["delayed-loop", "running-loop"])
def test_startup_timeout_cleans_loop_and_thread_before_restart(monkeypatch, loop_running):
    transport = IdleTransport()
    collector = Collector(transport, NoopParser())
    first_loop = collector._loop_runner.loop
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
            patch.setattr(
                loop_runner_module, "threading", SimpleNamespace(Event=TimedOutReadiness, Thread=StartupThread)
            )
            with pytest.raises(TimeoutError, match="startup timed out"):
                collector.start()
            assert not threads[0].is_alive()
            assert collector._loop_runner.thread is None and first_loop.is_closed()
            assert collector._loop_runner.subscription_future is None and collector._loop_runner.cleanup_future is None
            assert transport.stop_calls == 1
        collector.start()
        assert collector._loop_runner.loop is not first_loop and collector._loop_runner.loop.is_running()
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
            patch.setattr(loop_runner_module, "threading", SimpleNamespace(Event=TimedOutReadiness, Thread=Thread))
            patch.setattr(collector, "stop", fail_stop)
            with pytest.raises(TimeoutError, match="collector event loop startup timed out") as error:
                collector.start()
            assert error.value.__cause__ is cleanup_error
            assert collector._loop_runner.thread is not None and collector._loop_runner.thread.is_alive()
        collector.stop(timeout=1)
        assert collector._loop_runner.thread is None and collector._loop_runner.loop.is_closed()
    finally:
        collector.stop(timeout=1)


def test_stop_timeout_retains_pending_cleanup_until_retry():
    transport = IdleTransport()
    collector = Collector(transport, NoopParser())
    collector.start()
    thread = collector._loop_runner.thread
    entered, release = Event(), Event()

    def block_loop():
        entered.set()
        release.wait(3)

    collector._loop_runner.loop.call_soon_threadsafe(block_loop)
    try:
        assert entered.wait(1)
        with pytest.raises(TimeoutError, match="cleanup timed out"):
            collector.stop(timeout=0.01)
        pending = collector._loop_runner.cleanup_future
        assert pending is not None and not pending.done()
        assert collector._loop_runner.thread is thread and thread.is_alive()
        with pytest.raises(RuntimeError, match="stop the collector"):
            collector.start()
        with pytest.raises(TimeoutError, match="cleanup timed out"):
            collector.stop(timeout=0.01)
        assert collector._loop_runner.cleanup_future is pending
    finally:
        release.set()
        collector.stop(timeout=1)
    assert transport.stop_calls == 1
    assert not thread.is_alive() and collector._loop_runner.loop.is_closed()
    assert collector._loop_runner.thread is None and collector._loop_runner.cleanup_future is None


def test_stop_joins_thread_before_its_loop_has_started(monkeypatch):
    transport = IdleTransport()
    collector = Collector(transport, NoopParser())
    entered, release = Event(), Event()

    def delayed_loop():
        entered.set()
        release.wait(3)
        collector._loop_runner.loop.run_forever()

    thread = Thread(target=delayed_loop, daemon=True)
    collector._loop_runner.thread = thread
    thread.start()
    monkeypatch.setattr(transport, "stop", release.set)
    try:
        assert entered.wait(1)
        assert not collector._loop_runner.loop.is_running()
        collector.stop(timeout=1)
        assert not thread.is_alive() and collector._loop_runner.loop.is_closed()
    finally:
        release.set()
        collector.stop(timeout=1)


def test_start_rejects_duplicate_and_restarts_after_complete_cleanup():
    collector = Collector(IdleTransport(), NoopParser())
    try:
        collector.start()
        first_loop = collector._loop_runner.loop
        assert first_loop.is_running()
        with pytest.raises(RuntimeError, match="stop the collector"):
            collector.start()
        collector.stop(timeout=1)
        assert first_loop.is_closed()
        collector.start()
        assert collector._loop_runner.loop is not first_loop and collector._loop_runner.loop.is_running()
    finally:
        collector.stop(timeout=1)


def test_stop_timeout_finishes_in_background_without_another_stop():
    started, cleaning, cleaned = Event(), Event(), Event()
    release = asyncio.Event()

    class SlowCleanupTransport(IdleTransport):
        async def subscribe(self, handler):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                cleaned.set()

    transport = SlowCleanupTransport()
    collector = Collector(transport, NoopParser())
    collector.start()
    loop, thread = collector._loop_runner.loop, collector._loop_runner.thread
    try:
        assert started.wait(1)
        with pytest.raises(TimeoutError, match="cleanup timed out"):
            collector.stop(timeout=0.01)
        pending = collector._loop_runner.cleanup_future
        assert cleaning.wait(1) and pending is not None and not pending.done()
        assert thread.is_alive() and not cleaned.is_set()

        loop.call_soon_threadsafe(release.set)
        pending.result(timeout=1)
        thread.join(timeout=1)
        assert cleaned.is_set()
        assert not thread.is_alive() and loop.is_closed()

        collector.stop(timeout=1)
        collector.stop(timeout=1)
        assert transport.stop_calls == 1
        collector.start()
        assert collector._loop_runner.loop is not loop and collector._loop_runner.loop.is_running()
    finally:
        if thread.is_alive():
            loop.call_soon_threadsafe(release.set)
        collector.stop(timeout=1)


@pytest.mark.parametrize("cleanup_error", [ValueError("transport cleanup failed"), TimeoutError("transport timed out")])
def test_stop_error_closes_the_loop_and_preserves_the_cause(cleanup_error):
    started, cleaned = Event(), Event()

    class FailingStopTransport(IdleTransport):
        async def subscribe(self, handler):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        def stop(self):
            super().stop()
            if self.stop_calls == 1:
                raise cleanup_error

    transport = FailingStopTransport()
    collector = Collector(transport, NoopParser())
    collector.start()
    loop, thread = collector._loop_runner.loop, collector._loop_runner.thread
    try:
        assert started.wait(1)
        with pytest.raises(RuntimeError, match="cleanup did not finish") as error:
            collector.stop(timeout=1)
        assert type(error.value.__cause__) is type(cleanup_error)
        assert error.value.__cause__.args == cleanup_error.args
        thread.join(timeout=1)
        assert cleaned.is_set()
        assert not thread.is_alive() and loop.is_closed()

        assert collector._loop_runner.thread is None
        assert collector._loop_runner.cleanup_future is None
        collector.stop(timeout=1)
        assert transport.stop_calls == 1
        collector.start()
        assert collector._loop_runner.loop is not loop and collector._loop_runner.loop.is_running()
    finally:
        current_thread = collector._loop_runner.thread
        if current_thread is not None and current_thread.is_alive():
            collector.stop(timeout=1)


def test_cleanup_error_after_timeout_is_reported_by_late_stop():
    started, cleaning = Event(), Event()
    release = asyncio.Event()
    cleanup_error = ValueError("delayed transport cleanup failed")

    class DelayedFailureTransport(IdleTransport):
        async def subscribe(self, handler):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()

        def stop(self):
            super().stop()
            raise cleanup_error

    transport = DelayedFailureTransport()
    collector = Collector(transport, NoopParser())
    collector.start()
    loop, thread = collector._loop_runner.loop, collector._loop_runner.thread
    try:
        assert started.wait(1)
        with pytest.raises(TimeoutError, match="cleanup timed out"):
            collector.stop(timeout=0.01)
        pending = collector._loop_runner.cleanup_future
        assert cleaning.wait(1) and pending is not None
        loop.call_soon_threadsafe(release.set)
        with pytest.raises(ValueError) as background_error:
            pending.result(timeout=1)
        assert background_error.value is cleanup_error
        thread.join(timeout=1)
        assert not thread.is_alive() and loop.is_closed()

        with pytest.raises(RuntimeError, match="cleanup did not finish") as reported:
            collector.stop(timeout=1)
        assert reported.value.__cause__ is cleanup_error
        assert collector._loop_runner.cleanup_future is None
        collector.stop(timeout=1)
        assert transport.stop_calls == 1
    finally:
        if thread.is_alive():
            loop.call_soon_threadsafe(release.set)
            try:
                collector.stop(timeout=1)
            except RuntimeError:
                pass
