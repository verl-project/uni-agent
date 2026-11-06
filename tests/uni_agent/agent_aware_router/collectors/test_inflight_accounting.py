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

"""Exercise bookkeeping through the real callback/collector/application path."""

import logging
import struct
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from uni_agent.agent_aware_router.collectors.collector import Collector
from uni_agent.agent_aware_router.collectors.parse import KVCacheUpdate
from uni_agent.agent_aware_router.collectors.parse.basic.inflight import InflightParser
from uni_agent.agent_aware_router.collectors.transport.callback import CallbackTransport
from uni_agent.agent_aware_router.collectors.updater import default_updater
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.store.per_request_store import PerRequestStore
from uni_agent.agent_aware_router.types import Layer, MetricKey

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


class Callbacks:
    def __init__(self):
        self.callbacks = {}

    def register_call_back(self, event, fn):
        self.callbacks[event] = fn

    def un_register_call_back(self, event, fn):
        del self.callbacks[event]

    def acquire(self, rid="r", replica="s0", prompt=(1, 2, 3, 4)):
        self.callbacks["on_acquire"](rid, replica, list(prompt))

    def release(self, rid="r", replica="s0"):
        self.callbacks["on_release"](replica, rid)


@pytest.fixture
def running(monkeypatch):
    monkeypatch.setattr(PerRequestStore, "_instance", PerRequestStore(max_size=2))
    store = DataStore()
    callbacks = Callbacks()
    collector = Collector(CallbackTransport(callbacks), InflightParser())
    collector.start()
    try:
        yield callbacks, store, collector
    finally:
        collector.stop()
        collector._loop.close()
        assert callbacks.callbacks == {}


def assert_idle(store, replica="s0"):
    for key in (MetricKey.INFLIGHT_COUNT, MetricKey.INFLIGHT_TOKENS, MetricKey.INFLIGHT_TURN_SUM):
        assert store.get_metric(replica, key) == 0, (replica, key, store.get_metrics(replica))
    assert store.inflight_block_count(replica) == 0


@pytest.mark.parametrize("existing", [False, True])
def test_parse_failure_does_not_book_or_replace_dispatch(running, existing):
    callbacks, store, _ = running
    store.set_block_size(2)
    if existing:
        callbacks.acquire()
    before = store.get_metrics("s0")
    marker = store.get_marker("r")
    turn = store.requests.get_turn("r")
    length = store.requests.get_prompt_length("r")
    with pytest.raises(struct.error):
        callbacks.acquire(prompt=(-1, 2))
    assert store.get_metrics("s0") == before
    assert store.get_marker("r") == marker
    assert store.requests.get_turn("r") == turn
    assert store.requests.get_prompt_length("r") == length
    callbacks.release()
    assert_idle(store)
    assert store.get_metric("s0", MetricKey.COMPLETED_COUNT) == int(existing)


@pytest.mark.parametrize("block_size,prompt", [(None, (1, 2, 3, 4)), (2, (1,)), (2, ()), (2, (1, 2, 3, 4))])
@pytest.mark.parametrize("rid", ["r", ""])
def test_duplicate_release_is_noop_with_or_without_pinned_blocks(running, block_size, prompt, rid):
    callbacks, store, _ = running
    if block_size:
        store.set_block_size(block_size)
    callbacks.acquire(rid, prompt=prompt)
    callbacks.release(rid)
    first = store.get_metrics("s0")
    callbacks.release(rid)
    assert store.get_metrics("s0") == first
    assert store.get_metric("s0", MetricKey.COMPLETED_COUNT) == 1
    assert_idle(store)
    # A legitimate next dispatch with the same id can still complete.
    callbacks.acquire(rid, prompt=prompt)
    callbacks.release(rid)
    assert_idle(store)
    assert store.get_metric("s0", MetricKey.COMPLETED_COUNT) == 2
    assert store.requests.get_turn(rid) == 2


@pytest.mark.parametrize("blocks", [False, True])
def test_reacquire_unwinds_original_replica_and_ignores_its_late_release(running, blocks):
    callbacks, store, _ = running
    if blocks:
        store.set_block_size(2)
    callbacks.acquire()
    callbacks.acquire(replica="s1", prompt=(5, 6))
    assert_idle(store, "s0")
    assert store.get_metric("s1", MetricKey.INFLIGHT_COUNT) == 1
    assert store.get_metric("s1", MetricKey.INFLIGHT_TOKENS) == 2
    assert store.get_metric("s1", MetricKey.INFLIGHT_TURN_SUM) == 2
    callbacks.release(replica="s0")
    assert store.get_metric("s1", MetricKey.INFLIGHT_COUNT) == 1
    callbacks.release(replica="s1")
    assert_idle(store, "s1")
    assert store.get_metric("s0", MetricKey.COMPLETED_COUNT) == 0
    assert store.get_metric("s1", MetricKey.COMPLETED_COUNT) == 1


@pytest.mark.parametrize("blocks", [False, True])
@pytest.mark.parametrize("reacquire", [False, True])
def test_active_booking_survives_real_request_lru_eviction(running, blocks, reacquire):
    callbacks, store, _ = running
    if blocks:
        store.set_block_size(2)
    callbacks.acquire()
    store.requests.put_sticky_binding("other1", "s0")
    store.requests.put_sticky_binding("other2", "s0")
    assert store.requests.get_turn("r") == store.requests.get_prompt_length("r") == 0
    if reacquire:
        callbacks.acquire()
        assert store.get_metric("s0", MetricKey.INFLIGHT_TURN_SUM) == 2
    callbacks.release()
    assert_idle(store)
    callbacks.release()
    assert_idle(store)


def test_overlapping_acquires_are_serialized_through_booking_commit(running, monkeypatch):
    callbacks, store, collector = running
    store.set_block_size(2)
    entered = threading.Event()
    finish = threading.Event()
    second_started = threading.Event()
    second_prepared = threading.Event()
    original = collector._sink.prepare_inflight
    calls = 0

    def pause_after_context(event):
        nonlocal calls
        context = original(event)
        calls += 1
        if calls == 1:
            entered.set()
            assert finish.wait(5)
        else:
            second_prepared.set()
        return context

    monkeypatch.setattr(collector._sink, "prepare_inflight", pause_after_context)

    def second():
        second_started.set()
        callbacks.acquire()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(callbacks.acquire)
        try:
            assert entered.wait(5)
            other = pool.submit(second)
            assert second_started.wait(5)
            assert not second_prepared.wait(0.1)
        finally:
            finish.set()
        first.result(timeout=5)
        other.result(timeout=5)
    assert store.get_metric("s0", MetricKey.INFLIGHT_COUNT) == 1
    assert store.get_metric("s0", MetricKey.INFLIGHT_TOKENS) == 4
    assert store.get_metric("s0", MetricKey.INFLIGHT_TURN_SUM) == 2
    callbacks.release()
    assert_idle(store)


def test_recorded_blocks_are_logged_by_inflight_observer_not_kv_observer(running, caplog, monkeypatch):
    from uni_agent.agent_aware_router.insight import collector as observer_module

    callbacks, store, _ = running
    store.set_block_size(2)
    monkeypatch.setattr(observer_module, "_KV_EVENT_LOG_INTERVAL_S", 1)
    caplog.set_level(logging.DEBUG, logger=observer_module.__name__)
    callbacks.acquire()
    # Production constructs a separate updater/observer for KV events.
    kv_updater = default_updater()
    kv_updater.apply_kv(KVCacheUpdate("s0", add_blocks={Layer.GPU: ["engine"]}))
    assert any("router-dispatch blocks: router_recorded=2" in r.message for r in caplog.records)
    kv_logs = [r.message for r in caplog.records if "kv-events tally" in r.message]
    assert len(kv_logs) == 1
    assert "router_recorded" not in kv_logs[0]


def test_reacquire_refreshes_old_replica_observations_without_counting_completion(running, monkeypatch):
    from uni_agent.agent_aware_router.insight import emitter

    callbacks, _, _ = running
    writes = []
    monkeypatch.setattr(emitter, "on_write", writes.append)
    callbacks.acquire()
    writes.clear()
    callbacks.acquire(replica="s1")
    assert [event.node for event in writes] == ["s0", "s1"]
    old, new = writes
    assert old.new_values[MetricKey.INFLIGHT_TOKENS] == 0
    assert old.turn_sum == old.inflight_count == 0
    assert MetricKey.COMPLETED_COUNT not in old.deltas
    assert new.new_values[MetricKey.INFLIGHT_TOKENS] == 4
    assert new.turn_sum == 2 and new.inflight_count == 1
