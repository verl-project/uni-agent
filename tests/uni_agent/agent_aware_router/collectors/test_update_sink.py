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

"""Transport/parse dispatch can be tested without constructing any store."""

import pytest

from uni_agent.agent_aware_router.collectors.collector import Collector
from uni_agent.agent_aware_router.collectors.parse import (
    InflightBlockUpdate,
    KVCacheUpdate,
    MetricsUpdate,
    Parser,
    StickyUpdate,
)
from uni_agent.agent_aware_router.collectors.parse.basic.inflight import InflightParser
from uni_agent.agent_aware_router.collectors.sink import InflightContext
from uni_agent.agent_aware_router.collectors.transport.base import Transport
from uni_agent.agent_aware_router.collectors.transport.callback import StatisticEvent
from uni_agent.agent_aware_router.types import MetricKey

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


class RecordingSink:
    def __init__(self):
        self.calls = []
        self.context = InflightContext(turn=3, booked_len=17)

    def apply_kv(self, update):
        self.calls.append(("kv", update))

    def apply_metrics(self, update):
        self.calls.append(("metrics", update))

    def apply_sticky(self, update):
        self.calls.append(("sticky", update))

    def apply_inflight_blocks(self, update):
        self.calls.append(("blocks", update))

    def prepare_inflight(self, event):
        self.calls.append(("context", event))
        return self.context


class FeedTransport(Transport):
    is_async = False

    def __init__(self, payloads):
        self.payloads = payloads
        self.stopped = False

    async def subscribe(self, handler):
        for payload in self.payloads:
            handler(payload, "node")

    def stop(self):
        self.stopped = True


class EchoParser(Parser):
    def parse(self, raw_data, node_id):
        assert node_id == "node"
        return raw_data


@pytest.fixture
def forbid_store(monkeypatch):
    from uni_agent.agent_aware_router.store.data_store import DataStore

    def fail(*args, **kwargs):
        pytest.fail("Collector constructed a DataStore despite an injected sink")

    monkeypatch.setattr(DataStore, "__init__", fail)


def run_collector(parser, sink, payloads):
    transport = FeedTransport(payloads)
    collector = Collector(transport, parser, sink)
    try:
        collector.start()
    finally:
        collector.stop()
        collector._loop.close()
    assert transport.stopped


def test_dispatch_preserves_single_mixed_and_empty_results(forbid_store):
    kv = KVCacheUpdate("node")
    metrics = MetricsUpdate("node", {MetricKey.INFLIGHT_COUNT: 1}, is_delta=True)
    sticky = StickyUpdate("put", "r", "node")
    block = InflightBlockUpdate("node", "r", "unpin")
    sink = RecordingSink()
    run_collector(EchoParser(), sink, [kv, None, [], [sticky, block, metrics], object()])
    assert sink.calls == [("kv", kv), ("sticky", sticky), ("blocks", block), ("metrics", metrics)]


def test_inflight_uses_injected_booking_before_ordered_dispatch(forbid_store):
    event = StatisticEvent("on_release", "r", "node")
    sink = RecordingSink()
    run_collector(InflightParser(), sink, [event])
    assert sink.calls == [
        ("context", event),
        ("blocks", InflightBlockUpdate("node", "r", "unpin")),
        (
            "metrics",
            MetricsUpdate(
                "node",
                {
                    MetricKey.INFLIGHT_COUNT: -1,
                    MetricKey.COMPLETED_COUNT: 1,
                    MetricKey.INFLIGHT_TURN_SUM: -3,
                    MetricKey.INFLIGHT_TOKENS: -17,
                },
                is_delta=True,
                request_id="r",
            ),
        ),
    ]


@pytest.mark.parametrize(
    "payload", [b"invalid", StatisticEvent("on_acquire"), StatisticEvent("on_servers_removed", replica_id="node")]
)
def test_ignored_inflight_payloads_do_not_request_or_apply_context(forbid_store, payload):
    sink = RecordingSink()
    run_collector(InflightParser(), sink, [payload])
    assert sink.calls == []


def test_failed_application_stops_dispatch_and_propagates(forbid_store):
    class FailingSink(RecordingSink):
        def apply_kv(self, update):
            raise RuntimeError("write failed")

    sink = FailingSink()
    with pytest.raises(RuntimeError, match="write failed"):
        run_collector(EchoParser(), sink, [[KVCacheUpdate("node"), StickyUpdate("put", "r", "node")]])
    assert sink.calls == []
