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

"""Unit tests for vLLM HTTP metrics collection with a fake HTTP transport.

Test flow:
1. Feed pre-canned Prometheus metrics text via FakeHTTPTransport.
2. Create a Collector(FakeHTTPTransport, VLLMMetricsParser).
3. Call start() to begin metrics polling; the collector writes to the store.
4. Verify that expected metrics exist via DataStore.

No real vLLM service is required.
"""

from __future__ import annotations

import time

import pytest
from conftest import NODE_ID, VLLM_METRICS_TEXT, FakeHTTPTransport

from uni_agent.agent_aware_router.collectors.collector import Collector
from uni_agent.agent_aware_router.collectors.parse.vllm.metrics import VLLMMetricsParser
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.types import MetricKey

pytestmark = [pytest.mark.level0, pytest.mark.cpu]

POLL_WAIT = 0.3


def _make_collector():
    return Collector(FakeHTTPTransport(VLLM_METRICS_TEXT, interval=0.05), VLLMMetricsParser())


def _collect() -> DataStore:
    """Start the collector, let it poll a few cycles, stop, return the store."""
    store = DataStore()
    collector = _make_collector()

    collector.start()
    time.sleep(POLL_WAIT)
    collector.stop()

    return store


class TestVLLMMetricsCollector:
    """Unit tests: vLLM HTTP metrics collector with an injected metrics payload."""

    def test_start_and_metrics_exist(self):
        """
        Feature: Collector writes metrics to the store after start()
        Expectation:
            DataStore contains NODE_ID after one polling cycle.
            kv_cache_usage_perc → float, num_requests_running/waiting → int.
        """
        store = _collect()

        assert NODE_ID in store.get_metric_node_ids(), (
            f"Expected node_id '{NODE_ID}' in store, got {store.get_metric_node_ids()}"
        )
        assert isinstance(store.get_metric(NODE_ID, MetricKey.KV_CACHE_USAGE_PERC), float)
        assert isinstance(store.get_metric(NODE_ID, MetricKey.NUM_REQUESTS_RUNNING), int)
        assert isinstance(store.get_metric(NODE_ID, MetricKey.NUM_REQUESTS_WAITING), int)

    def test_metrics_values_are_sane(self):
        """
        Feature: Collected metric values are within reasonable bounds
        Expectation:
            kv_cache_usage_perc >= 0.0
            num_requests_running >= 0
            num_requests_waiting >= 0
        """
        store = _collect()

        assert store.get_metric(NODE_ID, MetricKey.KV_CACHE_USAGE_PERC) >= 0.0
        assert store.get_metric(NODE_ID, MetricKey.NUM_REQUESTS_RUNNING) >= 0
        assert store.get_metric(NODE_ID, MetricKey.NUM_REQUESTS_WAITING) >= 0

    def test_store_get_node_dict(self):
        """
        Feature: DataStore.get_metrics(node_id) returns the full node metrics dict
        Expectation:
            Dict contains kv_cache_usage_perc, num_requests_running, num_requests_waiting.
        """
        store = _collect()

        node_metrics = store.get_metrics(NODE_ID)
        assert isinstance(node_metrics, dict)
        assert MetricKey.KV_CACHE_USAGE_PERC in node_metrics
        assert MetricKey.NUM_REQUESTS_RUNNING in node_metrics
        assert MetricKey.NUM_REQUESTS_WAITING in node_metrics

    def test_multiple_poll_cycles_refresh(self):
        """
        Feature: Multiple polling cycles refresh the store with updated values
        Expectation:
            After several polling cycles the store contains data and values are reasonable.
        """
        store = _collect()

        assert len(store.get_metrics(NODE_ID)) > 0, "Store should have metrics after multiple poll cycles"
