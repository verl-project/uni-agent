"""Tests for vLLM HTTP metrics collection with real vLLM service.

Test flow:
1. Launch a real vLLM model service (Qwen3-4B).
2. Create a Collector(HTTPTransport, VLLMMetricsDecoder) via BUILTIN_REGISTRY.
3. Call start() to begin metrics polling; the decoder writes to MetricsStore.
4. Verify that expected metrics exist in the store.
"""

from __future__ import annotations

import time

import pytest
from conftest import NODE_ID

from uni_agent.llm_router.collectors.registry import BUILTIN_REGISTRY
from uni_agent.llm_router.metric_spec import MetricKey
from uni_agent.llm_router.store.metrics_store import MetricsStore

pytestmark = [pytest.mark.st, pytest.mark.gpu]


POLL_INTERVAL = 2.0
HTTP_TIMEOUT = 10.0


def _make_collector():
    return BUILTIN_REGISTRY.get_collector(
        "vllm_metrics",
        endpoints={NODE_ID: NODE_ID},
        interval=POLL_INTERVAL,
        http_timeout=HTTP_TIMEOUT,
    )


class TestVLLMMetricsCollectorWithRealService:
    """Integration tests: vLLM HTTP metrics collector against a live vLLM server."""

    def test_start_and_metrics_exist(self, vllm_service):
        """
        Feature: Collector writes real metrics to MetricsStore after start()
        Expectation:
            MetricsStore contains NODE_ID after one polling cycle.
            kv_cache_usage_perc → float, num_requests_running/waiting → int.
        """
        store = MetricsStore.default()
        collector = _make_collector()

        collector.start()
        time.sleep(POLL_INTERVAL + 3.0)
        collector.stop()

        assert NODE_ID in store.all_ids(), f"Expected node_id '{NODE_ID}' in store, got {store.all_ids()}"
        assert isinstance(store.get(NODE_ID, MetricKey.KV_CACHE_USAGE_PERC), float)
        assert isinstance(store.get(NODE_ID, MetricKey.NUM_REQUESTS_RUNNING), int)
        assert isinstance(store.get(NODE_ID, MetricKey.NUM_REQUESTS_WAITING), int)

    def test_metrics_values_are_sane(self, vllm_service):
        """
        Feature: Collected metric values are within reasonable bounds
        Expectation:
            kv_cache_usage_perc >= 0.0
            num_requests_running >= 0
            num_requests_waiting >= 0
        """
        store = MetricsStore.default()
        collector = _make_collector()

        collector.start()
        time.sleep(POLL_INTERVAL + 3.0)
        collector.stop()

        assert store.get(NODE_ID, MetricKey.KV_CACHE_USAGE_PERC) >= 0.0
        assert store.get(NODE_ID, MetricKey.NUM_REQUESTS_RUNNING) >= 0
        assert store.get(NODE_ID, MetricKey.NUM_REQUESTS_WAITING) >= 0

    def test_store_get_node_dict(self, vllm_service):
        """
        Feature: MetricsStore.get(node_id) returns the full node metrics dict
        Expectation:
            Dict contains kv_cache_usage_perc, num_requests_running, num_requests_waiting.
        """
        store = MetricsStore.default()
        collector = _make_collector()

        collector.start()
        time.sleep(POLL_INTERVAL + 3.0)
        collector.stop()

        node_metrics = store.get(NODE_ID)
        assert isinstance(node_metrics, dict)
        assert MetricKey.KV_CACHE_USAGE_PERC in node_metrics
        assert MetricKey.NUM_REQUESTS_RUNNING in node_metrics
        assert MetricKey.NUM_REQUESTS_WAITING in node_metrics

    def test_multiple_poll_cycles_refresh(self, vllm_service):
        """
        Feature: Multiple polling cycles refresh the store with updated values
        Expectation:
            After 3 polling cycles the store contains data and values are reasonable.
        """
        store = MetricsStore.default()
        collector = _make_collector()

        collector.start()
        time.sleep(POLL_INTERVAL * 3 + 2.0)
        collector.stop()

        assert len(store.get(NODE_ID)) > 0, "Store should have metrics after multiple poll cycles"
