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

from __future__ import annotations

from threading import Barrier
from types import SimpleNamespace

import httpx
import pytest
from omegaconf import OmegaConf

import uni_agent.agent_aware_router.balancer as balancer_module
from uni_agent.agent_aware_router.balancer import KVCAwareBalancer
from uni_agent.agent_aware_router.config.base import ConfigError
from uni_agent.agent_aware_router.strategies.kvc_aware import KVCacheAwareStrategy

from ._helpers import _FakeCollectorManager, _make_balancer, _router_config

pytestmark = [pytest.mark.level0, pytest.mark.cpu]


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn
        self.calls = 0

    def remote(self):
        self.calls += 1
        return self._fn()


class _RolloutServerHandle:
    def __init__(self, rollout_config, address=("127.0.0.1", 8000)):
        def get_rollout_config():
            if isinstance(rollout_config, Exception):
                raise rollout_config
            return rollout_config

        self.get_rollout_config = _RemoteMethod(get_rollout_config)
        self.get_server_address = _RemoteMethod(lambda: address)


def _kv_source(**overrides):
    config = {
        "publisher": "zmq",
        "endpoint": "tcp://10.0.0.1:41233",
        "replay_endpoint": "tcp://10.0.0.1:41234",
        "topic": "kv-events",
    }
    config.update(overrides)
    return {"0": config}


def _rollout_config(**overrides):
    config = {
        "custom": {
            "agent_framework": {
                "router": {"load_threshold": 0.75, "http_interval": 3.0},
            },
        },
        "max_num_seqs": 128,
        "max_num_batched_tokens": 4096,
    }
    config.update(overrides)
    return OmegaConf.create(config)


class TestKVEventSourceDiscovery:
    @pytest.mark.parametrize(
        ("replay_endpoint", "expected_replay"),
        [("tcp://10.0.0.1:41234", "10.0.0.1:41234"), (None, "")],
    )
    def test_single_dp_source_uses_collector_shape(self, replay_endpoint, expected_replay):
        assert KVCAwareBalancer._parse_kv_event_sources(_kv_source(replay_endpoint=replay_endpoint)) == [
            "10.0.0.1:41233",
            expected_replay,
            "zmq",
            "kv-events",
        ]

    def test_empty_response_disables_zmq_discovery(self):
        assert KVCAwareBalancer._parse_kv_event_sources({}) is None

    @pytest.mark.parametrize(
        ("sources", "error"),
        [
            (None, "invalid KV-event sources"),
            ({"0": None}, "invalid KV-event config"),
            (_kv_source(publisher="redis"), "unsupported KV-event publisher"),
            (_kv_source(topic=None), "invalid KV-event topic"),
            (_kv_source(endpoint="ipc:///tmp/kv-events"), "unsupported KV-event endpoint"),
            ({"0": {}, "1": {}}, "does not support multiple DP ranks"),
        ],
    )
    def test_invalid_sources_fail_fast(self, sources, error):
        with pytest.raises(RuntimeError, match=error):
            KVCAwareBalancer._parse_kv_event_sources(sources)

    def test_fetch_uses_vllm_http_api(self, monkeypatch):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=_kv_source())

        transport = httpx.MockTransport(handler)
        real_client = httpx.Client

        def client_factory(**kwargs):
            assert kwargs == {"timeout": 5.0, "trust_env": False}
            return real_client(transport=transport)

        monkeypatch.setattr(balancer_module.httpx, "Client", client_factory)

        assert KVCAwareBalancer._fetch_kv_event_endpoints("10.0.0.1:8000") == [
            "10.0.0.1:41233",
            "10.0.0.1:41234",
            "zmq",
            "kv-events",
        ]
        assert requests[0].url == httpx.URL("http://10.0.0.1:8000/kv_event_sources")

    def test_provider_discovers_sources_in_parallel_and_tolerates_unavailable_servers(self, monkeypatch):
        monkeypatch.setattr(balancer_module.ray, "get", lambda value: value)
        concurrent_fetches = Barrier(3, timeout=5)

        def fetch(_cls, address):
            concurrent_fetches.wait()
            if address.endswith(":8001"):
                raise httpx.ConnectError("offline")
            if address.endswith(":8002"):
                return None
            return ["10.0.0.1:41233", "10.0.0.1:41234", "zmq", "kv-events"]

        monkeypatch.setattr(KVCAwareBalancer, "_fetch_kv_event_endpoints", classmethod(fetch))
        servers = {
            "s0": _RolloutServerHandle(_rollout_config(), ("10.0.0.1", 8000)),
            "s1": _RolloutServerHandle(_rollout_config(), ("10.0.0.1", 8001)),
            "s2": _RolloutServerHandle(_rollout_config(), ("10.0.0.1", 8002)),
        }

        balancer = KVCAwareBalancer(servers, _router_config(), provider_factory=_FakeCollectorManager)

        assert balancer._provider.server_addresses == {
            "s0": "10.0.0.1:8000",
            "s1": "10.0.0.1:8001",
            "s2": "10.0.0.1:8002",
        }
        assert balancer._provider.kv_event_endpoints == {"s0": ["10.0.0.1:41233", "10.0.0.1:41234", "zmq", "kv-events"]}


class TestConstruction:
    def test_wires_strategy_and_collectors(self):
        balancer = _make_balancer({"s0": "h0"})

        assert balancer._provider.started
        assert balancer._provider.collection_names == [
            "inflight_stat",
            "sticky_stat",
            "vllm_metrics",
            "vllm_zmq",
        ]
        assert isinstance(balancer._strategy, KVCacheAwareStrategy)

    def test_rejects_empty_servers_and_missing_strategy(self):
        with pytest.raises(ValueError, match="servers must be non-empty"):
            KVCAwareBalancer({}, _router_config())
        with pytest.raises(ConfigError):
            KVCAwareBalancer({"s0": "h0"}, OmegaConf.create({}))

    def test_fetches_rollout_config_once(self, monkeypatch):
        monkeypatch.setattr(balancer_module.ray, "get", lambda value: value)
        monkeypatch.setattr(KVCAwareBalancer, "_fetch_kv_event_endpoints", classmethod(lambda _cls, _addr: None))
        server = _RolloutServerHandle(_rollout_config())

        balancer = KVCAwareBalancer({"s0": server}, _router_config(), provider_factory=_FakeCollectorManager)

        assert server.get_rollout_config.calls == 1
        assert balancer._config.strategy.load_threshold == 0.75
        assert balancer._config.collector.http_interval == 3.0
        assert balancer._strategy._max_num_seqs == 128
        assert balancer._strategy._max_num_batched_tokens == 4096

    def test_uses_next_server_when_rollout_config_rpc_fails(self, monkeypatch):
        monkeypatch.setattr(balancer_module.ray, "get", lambda value: value)
        monkeypatch.setattr(KVCAwareBalancer, "_fetch_kv_event_endpoints", classmethod(lambda _cls, _addr: None))
        failed = _RolloutServerHandle(RuntimeError("actor unavailable"), ("10.0.0.1", 8000))
        healthy = _RolloutServerHandle(_rollout_config(max_num_seqs=32), ("10.0.0.1", 8001))

        balancer = KVCAwareBalancer(
            {"failed": failed, "healthy": healthy},
            _router_config(),
            provider_factory=_FakeCollectorManager,
        )

        assert failed.get_rollout_config.calls == 1
        assert healthy.get_rollout_config.calls == 1
        assert balancer._strategy._max_num_seqs == 32

    def test_invalid_capacity_values_use_defaults(self, monkeypatch):
        monkeypatch.setattr(balancer_module.ray, "get", lambda value: value)
        monkeypatch.setattr(KVCAwareBalancer, "_fetch_kv_event_endpoints", classmethod(lambda _cls, _addr: None))
        server = _RolloutServerHandle(_rollout_config(max_num_seqs=0, max_num_batched_tokens="invalid"))

        balancer = KVCAwareBalancer({"s0": server}, _router_config(), provider_factory=_FakeCollectorManager)

        assert balancer._strategy._max_num_seqs == 256
        assert balancer._strategy._max_num_batched_tokens == 2048

    @pytest.mark.parametrize(
        ("rollout_config", "expected_override", "expected_capacity"),
        [
            (
                {
                    "custom": {"agent_framework": {"router": {"load_threshold": 0.5}}},
                    "max_num_seqs": 32,
                },
                {"load_threshold": 0.5},
                32,
            ),
            (
                SimpleNamespace(
                    custom=SimpleNamespace(
                        agent_framework=SimpleNamespace(router={"load_threshold": 0.5}),
                    ),
                    max_num_seqs=32,
                ),
                {"load_threshold": 0.5},
                32,
            ),
            ({}, None, 256),
        ],
    )
    def test_rollout_config_helpers_accept_mappings_and_objects(
        self, rollout_config, expected_override, expected_capacity
    ):
        assert KVCAwareBalancer._router_override(rollout_config) == expected_override
        assert KVCAwareBalancer._rollout_config_int(rollout_config, "max_num_seqs", 256) == expected_capacity


class TestBalancerProtocol:
    def test_status_describes_runtime_state(self):
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})

        assert balancer.get_all_servers() == ["s0", "s1"]
        assert balancer.get_status() == {
            "servers": ["s0", "s1"],
            "provider": "_FakeCollectorManager",
            "strategies": [{"type": "KVCacheAwareStrategy"}],
            "route_calls": 0,
            "sticky_size": 0,
            "total_inflight": 0,
        }

    def test_acquire_delegates_and_maps_selected_handle(self, monkeypatch):
        seen = {}

        def fake_route(strategy, prompt_ids, store, replicas, request_id):
            seen.update(
                strategy=strategy,
                prompt_ids=prompt_ids,
                store=store,
                replica_ids=[replica.replica_id for replica in replicas],
                request_id=request_id,
            )
            return ["s1", "s0"]

        monkeypatch.setattr(balancer_module, "route", fake_route)
        balancer = _make_balancer({"s0": "h0", "s1": "h1"})

        assert balancer.acquire_server("request-1", [7, 8, 9]) == ("s1", "h1")
        assert seen == {
            "strategy": balancer._strategy,
            "prompt_ids": [7, 8, 9],
            "store": balancer._store,
            "replica_ids": ["s0", "s1"],
            "request_id": "request-1",
        }

    def test_empty_ranking_raises(self, monkeypatch):
        monkeypatch.setattr(balancer_module, "route", lambda *_args: [])
        balancer = _make_balancer({"s0": "h0"})

        with pytest.raises(RuntimeError, match="no available replica"):
            balancer.acquire_server("request-1", [1])

    def test_callback_failure_does_not_break_other_callbacks(self):
        balancer = _make_balancer({"s0": "h0"})
        received = []

        def fail(*_args):
            raise RuntimeError("broken callback")

        balancer.register_call_back("custom", fail)
        balancer.register_call_back("custom", lambda value: received.append(value))

        balancer._fire("custom", "payload")

        assert received == ["payload"]

    def test_periodic_route_stats_are_cumulative(self, monkeypatch):
        balancer = _make_balancer({"s0": "h0"})
        balancer._ROUTE_LOG_EVERY = 2
        stats = []
        monkeypatch.setattr(
            balancer_module.logger,
            "info",
            lambda message, *args: stats.append(message % args),
        )

        for index in range(4):
            balancer.acquire_server(f"request-{index}", [index])

        assert len(stats) == 2
        assert stats[0].startswith("route-stats: calls=2 ")
        assert stats[1].startswith("route-stats: calls=4 ")
        assert balancer._route_calls == 4
        assert balancer._route_time_total_ms >= balancer._route_time_max_ms >= 0

    def test_add_and_remove_servers_update_pool_and_counters(self):
        balancer = _make_balancer({"s0": "h0"})

        balancer.add_servers({"s0": "new-h0", "s1": "h1"})
        assert balancer._servers == {"s0": "new-h0", "s1": "h1"}
        assert balancer._inflight == {"s0": 0, "s1": 0}

        balancer.remove_servers(["s0", "missing"])
        balancer.remove_servers([])
        assert balancer.get_all_servers() == ["s1"]

    def test_statistic_callbacks_unregister_on_provider_stop(self):
        balancer = _make_balancer({"s0": "h0"})
        assert any(balancer._callbacks.values())

        balancer._provider.stop()
        balancer._provider.stop()

        assert not any(balancer._callbacks.values())
