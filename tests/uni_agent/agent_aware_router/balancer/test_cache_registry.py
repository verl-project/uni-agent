"""Balancer registers metadata sources at startup only."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import uni_agent.agent_aware_router.balancer as balancer_module
from uni_agent.agent_aware_router.balancer import KVCAwareBalancer
from uni_agent.agent_aware_router.types import MetricKey

from ._helpers import _FakeCollectorManager, _make_balancer, _router_config


@pytest.mark.cpu
@pytest.mark.level0
def test_initial_metadata_wait_is_bounded_and_applies_only_ready_replicas(monkeypatch, layout):
    slow, failed, healthy = "ref-slow", "ref-failed", "ref-healthy"
    servers = {
        node: SimpleNamespace(get_kv_cache_metadata=SimpleNamespace(remote=Mock(return_value=ref)))
        for node, ref in (("a", slow), ("b", failed), ("c", healthy))
    }
    wait = Mock(return_value=([failed, healthy], [slow]))
    metadata = layout((("full_attention", 4),), unit=4)
    get = Mock(side_effect=[RuntimeError("unavailable"), metadata])
    monkeypatch.setattr(balancer_module.ray, "wait", wait)
    monkeypatch.setattr(balancer_module.ray, "get", get)
    balancer = KVCAwareBalancer(servers, _router_config(), provider_factory=_FakeCollectorManager)
    wait.assert_called_once_with([slow, failed, healthy], num_returns=3, timeout=5.0)
    assert [call.args[0] for call in get.call_args_list] == [failed, healthy]
    assert balancer._store.cache_registry_state("c").registry.local_hash_unit == 4
    assert balancer._store.cache_registry_state("a").registry is None
    assert balancer._store.cache_registry_state("b").registry is None
    balancer.acquire_server("request", [1, 2, 3])
    for server in servers.values():
        server.get_kv_cache_metadata.remote.assert_called_once_with()


@pytest.mark.cpu
@pytest.mark.level0
def test_replacing_initial_source_requires_rebuilding_balancer():
    balancer = _make_balancer({"a": "old-handle"})
    balancer.remove_servers(["a"])
    with pytest.raises(ValueError, match="requires a new balancer"):
        balancer.add_servers({"a": "new-handle"})
    assert balancer.get_all_servers() == []
    balancer.add_servers({"a": "old-handle"})
    assert balancer.get_all_servers() == ["a"]


@pytest.mark.cpu
@pytest.mark.level0
def test_no_metadata_does_not_prevent_routing():
    balancer = _make_balancer({"busy": "plain-server-a", "idle": "plain-server-b"})
    assert balancer._provider.started
    balancer._store.incr_metrics("busy", {MetricKey.INFLIGHT_COUNT: 3})
    assert balancer.acquire_server("request", [1, 2, 3])[0] == "idle"
