"""Balancer registers metadata sources at startup only."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from uni_agent.agent_aware_router.balancer import KVCAwareBalancer

from ._helpers import _FakeCollectorManager, _make_balancer, _router_config

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_runtime_routing_pool_changes_do_not_change_registered_sources():
    remote = Mock()
    first = SimpleNamespace(get_kv_cache_metadata=SimpleNamespace(remote=remote))
    second = SimpleNamespace(get_kv_cache_metadata=SimpleNamespace(remote=remote))
    balancer = KVCAwareBalancer({"a": first}, _router_config(), provider_factory=_FakeCollectorManager)
    assert balancer._provider.started
    assert balancer._provider.metadata_servers == {"a": first}
    remote.assert_not_called()
    balancer.add_servers({"b": second})
    assert balancer.get_all_servers() == ["a", "b"]
    assert balancer._provider.metadata_servers == {"a": first}
    balancer.remove_servers(["a"])
    assert balancer.get_all_servers() == ["b"]
    assert balancer._provider.metadata_servers == {"a": first}


def test_replacing_initial_source_requires_rebuilding_balancer():
    balancer = _make_balancer({"a": "old-handle"})
    balancer.remove_servers(["a"])
    with pytest.raises(ValueError, match="requires a new balancer"):
        balancer.add_servers({"a": "new-handle"})
    assert balancer.get_all_servers() == []
    balancer.add_servers({"a": "old-handle"})
    assert balancer.get_all_servers() == ["a"]


def test_no_metadata_does_not_prevent_routing():
    balancer = _make_balancer({"a": "server"})
    assert balancer._provider.started
    assert balancer.acquire_server("request", [1, 2, 3])[0] == "a"
