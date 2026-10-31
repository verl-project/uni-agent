"""The balancer installs the layout; the cache store owns it."""

from types import SimpleNamespace
from unittest.mock import Mock

import msgpack
import pytest

from uni_agent.agent_aware_router.collectors.collector import Collector
from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.types import Layer

from ._helpers import _FakeCollectorManager, _make_balancer, _router_config

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_one_replica_registers_shared_layout_before_collectors_start(monkeypatch, metadata):
    from uni_agent.agent_aware_router.balancer import KVCAwareBalancer

    first_call = Mock(return_value=metadata())
    second_call = Mock(side_effect=AssertionError("Do not query all replicas"))
    first = SimpleNamespace(get_kv_cache_metadata=SimpleNamespace(remote=first_call))
    second = SimpleNamespace(get_kv_cache_metadata=SimpleNamespace(remote=second_call))
    monkeypatch.setattr("uni_agent.agent_aware_router.balancer.ray.get", lambda ref, timeout: ref)

    class Provider(_FakeCollectorManager):
        def start(self):
            state = DataStore().cache_registry_state
            assert state.status == "registered"
            super().start()

    balancer = KVCAwareBalancer({"a": first, "b": second}, _router_config(), provider_factory=Provider)
    state = DataStore().cache_registry_state
    balancer.add_servers({"c": second})
    balancer.remove_servers(["a", "b"])
    assert DataStore().cache_registry_state is state
    first_call.assert_called_once()
    second_call.assert_not_called()


@pytest.mark.parametrize("failure", [None, TimeoutError("timeout")])
def test_failed_registration_has_reason_and_still_starts_existing_router(monkeypatch, failure, stored):
    def get(ref, timeout):
        if failure:
            raise failure
        return {"status": "ok", "schema_version": 1}

    monkeypatch.setattr("uni_agent.agent_aware_router.balancer.ray.get", get)
    handle = SimpleNamespace(get_kv_cache_metadata=SimpleNamespace(remote=lambda: None))
    balancer = _make_balancer({"a": handle})
    state = DataStore().cache_registry_state
    assert state.registry is None
    assert state.reason
    assert balancer._provider.started
    store = DataStore()
    assert not store.group_cache_enabled
    assert store.estimate_prefixes(["a"], [], 3) is None
    collector = Collector(object(), VLLMKVParser(group_aware=store.group_cache_enabled))
    try:
        update = collector._parser.parse(msgpack.packb({"events": [stored()]}), "a")
        collector._write_kv_update(update)
        assert update.group_events is None
        hashes = update.add_blocks[Layer.GPU]
        assert store.get_layer_prefix_hit_rate("a", hashes, Layer.GPU) == 1.0
    finally:
        collector._loop.close()
    assert balancer.acquire_server("request", [1, 2, 3])[0] == "a"
