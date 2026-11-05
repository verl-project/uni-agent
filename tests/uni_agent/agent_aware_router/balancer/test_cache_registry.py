"""The balancer installs the layout; the cache store owns it."""

from types import SimpleNamespace
from unittest.mock import ANY, Mock

import msgpack
import pytest

from uni_agent.agent_aware_router.collectors.collector import Collector
from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.collectors.transport.base import TransportMessage
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental

from ._helpers import _FakeCollectorManager, _make_balancer, _router_config

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_each_replica_registers_metadata_before_collectors_start(monkeypatch, metadata):
    from uni_agent.agent_aware_router.balancer import KVCAwareBalancer

    first_metadata = metadata()
    second_metadata = metadata()
    # Opaque, distinct profiles test forwarding without constructing a vLLM codec.
    first_metadata["native_hash_config"] = {"root_hash_hex": "11" * 32}
    second_metadata["native_hash_config"] = {"root_hash_hex": "22" * 32}
    metadata_by_node = {"a": first_metadata, "b": second_metadata}
    first_call = Mock(return_value=first_metadata)
    second_call = Mock(return_value=second_metadata)
    first = SimpleNamespace(get_kv_cache_metadata=SimpleNamespace(remote=first_call))
    second = SimpleNamespace(get_kv_cache_metadata=SimpleNamespace(remote=second_call))
    monkeypatch.setattr("uni_agent.agent_aware_router.balancer.ray.get", lambda ref, timeout: ref)
    configure_spy = Mock(wraps=DataStore.configure_cache_registry)

    def configure(store, cache_metadata, *, replica_metadata=None):
        return configure_spy(store, cache_metadata, replica_metadata=replica_metadata)

    monkeypatch.setattr(DataStore, "configure_cache_registry", configure)

    class Provider(_FakeCollectorManager):
        def start(self):
            configure_spy.assert_called_once_with(ANY, first_metadata, replica_metadata=metadata_by_node)
            state = DataStore().cache_registry_state
            assert state.status == "registered"
            super().start()

    balancer = KVCAwareBalancer({"a": first, "b": second}, _router_config(), provider_factory=Provider)
    assert balancer._provider.started
    state = DataStore().cache_registry_state
    balancer.add_servers({"c": second})
    balancer.remove_servers(["a", "b"])
    assert DataStore().cache_registry_state is state
    first_call.assert_called_once()
    assert second_call.call_count == 2  # The newly added replica binds its own profile.


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
    assert store.estimate_prefixes(["a"], [], 3)["a"].estimated_hit_tokens == 0
    collector = Collector(object(), VLLMKVParser())
    try:
        event = ["BlockStored", ["h4"], None, list(range(4)), 4, None, None]
        update = collector._parser.parse(TransportMessage(msgpack.packb([0, [event]])), "a")
        collector._write_kv_update(update)
        assert len(update.events) == 1
        hashes, _ = get_prefix_hashes_incremental(list(range(5)), 4, 0, 0)
        estimate = store.estimate_prefixes(["a"], [str(value) for value in hashes], 5)["a"]
        assert estimate.estimated_hit_tokens == (4 if failure else 0)
    finally:
        collector._loop.close()
    assert balancer.acquire_server("request", [1, 2, 3])[0] == "a"
