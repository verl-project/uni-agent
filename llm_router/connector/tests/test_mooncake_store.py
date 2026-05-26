"""Mooncake-backed KVStore implementations.

These tests skip entirely if the `mooncake` package is not installed.
On a real Mooncake-equipped host they round-trip a payload through the
configured backend.
"""
import ctypes
import json

import pytest

from llm_router.connector.prefix_hash import make_versioned_key  # noqa: E402
from llm_router.connector.store.mooncake import (  # noqa: E402
    MooncakeKVStore,
    MooncakePooledKVStore,
)


@pytest.fixture
def mooncake_store():
    pytest.importorskip(
        "mooncake.engine",
        reason="MooncakeKVStore tests require the `mooncake` package",
    )
    store = MooncakeKVStore.from_env(
        buffer_bytes=4 * 1024 * 1024,
        backend="local",
    )
    yield store
    store.close()


def test_put_and_get_roundtrip(mooncake_store):
    k = make_versioned_key([1, 2, 3], 3, "v0")
    mooncake_store.put(k, b"hello mooncake")
    assert mooncake_store.contains(k) is True
    assert mooncake_store.get(k) == b"hello mooncake"


def test_get_returns_none_on_miss(mooncake_store):
    assert mooncake_store.get(make_versioned_key([9], 1, "v0")) is None


def test_lru_eviction_reuses_registered_buffer(mooncake_store):
    k1 = make_versioned_key([1], 1, "v0")
    k2 = make_versioned_key([2], 1, "v0")
    k3 = make_versioned_key([3], 1, "v0")
    payload = b"x" * (2 * 1024 * 1024)

    mooncake_store.put(k1, payload)
    mooncake_store.put(k2, payload)
    # k1 becomes recent; k2 should be evicted first when k3 needs space.
    assert mooncake_store.get(k1) == payload
    mooncake_store.put(k3, payload)

    assert mooncake_store.contains(k1)
    assert not mooncake_store.contains(k2)
    assert mooncake_store.contains(k3)
    assert mooncake_store.stats().eviction_count >= 1


def test_delete_reuses_freed_slice(mooncake_store):
    k1 = make_versioned_key([1], 1, "v0")
    k2 = make_versioned_key([2], 1, "v0")
    mooncake_store.put(k1, b"A" * 1024)
    assert mooncake_store.delete(k1) is True
    mooncake_store.put(k2, b"B" * 1024)
    assert mooncake_store.get(k2) == b"B" * 1024


class FakeEngine:
    def shutdown(self):
        pass


@pytest.fixture
def fake_mooncake_store():
    from llm_router.connector.store.mooncake import MooncakeKVStore

    backing = ctypes.create_string_buffer(16)
    store = MooncakeKVStore(
        engine=FakeEngine(),
        buffer_ptr=ctypes.addressof(backing),
        buffer_bytes=16,
    )
    store._buffer_keepalive = backing
    return store


def test_allocator_eviction_without_mooncake_package(fake_mooncake_store):
    s = fake_mooncake_store
    k1 = make_versioned_key([1], 1, "v0")
    k2 = make_versioned_key([2], 1, "v0")
    k3 = make_versioned_key([3], 1, "v0")
    s.put(k1, b"AAAAAA")
    s.put(k2, b"BBBBBB")
    assert s.get(k1) == b"AAAAAA"
    s.put(k3, b"CCCCCC")
    assert s.contains(k1)
    assert not s.contains(k2)
    assert s.get(k3) == b"CCCCCC"
    assert s.stats().eviction_count == 1


def test_allocator_delete_reuses_free_list(fake_mooncake_store):
    s = fake_mooncake_store
    k1 = make_versioned_key([1], 1, "v0")
    k2 = make_versioned_key([2], 1, "v0")
    s.put(k1, b"AAAA")
    assert s.delete(k1)
    s.put(k2, b"B" * 16)
    assert s.get(k2) == b"B" * 16


class FakeReplicateConfig:
    def __init__(self):
        self.replica_num = 1
        self.with_soft_pin = False
        self.with_hard_pin = False
        self.preferred_segment = ""
        self.preferred_segments = []
        self.prefer_alloc_in_same_node = False


class FakeDistributedStore:
    def __init__(self):
        self.setup_config = None
        self.items = {}
        self.closed = False

    def setup(self, config):
        self.setup_config = dict(config)
        return 0

    def upsert(self, key, value, config):
        self.items[key] = bytes(value)
        self.last_config = config
        return 0

    def get(self, key):
        return self.items.get(key, b"")

    def is_exist(self, key):
        return 1 if key in self.items else 0

    def remove(self, key, force=False):
        self.items.pop(key, None)
        return 0

    def health_check(self):
        return 0

    def get_replica_desc(self, key):
        return []

    def close(self):
        self.closed = True


def test_pooled_store_roundtrip_uses_mooncake_distributed_store(monkeypatch):
    import llm_router.connector.store.mooncake as mooncake_store_module

    monkeypatch.setattr(
        mooncake_store_module,
        "MooncakePooledKVStore",
        mooncake_store_module.MooncakePooledKVStore,
    )
    monkeypatch.setattr(
        "mooncake.store.ReplicateConfig",
        FakeReplicateConfig,
    )

    store = MooncakePooledKVStore.from_env(
        extra_config={
            "mooncake_master": "127.0.0.1:50051",
            "mooncake_metadata_server": "P2PHANDSHAKE",
            "mooncake_local_hostname": "node-a",
            "mooncake_global_segment_bytes": 4096,
            "mooncake_local_buffer_bytes": 1024,
            "mooncake_protocol": "tcp",
            "mooncake_replica_num": 2,
            "mooncake_with_soft_pin": True,
        },
        store_factory=FakeDistributedStore,
    )

    k = make_versioned_key([1, 2, 3], 3, "v0")
    store.put(k, b"pooled-kv")

    assert store.contains(k)
    assert store.get(k) == b"pooled-kv"
    assert store.health_check() is True
    assert store.stats().capacity_bytes == 4096
    assert store.stats().used_bytes == len(b"pooled-kv")
    assert store._store.setup_config["master_server_addr"] == "127.0.0.1:50051"
    assert store._replicate_config.replica_num == 2
    assert store._replicate_config.with_soft_pin is True
    assert store._replicate_config.prefer_alloc_in_same_node is True
    assert store.delete(k) is True
    assert store.contains(k) is False
    store.close()
    assert store._store.closed is True


def test_pooled_replicate_config_can_disable_local_first(monkeypatch):
    monkeypatch.setattr(
        "mooncake.store.ReplicateConfig",
        FakeReplicateConfig,
    )

    config = MooncakePooledKVStore._build_replicate_config(
        {"mooncake_prefer_alloc_in_same_node": False}
    )

    assert config.prefer_alloc_in_same_node is False


class FakeBufferDescriptor:
    def __init__(self, endpoint):
        self.transport_endpoint = endpoint


class FakeMemoryDescriptor:
    def __init__(self, endpoint):
        self.buffer_descriptor = FakeBufferDescriptor(endpoint)


class FakeReplicaDescriptor:
    def __init__(self, endpoint):
        self._endpoint = endpoint

    def is_memory_replica(self):
        return True

    def get_memory_descriptor(self):
        return FakeMemoryDescriptor(self._endpoint)


def test_pooled_store_cpu_locations_use_local_server_for_local_descriptor():
    store = MooncakePooledKVStore(
        FakeDistributedStore(),
        setup_config={"local_hostname": "node-a", "global_segment_size": 4096},
        replicate_config=FakeReplicateConfig(),
    )
    key = make_versioned_key([1], 1, "v0")
    store._store.get_replica_desc = lambda _: [FakeReplicaDescriptor("node-a:1234")]

    assert store.cpu_locations(key, local_server_id="replica-a") == [
        "replica-a",
    ]


def test_pooled_store_cpu_locations_return_remote_endpoint_aliases():
    store = MooncakePooledKVStore(
        FakeDistributedStore(),
        setup_config={"local_hostname": "node-a", "global_segment_size": 4096},
        replicate_config=FakeReplicateConfig(),
    )
    key = make_versioned_key([1], 1, "v0")
    store._store.get_replica_desc = lambda _: [FakeReplicaDescriptor("node-b:1234")]

    assert store.cpu_locations(key, local_server_id="replica-a") == [
        "node-b:1234",
        "node-b",
    ]


def test_from_env_selects_pooled_backend_when_master_is_configured(monkeypatch):
    import llm_router.connector.store.mooncake as mooncake_store_module

    created = {}

    class FakePooledStore:
        @classmethod
        def has_pooled_config(cls, extra_config=None):
            return True

        @classmethod
        def from_env(cls, extra_config=None):
            created["extra"] = dict(extra_config or {})
            return "pooled-store"

    monkeypatch.setattr(
        mooncake_store_module,
        "MooncakePooledKVStore",
        FakePooledStore,
    )

    store = MooncakeKVStore.from_env(
        backend="auto",
        extra_config={"mooncake_master": "127.0.0.1:50051"},
    )

    assert store == "pooled-store"
    assert created["extra"]["mooncake_master"] == "127.0.0.1:50051"


def test_pooled_backend_requires_master_address():
    with pytest.raises(ValueError, match="requires a master address"):
        MooncakePooledKVStore._build_setup_config({})


def test_pooled_setup_config_can_read_mooncake_config_file(tmp_path):
    config_path = tmp_path / "mooncake.json"
    config_path.write_text(
        json.dumps(
            {
                "local_hostname": "node-file",
                "metadata_server": "meta-file:8080",
                "global_segment_size": 8192,
                "local_buffer_size": 4096,
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "master-file:50051",
            }
        )
    )

    setup = MooncakePooledKVStore._build_setup_config(
        {"mooncake_config_path": str(config_path)}
    )

    assert setup["local_hostname"] == "node-file"
    assert setup["metadata_server"] == "meta-file:8080"
    assert setup["global_segment_size"] == 8192
    assert setup["local_buffer_size"] == 4096
    assert setup["master_server_addr"] == "master-file:50051"
