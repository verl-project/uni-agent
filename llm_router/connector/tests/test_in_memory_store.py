"""InMemoryKVStore: LRU eviction by total byte capacity."""
import pytest

from llm_router.connector.prefix_hash import make_versioned_key
from llm_router.connector.store.in_memory import InMemoryKVStore


def _key(tokens, weight="v0"):
    return make_versioned_key(tokens, len(tokens), weight)


def test_put_and_get_roundtrip():
    s = InMemoryKVStore(max_bytes=1024)
    k = _key([1, 2, 3])
    s.put(k, b"hello")
    assert s.contains(k)
    assert s.get(k) == b"hello"


def test_get_returns_none_on_miss():
    s = InMemoryKVStore(max_bytes=1024)
    assert s.get(_key([9])) is None
    assert s.contains(_key([9])) is False


def test_lru_evicts_oldest_when_capacity_exceeded():
    s = InMemoryKVStore(max_bytes=10)
    s.put(_key([1, 1]), b"AAAA")  # 4 bytes
    s.put(_key([1, 2]), b"BBBB")  # 8 bytes total
    s.put(_key([1, 3]), b"CCCC")  # would be 12 bytes — evict oldest first
    assert not s.contains(_key([1, 1]))  # AAAA evicted
    assert s.contains(_key([1, 2]))
    assert s.contains(_key([1, 3]))


def test_get_promotes_lru_recency():
    s = InMemoryKVStore(max_bytes=10)
    s.put(_key([1, 1]), b"AAAA")
    s.put(_key([1, 2]), b"BBBB")
    # Access AAAA → it becomes most recent
    s.get(_key([1, 1]))
    s.put(_key([1, 3]), b"CCCC")  # forces eviction; should evict BBBB now
    assert s.contains(_key([1, 1]))
    assert not s.contains(_key([1, 2]))
    assert s.contains(_key([1, 3]))


def test_overwrite_same_key_does_not_double_count():
    s = InMemoryKVStore(max_bytes=10)
    s.put(_key([1, 1]), b"AAAA")
    s.put(_key([1, 1]), b"AAAA")  # same key — must not exceed cap
    s.put(_key([1, 2]), b"BBBB")
    s.put(_key([1, 3]), b"CC")  # 4+4+2=10, fits exactly
    assert s.contains(_key([1, 1]))
    assert s.contains(_key([1, 2]))
    assert s.contains(_key([1, 3]))


def test_payload_too_large_raises():
    s = InMemoryKVStore(max_bytes=10)
    with pytest.raises(ValueError, match="exceeds capacity"):
        s.put(_key([1, 1]), b"x" * 11)


def test_different_versions_are_distinct_keys():
    s = InMemoryKVStore(max_bytes=1024)
    k_v0 = _key([1, 2, 3], weight="v0")
    k_v1 = _key([1, 2, 3], weight="v1")
    s.put(k_v0, b"old")
    s.put(k_v1, b"new")
    assert s.get(k_v0) == b"old"
    assert s.get(k_v1) == b"new"


def test_delete_frees_capacity():
    s = InMemoryKVStore(max_bytes=8)
    k = _key([1, 2])
    s.put(k, b"AAAA")
    assert s.delete(k) is True
    assert s.delete(k) is False
    assert s.stats().used_bytes == 0
    s.put(_key([1, 3]), b"BBBBBBBB")
    assert s.stats().used_bytes == 8


def test_stats_tracks_evictions_and_capacity():
    s = InMemoryKVStore(max_bytes=8)
    s.put(_key([1, 1]), b"AAAA")
    s.put(_key([1, 2]), b"BBBB")
    s.put(_key([1, 3]), b"CCCC")
    stats = s.stats()
    assert stats.capacity_bytes == 8
    assert stats.used_bytes == 8
    assert stats.free_bytes == 0
    assert stats.entry_count == 2
    assert stats.eviction_count == 1


def test_default_async_get_contract_completes_sync_read():
    s = InMemoryKVStore(max_bytes=8)
    k = _key([1, 2])
    s.put(k, b"AAAA")
    transfer_id = s.begin_get(k)
    result = s.poll_get(transfer_id)
    assert result.state == "done"
    assert result.payload == b"AAAA"
