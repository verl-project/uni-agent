"""KVStore abstract base class contract."""
import pytest

from llm_router.connector.prefix_hash import make_versioned_key
from llm_router.connector.store.base import KVStore, StoreStats


def test_kvstore_is_abstract():
    with pytest.raises(TypeError):
        KVStore()


def test_kvstore_subclass_must_implement_get_put_contains():
    class Incomplete(KVStore):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_kvstore_minimal_subclass_works():
    class Minimal(KVStore):
        def __init__(self):
            self._d = {}

        def put(self, key, payload):
            self._d[key] = payload

        def get(self, key):
            return self._d.get(key)

        def contains(self, key):
            return key in self._d

        def delete(self, key):
            return self._d.pop(key, None) is not None

        def stats(self):
            used = sum(len(v) for v in self._d.values())
            return StoreStats(
                capacity_bytes=1024,
                used_bytes=used,
                free_bytes=1024 - used,
                entry_count=len(self._d),
            )

    s = Minimal()
    k = make_versioned_key([1, 2, 3], 3, "v0")
    assert s.contains(k) is False
    assert s.get(k) is None
    s.put(k, b"payload")
    assert s.contains(k) is True
    assert s.get(k) == b"payload"
    assert s.stats().entry_count == 1
    assert s.delete(k) is True
    assert s.contains(k) is False
