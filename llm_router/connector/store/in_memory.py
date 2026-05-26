"""In-memory KVStore for unit tests. Production uses MooncakeKVStore."""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

from llm_router.connector.prefix_hash import VersionedKey
from llm_router.connector.store.base import KVStore, StoreStats


class InMemoryKVStore(KVStore):
    """LRU-evicting blob store capped by total payload bytes.

    Eviction is exact: when a put would exceed `max_bytes`, the least-recently
    accessed entries are dropped until the new payload fits. A single payload
    larger than `max_bytes` is rejected with ValueError.
    """

    def __init__(self, max_bytes: int = 64 * 1024 * 1024 * 1024):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._max_bytes = max_bytes
        self._items: OrderedDict[VersionedKey, bytes] = OrderedDict()
        self._used_bytes = 0
        self._eviction_count = 0

    def put(self, key: VersionedKey, payload: bytes) -> None:
        size = len(payload)
        if size > self._max_bytes:
            raise ValueError(
                f"payload size {size} exceeds capacity {self._max_bytes}"
            )
        if key in self._items:
            self._used_bytes -= len(self._items[key])
            del self._items[key]
        while self._used_bytes + size > self._max_bytes and self._items:
            _, evicted = self._items.popitem(last=False)
            self._used_bytes -= len(evicted)
            self._eviction_count += 1
        self._items[key] = payload
        self._used_bytes += size

    def get(self, key: VersionedKey) -> Optional[bytes]:
        if key not in self._items:
            return None
        self._items.move_to_end(key, last=True)
        return self._items[key]

    def contains(self, key: VersionedKey) -> bool:
        return key in self._items

    def delete(self, key: VersionedKey) -> bool:
        if key not in self._items:
            return False
        payload = self._items.pop(key)
        self._used_bytes -= len(payload)
        return True

    def stats(self) -> StoreStats:
        return StoreStats(
            capacity_bytes=self._max_bytes,
            used_bytes=self._used_bytes,
            free_bytes=self._max_bytes - self._used_bytes,
            entry_count=len(self._items),
            eviction_count=self._eviction_count,
        )
