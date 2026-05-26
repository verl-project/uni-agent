"""Abstract base for the L2 KV store sitting behind a Mooncake KV connector.

Two implementations exist:
- InMemoryKVStore — used by unit tests; pure Python, LRU eviction.
- MooncakeKVStore — real Mooncake TransferEngine; CPU/RDMA-resident pool.

The connector serializes a list of per-layer KV tensors into bytes, then
hands them to the store via put(); on miss it queries get() to reconstruct
KV blocks. Layout is opaque to the store — the store treats payloads as
blobs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional
from uuid import uuid4

from llm_router.connector.prefix_hash import VersionedKey


@dataclass(frozen=True)
class StoreStats:
    """Point-in-time capacity and transfer counters for a KVStore."""

    capacity_bytes: int
    used_bytes: int
    free_bytes: int
    entry_count: int
    eviction_count: int = 0
    pending_transfer_count: int = 0


@dataclass(frozen=True)
class TransferResult:
    """Result of an asynchronous store read."""

    state: Literal["pending", "done", "failed"]
    payload: Optional[bytes] = None
    error: Optional[str] = None

    @classmethod
    def pending(cls) -> TransferResult:
        return cls(state="pending")

    @classmethod
    def done(cls, payload: Optional[bytes]) -> TransferResult:
        return cls(state="done", payload=payload)

    @classmethod
    def failed(cls, error: BaseException | str) -> TransferResult:
        return cls(state="failed", error=str(error))


class KVStore(ABC):
    """Versioned blob store for serialized KV blocks.

    The synchronous methods are the core contract. The async methods default
    to a completed sync read so test backends do not need thread machinery,
    while Mooncake-backed stores can override them with real RDMA polling.
    """

    @abstractmethod
    def put(self, key: VersionedKey, payload: bytes) -> None: ...

    @abstractmethod
    def get(self, key: VersionedKey) -> Optional[bytes]: ...

    @abstractmethod
    def contains(self, key: VersionedKey) -> bool: ...

    @abstractmethod
    def delete(self, key: VersionedKey) -> bool: ...

    @abstractmethod
    def stats(self) -> StoreStats: ...

    def cpu_locations(
        self,
        key: VersionedKey,
        *,
        local_server_id: str | None = None,
    ) -> list[str]:
        """Return router-visible owners for a CPU-resident copy of `key`.

        Local test stores and the compatibility TransferEngine buffer are
        process-local, so the only meaningful CPU owner is the current server.
        Pooled Mooncake stores override this with placement descriptors.
        """
        if local_server_id and self.contains(key):
            return [local_server_id]
        return []

    def begin_get(self, key: VersionedKey) -> str:
        """Start an async read and return a transfer id.

        Default implementation performs the read immediately and stores the
        completed result for `poll_get`.
        """
        transfer_id = uuid4().hex
        try:
            result = TransferResult.done(self.get(key))
        except Exception as exc:  # pragma: no cover - defensive default
            result = TransferResult.failed(exc)
        transfers = self.__dict__.setdefault("_sync_transfer_results", {})
        transfers[transfer_id] = result
        return transfer_id

    def poll_get(self, transfer_id: str) -> TransferResult:
        transfers = self.__dict__.setdefault("_sync_transfer_results", {})
        return transfers.pop(transfer_id, TransferResult.failed("unknown transfer_id"))

    def cancel(self, transfer_id: str) -> None:
        transfers = self.__dict__.setdefault("_sync_transfer_results", {})
        transfers.pop(transfer_id, None)
