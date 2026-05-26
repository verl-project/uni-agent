"""Mooncake-backed L2 KV stores.

Storage model
-------------
The production path uses Mooncake's ``MooncakeDistributedStore`` client. That
is the pooled L2: writes go through Mooncake's master/metadata services and can
be placed on remote segments rather than being capped by one replica's local
buffer.

The older TransferEngine-backed local buffer is kept as a compatibility and
test backend. It is useful on a single host, but it is not a pooled L2.
"""
from __future__ import annotations

import os
import socket
import threading
from collections import OrderedDict
from typing import Any, Optional
from urllib.parse import urlparse

from llm_router.connector.prefix_hash import VersionedKey
from llm_router.connector.store.base import KVStore, StoreStats

DEFAULT_BUFFER_BYTES = 32 * 1024 * 1024 * 1024  # 32 GiB
DEFAULT_DEVICE_NAME = ""  # let TransferEngine pick
DEFAULT_GLOBAL_SEGMENT_BYTES = 3355443200  # MooncakeConfig default: 3.125 GiB
DEFAULT_LOCAL_BUFFER_BYTES = 1073741824  # MooncakeConfig default: 1 GiB
DEFAULT_METADATA_SERVER = "P2PHANDSHAKE"
DEFAULT_PROTOCOL = "tcp"

_POOLED_BACKENDS = {"pooled", "distributed", "store", "mooncake_store"}
_LOCAL_BACKENDS = {"local", "transfer_engine", "buffer"}


def _get_any(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text.endswith("gb"):
            return int(text[:-2].strip()) * 1024 * 1024 * 1024
        if text.endswith("mb"):
            return int(text[:-2].strip()) * 1024 * 1024
    return int(value)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class MooncakePooledKVStore(KVStore):
    """KVStore adapter over Mooncake's distributed pooled store."""

    def __init__(
        self,
        store: Any,
        *,
        setup_config: dict[str, Any],
        replicate_config: Any,
    ):
        self._store = store
        self._setup_config = dict(setup_config)
        self._replicate_config = replicate_config
        self._known_sizes: dict[str, int] = {}

    @classmethod
    def from_env(
        cls,
        *,
        extra_config: dict[str, Any] | None = None,
        store_factory: Any = None,
    ) -> MooncakePooledKVStore:
        extra = extra_config or {}
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:  # pragma: no cover - covered by host-only tests
            raise ImportError(
                "Mooncake pooled L2 requires `mooncake.store.MooncakeDistributedStore`. "
                "Install mooncake-transfer-engine with store support."
            ) from e

        setup_config = cls._build_setup_config(extra)
        store = (store_factory or MooncakeDistributedStore)()
        ret = store.setup(setup_config)
        if ret != 0:
            raise RuntimeError(
                "MooncakeDistributedStore.setup failed "
                f"ret={ret}, config={setup_config!r}"
            )
        return cls(
            store,
            setup_config=setup_config,
            replicate_config=cls._build_replicate_config(extra),
        )

    @classmethod
    def has_pooled_config(cls, extra_config: dict[str, Any] | None = None) -> bool:
        extra = extra_config or {}
        return bool(
            os.getenv("MOONCAKE_CONFIG_PATH")
            or os.getenv("MOONCAKE_MASTER")
            or _get_any(
                extra,
                "mooncake_master",
                "mooncake_master_server_addr",
                "mooncake_master_server_address",
                "master_server_addr",
                "master_server_address",
            )
        )

    @classmethod
    def _build_setup_config(cls, extra: dict[str, Any]) -> dict[str, Any]:
        file_defaults = cls._load_config_file_defaults(extra)
        master = _get_any(
            extra,
            "mooncake_master",
            "mooncake_master_server_addr",
            "mooncake_master_server_address",
            "master_server_addr",
            "master_server_address",
            default=os.getenv("MOONCAKE_MASTER")
            or file_defaults.get("master_server_addr"),
        )
        if not master:
            raise ValueError(
                "Mooncake pooled L2 requires a master address. Set "
                "`mooncake_master` in kv_connector_extra_config or MOONCAKE_MASTER."
            )

        local_hostname = _get_any(
            extra,
            "mooncake_local_hostname",
            "local_hostname",
            default=os.getenv("MOONCAKE_LOCAL_HOSTNAME")
            or file_defaults.get("local_hostname")
            or socket.gethostname(),
        )
        metadata_server = _get_any(
            extra,
            "mooncake_metadata_server",
            "metadata_server",
            default=os.getenv("MOONCAKE_TE_META_DATA_SERVER")
            or file_defaults.get("metadata_server")
            or DEFAULT_METADATA_SERVER,
        )
        local_buffer = _as_int(
            _get_any(
                extra,
                "mooncake_local_buffer_bytes",
                "mooncake_local_buffer_size",
                "local_buffer_size",
                "mooncake_buffer_bytes",
                default=os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE")
                or file_defaults.get("local_buffer_size"),
            ),
            DEFAULT_LOCAL_BUFFER_BYTES,
        )
        global_segment = _as_int(
            _get_any(
                extra,
                "mooncake_global_segment_bytes",
                "mooncake_global_segment_size",
                "global_segment_size",
                default=os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE")
                or file_defaults.get("global_segment_size"),
            ),
            DEFAULT_GLOBAL_SEGMENT_BYTES,
        )
        protocol = _get_any(
            extra,
            "mooncake_protocol",
            "protocol",
            default=os.getenv("MOONCAKE_PROTOCOL")
            or file_defaults.get("protocol")
            or DEFAULT_PROTOCOL,
        )
        rdma_devices = _get_any(
            extra,
            "mooncake_rdma_devices",
            "mooncake_device",
            "mooncake_device_name",
            "rdma_devices",
            "device_name",
            default=os.getenv("MOONCAKE_DEVICE")
            or file_defaults.get("rdma_devices")
            or DEFAULT_DEVICE_NAME,
        )

        setup_config = {
            "local_hostname": str(local_hostname),
            "metadata_server": str(metadata_server),
            "global_segment_size": int(global_segment),
            "local_buffer_size": int(local_buffer),
            "protocol": str(protocol),
            "rdma_devices": str(rdma_devices or ""),
            "master_server_addr": str(master),
        }
        ipc_socket_path = _get_any(extra, "mooncake_ipc_socket_path", "ipc_socket_path")
        if ipc_socket_path:
            setup_config["ipc_socket_path"] = str(ipc_socket_path)
        enable_ssd = _get_any(extra, "mooncake_enable_ssd_offload", "enable_ssd_offload")
        if enable_ssd is not None:
            setup_config["enable_ssd_offload"] = _as_bool(enable_ssd)
        ssd_path = _get_any(extra, "mooncake_ssd_offload_path", "ssd_offload_path")
        if ssd_path:
            setup_config["ssd_offload_path"] = str(ssd_path)
        return setup_config

    @classmethod
    def _load_config_file_defaults(cls, extra: dict[str, Any]) -> dict[str, Any]:
        config_path = _get_any(
            extra,
            "mooncake_config_path",
            "config_path",
            default=os.getenv("MOONCAKE_CONFIG_PATH"),
        )
        if not config_path:
            return {}
        from mooncake.mooncake_config import MooncakeConfig

        cfg = MooncakeConfig.from_file(str(config_path))
        return {
            "local_hostname": cfg.local_hostname,
            "metadata_server": cfg.metadata_server,
            "global_segment_size": cfg.global_segment_size,
            "local_buffer_size": cfg.local_buffer_size,
            "protocol": cfg.protocol,
            "rdma_devices": cfg.device_name or "",
            "master_server_addr": cfg.master_server_address,
        }

    @classmethod
    def _build_replicate_config(cls, extra: dict[str, Any]) -> Any:
        from mooncake.store import ReplicateConfig

        config = ReplicateConfig()
        config.replica_num = _as_int(
            _get_any(extra, "mooncake_replica_num", "replica_num"),
            int(config.replica_num),
        )
        config.with_soft_pin = _as_bool(
            _get_any(extra, "mooncake_with_soft_pin", "with_soft_pin"),
            bool(config.with_soft_pin),
        )
        config.with_hard_pin = _as_bool(
            _get_any(extra, "mooncake_with_hard_pin", "with_hard_pin"),
            bool(config.with_hard_pin),
        )
        config.prefer_alloc_in_same_node = _as_bool(
            _get_any(
                extra,
                "mooncake_prefer_alloc_in_same_node",
                "prefer_alloc_in_same_node",
            ),
            True,
        )
        preferred_segment = _get_any(
            extra, "mooncake_preferred_segment", "preferred_segment"
        )
        if preferred_segment:
            config.preferred_segment = str(preferred_segment)
        preferred_segments = _get_any(
            extra, "mooncake_preferred_segments", "preferred_segments"
        )
        if preferred_segments:
            if isinstance(preferred_segments, str):
                config.preferred_segments = [
                    item.strip()
                    for item in preferred_segments.split(",")
                    if item.strip()
                ]
            else:
                config.preferred_segments = [str(item) for item in preferred_segments]
        return config

    def put(self, key: VersionedKey, payload: bytes) -> None:
        store_key = key.to_string()
        ret = self._store.upsert(store_key, payload, self._replicate_config)
        if ret != 0:
            raise RuntimeError(f"MooncakeDistributedStore.upsert failed ret={ret}")
        self._known_sizes[store_key] = len(payload)

    def get(self, key: VersionedKey) -> Optional[bytes]:
        if not self.contains(key):
            return None
        payload = self._store.get(key.to_string())
        if payload is None:
            return None
        payload = bytes(payload)
        self._known_sizes[key.to_string()] = len(payload)
        return payload

    def contains(self, key: VersionedKey) -> bool:
        try:
            return int(self._store.is_exist(key.to_string())) == 1
        except Exception:
            return False

    def delete(self, key: VersionedKey) -> bool:
        store_key = key.to_string()
        if not self.contains(key):
            return False
        try:
            ret = self._store.remove(store_key, True)
        except TypeError:
            ret = self._store.remove(store_key)
        if ret == 0:
            self._known_sizes.pop(store_key, None)
            return True
        return False

    def stats(self) -> StoreStats:
        capacity = int(self._setup_config.get("global_segment_size", 0))
        used = sum(self._known_sizes.values())
        return StoreStats(
            capacity_bytes=capacity,
            used_bytes=used,
            free_bytes=max(0, capacity - used),
            entry_count=len(self._known_sizes),
        )

    def health_check(self) -> bool:
        try:
            return int(self._store.health_check()) == 0
        except Exception:
            return False

    def cpu_locations(
        self,
        key: VersionedKey,
        *,
        local_server_id: str | None = None,
    ) -> list[str]:
        """Return local server id or Mooncake endpoint aliases that own `key`.

        ``ReplicateConfig.prefer_alloc_in_same_node=True`` asks Mooncake to
        allocate on the writer's node first. If Mooncake falls back to a remote
        memory segment, ``get_replica_desc`` exposes that segment endpoint; the
        LoadBalancer resolves endpoint/host aliases to rollout server ids.
        """
        try:
            replicas = self._store.get_replica_desc(key.to_string())
        except Exception:
            return [local_server_id] if local_server_id and self.contains(key) else []

        locations: list[str] = []
        local_hostname = str(self._setup_config.get("local_hostname", ""))
        for replica in replicas or []:
            if not _is_memory_replica(replica):
                continue
            for alias in _memory_replica_aliases(replica):
                host = _endpoint_host(alias)
                if local_server_id and (alias == local_hostname or host == local_hostname):
                    locations.append(local_server_id)
                else:
                    locations.append(alias)
                    if host and host != alias:
                        locations.append(host)
        return _dedupe(locations)

    def close(self) -> None:
        close = getattr(self._store, "close", None)
        if close is not None:
            close()


class MooncakeKVStore(KVStore):
    """Compatibility wrapper over a single local TransferEngine buffer.

    Use ``MooncakeKVStore.from_env(..., backend="pooled")`` for the real
    Mooncake distributed L2. This class remains import-compatible with the
    original Plan B local-buffer implementation.
    """

    def __init__(
        self,
        engine,  # mooncake.engine.TransferEngine
        buffer_ptr: int,
        buffer_bytes: int,
    ):
        self._engine = engine
        self._buffer_ptr = buffer_ptr
        self._buffer_bytes = buffer_bytes
        self._index: OrderedDict[VersionedKey, tuple[int, int]] = OrderedDict()
        self._free: list[tuple[int, int]] = [(0, buffer_bytes)]
        self._used_bytes = 0
        self._eviction_count = 0
        self._lock = threading.Lock()

    @classmethod
    def from_env(
        cls,
        buffer_bytes: int = DEFAULT_BUFFER_BYTES,
        device_name: str = DEFAULT_DEVICE_NAME,
        *,
        backend: str = "auto",
        extra_config: dict[str, Any] | None = None,
    ) -> KVStore:
        """Build a store using the same init sequence as verl's MooncakeCheckpointEngine."""
        extra = extra_config or {}
        backend_name = str(backend or "auto").lower()
        if backend_name in _POOLED_BACKENDS or (
            backend_name == "auto" and MooncakePooledKVStore.has_pooled_config(extra)
        ):
            return MooncakePooledKVStore.from_env(extra_config=extra)
        if backend_name not in _LOCAL_BACKENDS and backend_name != "auto":
            raise ValueError(f"unknown Mooncake store backend: {backend!r}")

        import ray
        import torch

        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:  # pragma: no cover - covered by host-only tests
            raise ImportError(
                "MooncakeKVStore requires the `mooncake` Python package. "
                "Install it on the GPU/Mooncake host before using from_env()."
            ) from e

        engine = TransferEngine()
        hostname = ray.util.get_node_ip_address().strip("[]")
        ret = engine.initialize(hostname, "P2PHANDSHAKE", "rdma", device_name)
        if ret != 0:
            raise RuntimeError(f"Mooncake TransferEngine.initialize failed ret={ret}")

        buffer = torch.empty(buffer_bytes, dtype=torch.uint8, device="cpu").pin_memory()
        ret = engine.batch_register_memory([buffer.data_ptr()], [buffer_bytes])
        if ret != 0:
            raise RuntimeError(f"Mooncake batch_register_memory failed ret={ret}")
        # Pin the buffer onto the instance so it isn't garbage-collected.
        store = cls(engine=engine, buffer_ptr=buffer.data_ptr(), buffer_bytes=buffer_bytes)
        store._buffer_keepalive = buffer  # type: ignore[attr-defined]
        return store

    def put(self, key: VersionedKey, payload: bytes) -> None:
        size = len(payload)
        if size > self._buffer_bytes:
            raise ValueError(
                f"payload size {size} exceeds buffer capacity {self._buffer_bytes}"
            )
        with self._lock:
            if key in self._index:
                self._free_entry(key)
            offset = self._allocate(size)
            while offset is None and self._index:
                old_key, _ = next(iter(self._index.items()))
                self._free_entry(old_key)
                self._eviction_count += 1
                offset = self._allocate(size)
            if offset is None:
                raise RuntimeError(
                    f"MooncakeKVStore could not allocate {size} bytes despite "
                    f"{self._buffer_bytes - self._used_bytes} free bytes"
                )
            self._copy_into_buffer(offset, payload)
            self._index[key] = (offset, size)
            self._used_bytes += size

    def get(self, key: VersionedKey) -> Optional[bytes]:
        with self._lock:
            entry = self._index.get(key)
            if entry is None:
                return None
            self._index.move_to_end(key, last=True)
            offset, size = entry
            return self._copy_from_buffer(offset, size)

    def contains(self, key: VersionedKey) -> bool:
        with self._lock:
            return key in self._index

    def delete(self, key: VersionedKey) -> bool:
        with self._lock:
            if key not in self._index:
                return False
            self._free_entry(key)
            return True

    def stats(self) -> StoreStats:
        with self._lock:
            return StoreStats(
                capacity_bytes=self._buffer_bytes,
                used_bytes=self._used_bytes,
                free_bytes=self._buffer_bytes - self._used_bytes,
                entry_count=len(self._index),
                eviction_count=self._eviction_count,
            )

    def close(self) -> None:
        """Best-effort engine shutdown for tests."""
        try:
            shutdown = getattr(self._engine, "shutdown", None)
            if shutdown is not None:
                shutdown()
        except Exception:
            pass

    # -- low-level buffer access ------------------------------------------------

    def _copy_into_buffer(self, offset: int, payload: bytes) -> None:
        import ctypes

        ctypes.memmove(self._buffer_ptr + offset, payload, len(payload))

    def _copy_from_buffer(self, offset: int, size: int) -> bytes:
        import ctypes

        return ctypes.string_at(self._buffer_ptr + offset, size)

    def _allocate(self, size: int) -> int | None:
        for i, (offset, length) in enumerate(self._free):
            if length < size:
                continue
            if length == size:
                del self._free[i]
            else:
                self._free[i] = (offset + size, length - size)
            return offset
        return None

    def _free_entry(self, key: VersionedKey) -> None:
        offset, size = self._index.pop(key)
        self._used_bytes -= size
        self._free.append((offset, size))
        self._coalesce_free()

    def _coalesce_free(self) -> None:
        if not self._free:
            return
        merged: list[tuple[int, int]] = []
        for offset, size in sorted(self._free):
            if not merged:
                merged.append((offset, size))
                continue
            prev_offset, prev_size = merged[-1]
            prev_end = prev_offset + prev_size
            if prev_end == offset:
                merged[-1] = (prev_offset, prev_size + size)
            else:
                merged.append((offset, size))
        self._free = merged


def _dedupe(items: list[str | None]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item:
            continue
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _endpoint_host(endpoint: str) -> str:
    text = str(endpoint)
    parsed = urlparse(text if "://" in text else f"//{text}")
    if parsed.hostname:
        return parsed.hostname.strip("[]")
    return text.split(":", 1)[0].strip("[]")


def _is_memory_replica(replica: Any) -> bool:
    try:
        return bool(replica.is_memory_replica())
    except Exception:
        return False


def _memory_replica_aliases(replica: Any) -> list[str]:
    try:
        memory = replica.get_memory_descriptor()
        descriptor = memory.buffer_descriptor
        endpoint = str(descriptor.transport_endpoint)
    except Exception:
        return []
    return _dedupe([endpoint])
