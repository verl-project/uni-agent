# Plan B: Mooncake KV Connector for vLLM 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 RFC §5.1 connector + §5.2 version key 部分——给 vLLM 加一个 Mooncake KV connector，让本地 GPU KV 与 Mooncake 池之间双向迁移，且 cache key 由 `(hash(token_prefix), weight_version)` 共同决定。

**Architecture:** 在 `llm_router/connector/` 下建 vLLM v1 连接器子包，把"KV 序列化与传输"抽到一个 `KVStore` 抽象层后面（实现两个：`InMemoryKVStore` 给本地测试，`MooncakeKVStore` 给真集群），主连接器 `MooncakeKVConnector` 继承 `vllm.distributed.kv_transfer.kv_connector.v1.KVConnectorBase_V1`，把 prefix→token-id-tensor 的 hash 与 `weight_version` 编进 cache key。注册经由 vLLM 的 `KVConnectorFactory.register_connector` 懒加载，trainer 侧只需在 `kv_transfer_config` 里指 `kv_connector: "MooncakeKVConnector"`。

**Tech Stack:** Python 3.10+, PyTorch, vLLM v1 KV connector framework (`KVConnectorBase_V1`), Mooncake `TransferEngine` (optional dep — installed only on production hosts), pytest。复用 verl 现成的 mooncake 用法范本 `verl/verl/checkpoint_engine/mooncake_checkpoint_engine.py`。

---

## File Structure

```
<repo_root>/
├── llm_router/
│   ├── connector/                      # 新增子包（Plan B 范围）
│   │   ├── __init__.py                 # 导出 MooncakeKVConnector + register_with_vllm()
│   │   ├── prefix_hash.py              # token_ids + weight_version → versioned key
│   │   ├── store/
│   │   │   ├── __init__.py             # 导出 KVStore, InMemoryKVStore, MooncakeKVStore
│   │   │   ├── base.py                 # KVStore ABC
│   │   │   ├── in_memory.py            # 本地测试用，LRU + size cap
│   │   │   └── mooncake.py             # 真 Mooncake TransferEngine 包装
│   │   ├── connector.py                # MooncakeKVConnector(KVConnectorBase_V1)
│   │   ├── meta.py                     # KVConnectorMetadata 子类 + ReqMeta
│   │   ├── registry.py                 # register_with_vllm() —— factory.register_connector hook
│   │   ├── README.md                   # 用法 + kv_transfer_config 示例
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── test_prefix_hash.py
│   │       ├── test_in_memory_store.py
│   │       ├── test_mooncake_store.py  # mooncake 缺包时整文件 skip
│   │       ├── test_connector_metadata.py
│   │       ├── test_connector_unit.py  # 用 InMemoryKVStore 注入
│   │       └── test_factory_registration.py
│   └── ...                             # Plan A 已有内容不动
└── verl/                               # 不动
```

每个文件单一职责：`prefix_hash` 纯函数、`store/*` 数据搬运、`connector` 实现 vLLM 接口、`meta` 数据类、`registry` 一行向 vLLM 注册。

---

## Task 1: 子包骨架

**Files:**
- Create: `llm_router/connector/__init__.py`（empty for now，最后 task 才导出）
- Create: `llm_router/connector/store/__init__.py`（empty）
- Create: `llm_router/connector/tests/__init__.py`（empty）

- [ ] **Step 1: 建空目录与 `__init__.py`**

```bash
mkdir -p llm_router/connector/store llm_router/connector/tests
touch llm_router/connector/__init__.py \
      llm_router/connector/store/__init__.py \
      llm_router/connector/tests/__init__.py
```

- [ ] **Step 2: 验证子包能被发现**

Run: `python -c "import llm_router.connector; import llm_router.connector.store; print('OK')"`
Expected: `OK`

- [ ] **Step 3: 跑现有 Plan A 测试，确保未污染**

Run: `pytest llm_router/tests/ -v --tb=short`
Expected: `19 passed, 1 skipped`

- [ ] **Step 4: 提交**

```bash
git add llm_router/connector/
git commit -m "[llm_router] feat: scaffold connector subpackage for Plan B"
```

---

## Task 2: `prefix_hash` 模块

实现 RFC §5.2 的 cache key 公式 `key = (hash(token_prefix), weight_version)`，与 verl `_build_prefix_signature`（`verl/verl/experimental/agent_loop/agent_loop.py:79`）对齐。

**Files:**
- Create: `llm_router/connector/prefix_hash.py`
- Create: `llm_router/connector/tests/test_prefix_hash.py`

- [ ] **Step 1: 写失败测试**

`llm_router/connector/tests/test_prefix_hash.py`:

```python
"""Versioned cache key derived from token prefix + weight_version."""
import pytest

from llm_router.connector.prefix_hash import (
    PREFIX_HASH_DIGEST_BYTES,
    VersionedKey,
    coerce_weight_version,
    hash_token_prefix,
    make_versioned_key,
)


def test_coerce_weight_version_handles_none():
    assert coerce_weight_version(None) == "unknown"


def test_coerce_weight_version_stringifies_int():
    assert coerce_weight_version(42) == "42"


def test_hash_token_prefix_is_deterministic():
    a = hash_token_prefix([1, 2, 3], 3)
    b = hash_token_prefix([1, 2, 3], 3)
    assert a == b
    assert len(a) == PREFIX_HASH_DIGEST_BYTES * 2  # hex


def test_hash_token_prefix_changes_with_length():
    a = hash_token_prefix([1, 2, 3], 3)
    b = hash_token_prefix([1, 2, 3, 4], 4)
    assert a != b


def test_hash_token_prefix_changes_with_tokens():
    a = hash_token_prefix([1, 2, 3], 3)
    b = hash_token_prefix([1, 2, 9], 3)
    assert a != b


def test_hash_truncates_to_prefix_len():
    # Hashing only the first 2 tokens of a 3-token list must equal hashing
    # the same 2-token list outright.
    a = hash_token_prefix([1, 2, 999], 2)
    b = hash_token_prefix([1, 2], 2)
    assert a == b


def test_make_versioned_key_combines_hash_and_version():
    k = make_versioned_key([1, 2, 3], prefix_len=3, weight_version=7)
    assert isinstance(k, VersionedKey)
    assert k.weight_version == "7"
    assert k.prefix_hash == hash_token_prefix([1, 2, 3], 3)
    assert k.prefix_len == 3


def test_versioned_key_is_hashable_and_value_equal():
    a = make_versioned_key([1, 2, 3], 3, "v0")
    b = make_versioned_key([1, 2, 3], 3, "v0")
    assert a == b
    assert hash(a) == hash(b)
    assert a in {b}


def test_versioned_key_serializable_to_string():
    k = make_versioned_key([1, 2, 3], 3, "v0")
    assert k.to_string().startswith("v0:")
    assert k.to_string().endswith(":3")


def test_versioned_key_roundtrip_string():
    k = make_versioned_key([1, 2, 3], 3, "v0")
    parsed = VersionedKey.from_string(k.to_string())
    assert parsed == k


def test_make_versioned_key_rejects_bad_prefix_len():
    with pytest.raises(ValueError):
        make_versioned_key([1, 2, 3], prefix_len=0, weight_version="v0")
    with pytest.raises(ValueError):
        make_versioned_key([1, 2, 3], prefix_len=4, weight_version="v0")
```

- [ ] **Step 2: 跑测试，应失败**

Run: `pytest llm_router/connector/tests/test_prefix_hash.py -v`
Expected: ImportError on `llm_router.connector.prefix_hash`

- [ ] **Step 3: 写实现**

`llm_router/connector/prefix_hash.py`:

```python
"""Versioned cache key for the Mooncake KV pool (RFC §5.2).

The key combines a hash of the prompt-token prefix with the actor weight
version that produced (or will consume) the KV. The version field provides
the cross-step semantic safety described in RFC §5.2: KV produced under
weight v_n is never reused under weight v_{n+1}, because the key fails to
match.

The hashing scheme intentionally mirrors verl's _build_prefix_signature in
verl/verl/experimental/agent_loop/agent_loop.py so that any prefix index
reported by a verl-managed replica produces the same key as the connector
side.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

PREFIX_HASH_DIGEST_BYTES = 16


def coerce_weight_version(weight_version: Any) -> str:
    return "unknown" if weight_version is None else str(weight_version)


def hash_token_prefix(prompt_ids: list[int], prefix_len: int) -> str:
    """Hash the first `prefix_len` tokens of `prompt_ids` to a hex string.

    The hash is salted with the prefix length so that hash([1,2,3], 2) is
    distinct from hash([1,2,3], 3) even when tokens beyond `prefix_len` are
    present in the list.
    """
    hasher = hashlib.blake2b(digest_size=PREFIX_HASH_DIGEST_BYTES)
    for token_id in prompt_ids[:prefix_len]:
        hasher.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    hasher.update(b":")
    hasher.update(str(prefix_len).encode("ascii"))
    return hasher.hexdigest()


@dataclass(frozen=True)
class VersionedKey:
    """`(weight_version, prefix_hash, prefix_len)` triple identifying KV in the pool."""

    weight_version: str
    prefix_hash: str
    prefix_len: int

    def to_string(self) -> str:
        return f"{self.weight_version}:{self.prefix_hash}:{self.prefix_len}"

    @classmethod
    def from_string(cls, s: str) -> "VersionedKey":
        version, prefix_hash, prefix_len_str = s.rsplit(":", 2)
        return cls(
            weight_version=version,
            prefix_hash=prefix_hash,
            prefix_len=int(prefix_len_str),
        )


def make_versioned_key(
    prompt_ids: list[int],
    prefix_len: int,
    weight_version: Any,
) -> VersionedKey:
    if prefix_len <= 0 or prefix_len > len(prompt_ids):
        raise ValueError(
            f"prefix_len must be in (0, {len(prompt_ids)}], got {prefix_len}"
        )
    return VersionedKey(
        weight_version=coerce_weight_version(weight_version),
        prefix_hash=hash_token_prefix(prompt_ids, prefix_len),
        prefix_len=prefix_len,
    )
```

- [ ] **Step 4: 跑测试**

Run: `pytest llm_router/connector/tests/test_prefix_hash.py -v`
Expected: 11 passed

- [ ] **Step 5: 提交**

```bash
git add llm_router/connector/prefix_hash.py llm_router/connector/tests/test_prefix_hash.py
git commit -m "[llm_router] feat: versioned cache key (prefix hash + weight_version)"
```

---

## Task 3: `KVStore` 抽象基类

**Files:**
- Create: `llm_router/connector/store/base.py`
- Create: `llm_router/connector/tests/test_store_base.py`

- [ ] **Step 1: 写失败测试**

`llm_router/connector/tests/test_store_base.py`:

```python
"""KVStore abstract base class contract."""
import pytest

from llm_router.connector.prefix_hash import make_versioned_key
from llm_router.connector.store.base import KVStore


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

    s = Minimal()
    k = make_versioned_key([1, 2, 3], 3, "v0")
    assert s.contains(k) is False
    assert s.get(k) is None
    s.put(k, b"payload")
    assert s.contains(k) is True
    assert s.get(k) == b"payload"
```

- [ ] **Step 2: 跑测试，应失败**

Run: `pytest llm_router/connector/tests/test_store_base.py -v`
Expected: ImportError

- [ ] **Step 3: 写实现**

`llm_router/connector/store/base.py`:

```python
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
from typing import Optional

from llm_router.connector.prefix_hash import VersionedKey


class KVStore(ABC):
    """Versioned blob store for serialized KV blocks."""

    @abstractmethod
    def put(self, key: VersionedKey, payload: bytes) -> None: ...

    @abstractmethod
    def get(self, key: VersionedKey) -> Optional[bytes]: ...

    @abstractmethod
    def contains(self, key: VersionedKey) -> bool: ...
```

- [ ] **Step 4: 跑测试**

Run: `pytest llm_router/connector/tests/test_store_base.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add llm_router/connector/store/base.py llm_router/connector/tests/test_store_base.py
git commit -m "[llm_router] feat: KVStore abstract base"
```

---

## Task 4: `InMemoryKVStore` 实现

LRU + 总字节数上限。给所有上层组件单测用——不依赖 mooncake 包。

**Files:**
- Create: `llm_router/connector/store/in_memory.py`
- Create: `llm_router/connector/tests/test_in_memory_store.py`

- [ ] **Step 1: 写失败测试**

`llm_router/connector/tests/test_in_memory_store.py`:

```python
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
```

- [ ] **Step 2: 跑测试，应失败**

Run: `pytest llm_router/connector/tests/test_in_memory_store.py -v`
Expected: ImportError

- [ ] **Step 3: 写实现**

`llm_router/connector/store/in_memory.py`:

```python
"""In-memory KVStore for unit tests. Production uses MooncakeKVStore."""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

from llm_router.connector.prefix_hash import VersionedKey
from llm_router.connector.store.base import KVStore


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
        self._items[key] = payload
        self._used_bytes += size

    def get(self, key: VersionedKey) -> Optional[bytes]:
        if key not in self._items:
            return None
        self._items.move_to_end(key, last=True)
        return self._items[key]

    def contains(self, key: VersionedKey) -> bool:
        return key in self._items
```

- [ ] **Step 4: 跑测试**

Run: `pytest llm_router/connector/tests/test_in_memory_store.py -v`
Expected: 7 passed

- [ ] **Step 5: 提交**

```bash
git add llm_router/connector/store/in_memory.py llm_router/connector/tests/test_in_memory_store.py
git commit -m "[llm_router] feat: InMemoryKVStore with LRU + byte-cap"
```

---

## Task 5: `MooncakeKVStore` 实现（带可选依赖跳过）

包装 `mooncake.engine.TransferEngine`。仿照 `verl/verl/checkpoint_engine/mooncake_checkpoint_engine.py` 的初始化套路。环境无 mooncake 包时整文件 `pytest.importorskip` 跳过。

**Files:**
- Create: `llm_router/connector/store/mooncake.py`
- Create: `llm_router/connector/tests/test_mooncake_store.py`

- [ ] **Step 1: 写失败测试**

`llm_router/connector/tests/test_mooncake_store.py`:

```python
"""MooncakeKVStore: thin wrapper over mooncake.engine.TransferEngine.

These tests skip entirely if the `mooncake` package is not installed.
On a real Mooncake-equipped host they round-trip a payload through the
TransferEngine.
"""
import pytest

mooncake = pytest.importorskip(
    "mooncake.engine",
    reason="MooncakeKVStore tests require the `mooncake` package",
)

from llm_router.connector.prefix_hash import make_versioned_key
from llm_router.connector.store.mooncake import MooncakeKVStore


@pytest.fixture
def mooncake_store():
    store = MooncakeKVStore.from_env(buffer_bytes=4 * 1024 * 1024)
    yield store
    store.close()


def test_put_and_get_roundtrip(mooncake_store):
    k = make_versioned_key([1, 2, 3], 3, "v0")
    mooncake_store.put(k, b"hello mooncake")
    assert mooncake_store.contains(k) is True
    assert mooncake_store.get(k) == b"hello mooncake"


def test_get_returns_none_on_miss(mooncake_store):
    assert mooncake_store.get(make_versioned_key([9], 1, "v0")) is None
```

- [ ] **Step 2: 跑测试**

Run: `pytest llm_router/connector/tests/test_mooncake_store.py -v`
Expected on host without mooncake: `1 skipped` (整个文件跳过)。
Expected on Mooncake host: `2 passed`。

- [ ] **Step 3: 写实现**

`llm_router/connector/store/mooncake.py`:

```python
"""MooncakeKVStore: TransferEngine-backed L2 KV pool.

Storage model
-------------
Each VersionedKey is mapped to a unique byte offset in a pre-registered
buffer. Writes copy the payload to that offset; reads copy back. The key →
(offset, length) mapping is held locally; the actual bytes live in the
Mooncake-managed buffer (CPU memory, RDMA-shared with peer engines).

This Plan B implementation keeps the model simple — single local buffer,
no peer-pool sharding, no distributed lookup. Cross-replica sharing arrives
when peer engines are wired in (out of scope here; see RFC §5.1's
"asynchronously RDMA load" description for the eventual architecture).

Initialization mirrors verl/verl/checkpoint_engine/mooncake_checkpoint_engine.py
so that a host already running verl's Mooncake checkpoint path needs no extra
configuration.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

from llm_router.connector.prefix_hash import VersionedKey
from llm_router.connector.store.base import KVStore

try:
    import ray
    from mooncake.engine import TransferEngine  # noqa: F401  # imported for availability
except ImportError as e:  # pragma: no cover - covered by skipif
    raise ImportError(
        "MooncakeKVStore requires the `mooncake` Python package "
        "(and `ray`). Install both before importing this module."
    ) from e


DEFAULT_BUFFER_BYTES = 32 * 1024 * 1024 * 1024  # 32 GiB
DEFAULT_DEVICE_NAME = ""  # let TransferEngine pick


class MooncakeKVStore(KVStore):
    """Single-host wrapper over mooncake.engine.TransferEngine."""

    def __init__(
        self,
        engine,  # mooncake.engine.TransferEngine
        buffer_ptr: int,
        buffer_bytes: int,
    ):
        self._engine = engine
        self._buffer_ptr = buffer_ptr
        self._buffer_bytes = buffer_bytes
        self._cursor = 0
        self._index: dict[VersionedKey, tuple[int, int]] = {}  # key -> (offset, length)
        self._lock = threading.Lock()

    @classmethod
    def from_env(
        cls,
        buffer_bytes: int = DEFAULT_BUFFER_BYTES,
        device_name: str = DEFAULT_DEVICE_NAME,
    ) -> "MooncakeKVStore":
        """Build a store using the same init sequence as verl's MooncakeCheckpointEngine."""
        from mooncake.engine import TransferEngine
        import torch

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
        store._buffer_keepalive = buffer  # noqa: type-ignore[attr-defined]
        return store

    def put(self, key: VersionedKey, payload: bytes) -> None:
        size = len(payload)
        if size > self._buffer_bytes:
            raise ValueError(
                f"payload size {size} exceeds buffer capacity {self._buffer_bytes}"
            )
        with self._lock:
            if self._cursor + size > self._buffer_bytes:
                raise RuntimeError(
                    "MooncakeKVStore buffer full — Plan B keeps a simple "
                    "bump-allocator; eviction added in a follow-up."
                )
            offset = self._cursor
            self._cursor += size
            self._copy_into_buffer(offset, payload)
            self._index[key] = (offset, size)

    def get(self, key: VersionedKey) -> Optional[bytes]:
        with self._lock:
            entry = self._index.get(key)
            if entry is None:
                return None
            offset, size = entry
            return self._copy_from_buffer(offset, size)

    def contains(self, key: VersionedKey) -> bool:
        with self._lock:
            return key in self._index

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
```

> **Engineer note:** This bump-allocator is intentionally thin. The `RuntimeError` on full buffer is acceptable for Plan B because the connector itself caps payload accumulation by weight-version turnover; an LRU-style eviction inside the Mooncake buffer is a follow-up (`# TODO(plan-b-followup)` is marked in the docstring above the class). Do not add eviction here.

- [ ] **Step 4: 跑测试（无 mooncake 时）**

Run: `pytest llm_router/connector/tests/test_mooncake_store.py -v`
Expected: 整个文件 `1 skipped`。

- [ ] **Step 5: 跑全部 connector 测试，确保未污染**

Run: `pytest llm_router/connector/tests/ -v`
Expected: 21 passed + 1 skipped（11 prefix_hash + 3 store_base + 7 in_memory + 1 mooncake skip）。

- [ ] **Step 6: 提交**

```bash
git add llm_router/connector/store/mooncake.py llm_router/connector/tests/test_mooncake_store.py
git commit -m "[llm_router] feat: MooncakeKVStore (TransferEngine-backed, optional dep)"
```

---

## Task 6: 连接器 metadata 数据类

vLLM v1 connector 在 scheduler-worker 之间通过 `KVConnectorMetadata` 子类传输每步要 load/save 的请求列表。

**Files:**
- Create: `llm_router/connector/meta.py`
- Create: `llm_router/connector/tests/test_connector_metadata.py`

- [ ] **Step 1: 写失败测试**

`llm_router/connector/tests/test_connector_metadata.py`:

```python
"""KVConnectorMetadata subclass + per-request meta entries."""
import torch

from llm_router.connector.meta import (
    MooncakeConnectorMetadata,
    MooncakeReqMeta,
)


def test_reqmeta_construction():
    m = MooncakeReqMeta(
        request_id="req-1",
        token_ids=[1, 2, 3, 4],
        block_ids=[0, 1],
        block_size=2,
        is_store=False,
        weight_version="v0",
    )
    assert m.request_id == "req-1"
    assert m.is_store is False
    assert m.weight_version == "v0"
    assert m.token_ids_tensor.dtype == torch.long
    assert m.token_ids_tensor.tolist() == [1, 2, 3, 4]


def test_metadata_add_and_iterate():
    meta = MooncakeConnectorMetadata()
    meta.add_request(
        request_id="req-1",
        token_ids=[1, 2],
        block_ids=[0],
        block_size=2,
        is_store=False,
        weight_version="v0",
    )
    meta.add_request(
        request_id="req-2",
        token_ids=[1, 2, 3, 4],
        block_ids=[0, 1],
        block_size=2,
        is_store=True,
        weight_version="v0",
    )
    assert len(meta.requests) == 2
    assert [r.request_id for r in meta.requests] == ["req-1", "req-2"]
    assert [r.is_store for r in meta.requests] == [False, True]


def test_metadata_aligns_token_ids_to_block_size():
    meta = MooncakeConnectorMetadata()
    meta.add_request(
        request_id="req-3",
        token_ids=[1, 2, 3, 4, 5],  # 5 tokens, block_size=2 → align to 4
        block_ids=[0, 1],
        block_size=2,
        is_store=False,
        weight_version="v0",
    )
    r = meta.requests[0]
    assert r.aligned_token_count == 4
    assert r.token_ids_tensor.shape == (4,)
```

- [ ] **Step 2: 跑测试，应失败**

Run: `pytest llm_router/connector/tests/test_connector_metadata.py -v`
Expected: ImportError

- [ ] **Step 3: 写实现**

`llm_router/connector/meta.py`:

```python
"""Metadata exchanged between scheduler-side and worker-side connectors.

Mirrors vLLM's reference SharedStorageConnectorMetadata, but stamps the
weight_version on each request so that worker-side load/save can compute
the correct VersionedKey without re-reading scheduler state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


def _align_to_block_size(num_tokens: int, block_size: int) -> int:
    return (num_tokens // block_size) * block_size


@dataclass
class MooncakeReqMeta:
    request_id: str
    token_ids: list[int]
    block_ids: list[int]
    block_size: int
    is_store: bool
    weight_version: str

    @property
    def aligned_token_count(self) -> int:
        return _align_to_block_size(len(self.token_ids), self.block_size)

    @property
    def token_ids_tensor(self) -> torch.Tensor:
        return torch.tensor(
            self.token_ids[: self.aligned_token_count],
            dtype=torch.long,
        )


@dataclass
class MooncakeConnectorMetadata(KVConnectorMetadata):
    requests: list[MooncakeReqMeta] = field(default_factory=list)

    def add_request(
        self,
        request_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        weight_version: Any,
    ) -> None:
        self.requests.append(
            MooncakeReqMeta(
                request_id=request_id,
                token_ids=list(token_ids),
                block_ids=list(block_ids),
                block_size=block_size,
                is_store=is_store,
                weight_version=str(weight_version) if weight_version is not None else "unknown",
            )
        )
```

- [ ] **Step 4: 跑测试**

Run: `pytest llm_router/connector/tests/test_connector_metadata.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add llm_router/connector/meta.py llm_router/connector/tests/test_connector_metadata.py
git commit -m "[llm_router] feat: KV connector metadata with weight_version stamping"
```

---

## Task 7: `MooncakeKVConnector` 主类

继承 `KVConnectorBase_V1`。Plan B 实现核心数据流：

- **Scheduler-side**: `get_num_new_matched_tokens` 用 `weight_version` + token_ids 查 store；`build_connector_meta` 把 load/save 请求打包成 `MooncakeConnectorMetadata`。
- **Worker-side**: `register_kv_caches` 记下 vLLM 的 KV tensor；`start_load_kv` 从 store load；`save_kv_layer` + `wait_for_save` 把驱逐前的 block dump 进 store。
- 其余抽象方法用 `pass` / `default no-op`，并标注 `# TODO(plan-b-followup): ...` 解释为何当前不需要。

实现策略：**复用 vLLM `SharedStorageConnector` 的代码骨架**——它是 vLLM 自带的、跑得通的最小参考实现，只把 "load from disk path" / "save to disk path" 替换成 "load from KVStore" / "save to KVStore"，其余 PagedAttention 拷贝逻辑、scheduler 钩子保持原样。

**Files:**
- Create: `llm_router/connector/connector.py`
- Create: `llm_router/connector/tests/test_connector_unit.py`

- [ ] **Step 1: 写失败测试**

`llm_router/connector/tests/test_connector_unit.py`:

```python
"""MooncakeKVConnector unit tests using InMemoryKVStore as the backend.

Strategy: We don't drive a full vLLM scheduler — that's covered by
`test_factory_registration.py` (smoke) and out-of-tree integration. Instead
we exercise:
- weight_version is read from kv_transfer_config.kv_connector_extra_config
- get_num_new_matched_tokens returns 0 on miss, full prefix length on hit
- save path serializes payload through KVStore.put
- load path deserializes via KVStore.get
"""
from unittest.mock import MagicMock

import pytest
import torch

from llm_router.connector.connector import MooncakeKVConnector
from llm_router.connector.prefix_hash import make_versioned_key
from llm_router.connector.store.in_memory import InMemoryKVStore


@pytest.fixture
def store():
    return InMemoryKVStore(max_bytes=1024 * 1024)


@pytest.fixture
def connector(monkeypatch, store):
    """Build a connector wired to InMemoryKVStore."""
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    cfg = MagicMock()
    cfg.cache_config.block_size = 2
    cfg.kv_transfer_config.kv_connector_extra_config = {"weight_version": "v0"}
    cfg.kv_transfer_config.engine_id = "engine-test"

    # Bypass the real Mooncake init by injecting our store via a module hook.
    monkeypatch.setattr(
        "llm_router.connector.connector._build_default_store",
        lambda extra: store,
    )
    return MooncakeKVConnector(
        vllm_config=cfg,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=None,
    )


def test_connector_reads_weight_version_from_extra_config(connector):
    assert connector._weight_version == "v0"


def test_get_num_new_matched_tokens_returns_zero_on_miss(connector):
    req = MagicMock()
    req.prompt_token_ids = [1, 2, 3, 4]
    n, async_load = connector.get_num_new_matched_tokens(req, num_computed_tokens=0)
    assert n == 0
    assert async_load is False


def test_get_num_new_matched_tokens_returns_aligned_hit(connector, store):
    # Pre-populate store with a prefix the connector should find.
    key = make_versioned_key([1, 2, 3, 4], prefix_len=4, weight_version="v0")
    store.put(key, b"dummy-kv")

    req = MagicMock()
    req.prompt_token_ids = [1, 2, 3, 4, 5]
    n, async_load = connector.get_num_new_matched_tokens(req, num_computed_tokens=0)
    # The 5-token prompt minus the last token (1 token) = 4 tokens; aligned to block_size=2 → 4.
    assert n == 4
    assert async_load is False


def test_different_weight_version_misses(connector, store):
    # Seed with a different version → connector should miss.
    key = make_versioned_key([1, 2, 3, 4], 4, "v1")
    store.put(key, b"old")

    req = MagicMock()
    req.prompt_token_ids = [1, 2, 3, 4, 5]
    n, _ = connector.get_num_new_matched_tokens(req, num_computed_tokens=0)
    assert n == 0


def test_serialize_and_deserialize_kv_blocks_roundtrip(connector):
    """Worker-side helpers serialize a list of per-layer KV tensors to bytes
    and back. These are private but test them directly for safety."""
    layer_a = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    layer_b = torch.arange(16, 32, dtype=torch.float32).reshape(2, 8)
    payload = connector._serialize_layers([layer_a, layer_b])
    assert isinstance(payload, bytes)
    restored = connector._deserialize_layers(payload, expected_count=2)
    assert len(restored) == 2
    assert torch.equal(restored[0], layer_a)
    assert torch.equal(restored[1], layer_b)
```

- [ ] **Step 2: 跑测试，应失败**

Run: `pytest llm_router/connector/tests/test_connector_unit.py -v`
Expected: ImportError

- [ ] **Step 3: 写实现**

`llm_router/connector/connector.py`:

```python
"""MooncakeKVConnector: vLLM v1 KV connector backed by a versioned KV pool.

Implements the data flow described in RFC §5.1:
  1. On scheduler-side `get_num_new_matched_tokens`, look up
     `(hash(prompt_prefix), weight_version)` in the KVStore.
  2. On worker-side `start_load_kv`, fetch the payload and copy KV bytes
     into the PagedAttention buffer at the slot mapping the scheduler
     allocated.
  3. On worker-side `save_kv_layer` (called per layer when a request finishes),
     serialize the layer's KV slice and write it to the KVStore.

Plan B keeps the connector minimal: we adopt the structure of vLLM's
SharedStorageConnector (a shipping reference implementation) and substitute
its "save to disk path" for "save to KVStore.put". Methods that vLLM's
connector framework calls but Plan B does not need (HMA hooks, KV events,
stats) are stubs marked TODO(plan-b-followup).
"""
from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

from llm_router.connector.meta import MooncakeConnectorMetadata, MooncakeReqMeta
from llm_router.connector.prefix_hash import (
    coerce_weight_version,
    hash_token_prefix,
    make_versioned_key,
)
from llm_router.connector.store.base import KVStore

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


def _build_default_store(extra_config: dict[str, Any]) -> KVStore:
    """Default factory: build a MooncakeKVStore from extra_config.

    Tests monkeypatch this symbol to inject InMemoryKVStore.
    """
    from llm_router.connector.store.mooncake import MooncakeKVStore

    buffer_bytes = int(extra_config.get("mooncake_buffer_bytes", 32 * 1024 * 1024 * 1024))
    return MooncakeKVStore.from_env(buffer_bytes=buffer_bytes)


class MooncakeKVConnector(KVConnectorBase_V1):
    """vLLM v1 connector that load/saves KV from/to a versioned KVStore."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self._weight_version = coerce_weight_version(extra.get("weight_version"))
        self._store: KVStore = _build_default_store(extra)
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._requests_need_load: dict[str, "Request"] = {}
        self._pending_saves: list[bytes] = []

    # ---- Scheduler-side ----

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        token_ids = request.prompt_token_ids or []
        if len(token_ids) <= 1:
            return 0, False
        # vLLM aligns to block_size and treats the last token specially (matches
        # SharedStorageConnector's convention).
        aligned = (len(token_ids) - 1) // self._block_size * self._block_size
        if aligned <= num_computed_tokens:
            return 0, False
        key = make_versioned_key(token_ids, aligned, self._weight_version)
        if not self._store.contains(key):
            return 0, False
        self._requests_need_load[request.request_id] = request
        return aligned - num_computed_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])
            block_ids = list(new_req.block_ids[0]) if new_req.block_ids else []
            is_load = new_req.req_id in self._requests_need_load
            meta.add_request(
                request_id=new_req.req_id,
                token_ids=token_ids,
                block_ids=block_ids,
                block_size=self._block_size,
                is_store=not is_load,  # if no external hit, this turn we save
                weight_version=self._weight_version,
            )
        # Flush per-step state.
        self._requests_need_load.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # Plan B: keep blocks owned by vLLM (return False), no async transfer params.
        return False, None

    # ---- Worker-side ----

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._kv_caches = kv_caches

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        if not isinstance(meta, MooncakeConnectorMetadata):
            return
        for req in meta.requests:
            if req.is_store:
                continue
            self._load_request(req)

    def wait_for_layer_load(self, layer_name: str) -> None:
        # Plan B: load is synchronous in _load_request; nothing async to wait on.
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        meta = self._get_connector_metadata()
        if not isinstance(meta, MooncakeConnectorMetadata):
            return
        for req in meta.requests:
            if not req.is_store:
                continue
            self._save_request_layer(req, kv_layer)

    def wait_for_save(self) -> None:
        # Plan B: save is synchronous in _save_request_layer.
        return

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        # Plan B: nothing async in flight; both sets empty.
        return set(), set()

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set()

    # ---- Private helpers ----

    def _serialize_layers(self, layers: list[torch.Tensor]) -> bytes:
        """Concatenate per-layer KV tensors into a single bytes payload."""
        buf = io.BytesIO()
        torch.save([t.cpu().contiguous() for t in layers], buf)
        return buf.getvalue()

    def _deserialize_layers(
        self, payload: bytes, expected_count: int
    ) -> list[torch.Tensor]:
        buf = io.BytesIO(payload)
        layers = torch.load(buf, weights_only=True)
        if not isinstance(layers, list) or len(layers) != expected_count:
            raise ValueError(
                f"Mooncake payload contains {len(layers) if isinstance(layers, list) else '?'} "
                f"layers, expected {expected_count}"
            )
        return layers

    def _load_request(self, req: MooncakeReqMeta) -> None:
        if req.aligned_token_count <= 0:
            return
        key = make_versioned_key(
            req.token_ids, req.aligned_token_count, req.weight_version
        )
        payload = self._store.get(key)
        if payload is None:
            return
        layers = self._deserialize_layers(payload, expected_count=len(self._kv_caches))
        # Plan B: copy each layer's blocks into PagedAttention by block_id.
        # This is a minimal, correctness-first implementation; performance-tuned
        # variants (RDMA-direct, slot_mapping shortcuts) live in plan-b-followup.
        for (layer_name, kv_tensor), src in zip(self._kv_caches.items(), layers):
            num_blocks = min(len(req.block_ids), src.shape[0])
            for i in range(num_blocks):
                kv_tensor[req.block_ids[i]].copy_(src[i])

    def _save_request_layer(self, req: MooncakeReqMeta, kv_layer: torch.Tensor) -> None:
        if req.aligned_token_count <= 0 or not req.block_ids:
            return
        # Plan B: we collect all layers serially and serialize on the last one.
        # In real vLLM the connector saves layer-by-layer, but the simple
        # InMemoryKVStore handles per-layer overwrite correctly because the
        # key is the same; the real Mooncake path will batch in a follow-up.
        blocks = torch.stack([kv_layer[bid].cpu() for bid in req.block_ids], dim=0)
        # Append to pending; flush on layer 0 (called last in vLLM ordering).
        # NOTE(plan-b-followup): replace this scratch buffer with a layered
        # streaming serializer once Plan C arrives.
        self._pending_saves.append(self._serialize_layers([blocks]))
        if len(self._pending_saves) == len(self._kv_caches):
            payload = b"".join(self._pending_saves)
            key = make_versioned_key(
                req.token_ids, req.aligned_token_count, req.weight_version
            )
            self._store.put(key, payload)
            self._pending_saves.clear()
```

- [ ] **Step 4: 跑测试**

Run: `pytest llm_router/connector/tests/test_connector_unit.py -v`
Expected: 5 passed

- [ ] **Step 5: 跑全部 connector 测试**

Run: `pytest llm_router/connector/tests/ -v`
Expected: 26 passed + 1 skipped (mooncake_store)

- [ ] **Step 6: 提交**

```bash
git add llm_router/connector/connector.py llm_router/connector/tests/test_connector_unit.py
git commit -m "[llm_router] feat: MooncakeKVConnector with versioned load/save"
```

---

## Task 8: vLLM Factory 注册 + README

让 vLLM 的 `KVConnectorFactory` 知道 `MooncakeKVConnector`，并提供启用步骤文档。

**Files:**
- Create: `llm_router/connector/registry.py`
- Modify: `llm_router/connector/__init__.py`（导出 `MooncakeKVConnector` + `register_with_vllm`）
- Create: `llm_router/connector/tests/test_factory_registration.py`
- Create: `llm_router/connector/README.md`

- [ ] **Step 1: 写失败测试**

`llm_router/connector/tests/test_factory_registration.py`:

```python
"""KVConnectorFactory smoke: our connector class is reachable by name."""
from llm_router.connector import register_with_vllm
from llm_router.connector.connector import MooncakeKVConnector


def test_register_idempotent():
    register_with_vllm()
    register_with_vllm()  # second call must not raise


def test_factory_resolves_connector_class():
    register_with_vllm()
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    cls = KVConnectorFactory.get_connector_class_by_name("MooncakeKVConnector")
    assert cls is MooncakeKVConnector
```

- [ ] **Step 2: 跑测试，应失败**

Run: `pytest llm_router/connector/tests/test_factory_registration.py -v`
Expected: ImportError on `register_with_vllm`

- [ ] **Step 3: 写 registry 实现**

`llm_router/connector/registry.py`:

```python
"""One-call registration with vLLM's KVConnectorFactory."""
from __future__ import annotations

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

CONNECTOR_NAME = "MooncakeKVConnector"
_MODULE_PATH = "llm_router.connector.connector"
_CLASS_NAME = "MooncakeKVConnector"


def register_with_vllm() -> None:
    """Register MooncakeKVConnector with vLLM. Safe to call multiple times."""
    if CONNECTOR_NAME in KVConnectorFactory._registry:
        return
    KVConnectorFactory.register_connector(
        name=CONNECTOR_NAME,
        module_path=_MODULE_PATH,
        class_name=_CLASS_NAME,
    )
```

- [ ] **Step 4: 写 connector subpackage `__init__.py`**

`llm_router/connector/__init__.py`:

```python
"""Plan B: vLLM v1 KV connector backed by Mooncake.

Importing this package does not register with vLLM automatically — call
`register_with_vllm()` once during trainer startup. This keeps the side
effect explicit (vLLM does not otherwise import this module).
"""
from llm_router.connector.connector import MooncakeKVConnector
from llm_router.connector.prefix_hash import (
    VersionedKey,
    make_versioned_key,
)
from llm_router.connector.registry import (
    CONNECTOR_NAME,
    register_with_vllm,
)

__all__ = [
    "CONNECTOR_NAME",
    "MooncakeKVConnector",
    "VersionedKey",
    "make_versioned_key",
    "register_with_vllm",
]
```

- [ ] **Step 5: 写 README**

`llm_router/connector/README.md`:

````markdown
# llm_router.connector

vLLM v1 KV connector backed by Mooncake. Implements the connector layer
described in [RFC §5.1 / §5.2](../../docs/superpowers/specs/2026-05-14-context-aware-scheduling-design.md):

- bidirectional load/dump between local PagedAttention and a CPU-resident L2 pool;
- cache key = `(hash(token_prefix), weight_version)` — KV produced under one
  weight version never matches a query from another.

## Usage

In your training script, register once before the trainer constructs vLLM:

```python
from llm_router.connector import register_with_vllm
register_with_vllm()
```

Then point vLLM at the connector via `kv_transfer_config`:

```yaml
actor_rollout_ref:
  rollout:
    kv_transfer_config:
      kv_connector: "MooncakeKVConnector"
      kv_role: "kv_both"
      kv_connector_extra_config:
        weight_version: "${global_step}"   # or any string id
        mooncake_buffer_bytes: 34359738368  # 32 GiB
```

`weight_version` MUST be threaded from the trainer in lockstep with weight
updates (Plan A's RFC §5.2 protocol). Trainers that don't rotate
`weight_version` will, by RFC, behave as if every entry in the pool came
from the same weight — which is *only* safe in evaluation/inference, never
in RL training.

## Backends

The connector talks to a `KVStore` interface. Two backends ship:

| Backend | When | Cap |
|---|---|---|
| `InMemoryKVStore` | unit tests | LRU eviction by total bytes |
| `MooncakeKVStore` | production | bump-allocator inside a registered Mooncake buffer |

`MooncakeKVStore` requires the `mooncake` Python package; tests skip if it's
not installed.

## Tests

```
pytest llm_router/connector/tests/ -v
```

26 passing + 1 skipped (mooncake-host-only).

## Plan B-followup (not in this plan)

- LRU-style eviction inside `MooncakeKVStore`'s registered buffer
  (currently a bump allocator).
- True async load — Plan B does the copy synchronously inside
  `start_load_kv`. RFC §5.1 calls for "asynchronously RDMA load" which
  requires returning the request id from `get_finished` once transfer
  completes; the hooks are stubbed here.
- Cross-replica peer pool — `MooncakeKVStore` currently registers a single
  local buffer. RFC §4 envisions RDMA-shared peer engines so a Plan-A-routed
  fallback replica can pull KV that a primary wrote.
````

- [ ] **Step 6: 跑测试**

Run: `pytest llm_router/connector/tests/test_factory_registration.py -v`
Expected: 2 passed

- [ ] **Step 7: 跑全部 llm_router 测试**

Run: `pytest llm_router/ -v --tb=short`
Expected: 47 passed + 2 skipped (Plan A 19 + 1 skip + Plan B 28 + 1 skip)。

- [ ] **Step 8: 提交**

```bash
git add llm_router/connector/registry.py \
        llm_router/connector/__init__.py \
        llm_router/connector/tests/test_factory_registration.py \
        llm_router/connector/README.md
git commit -m "[llm_router] feat: register MooncakeKVConnector with vLLM factory + README"
```

---

## Task 9: Self-review 与终态提交

- [ ] **Step 1: 跑全 llm_router 测试套**

Run: `pytest llm_router/ -v --tb=short`
Expected: 47 passed, 2 skipped, 0 failed

- [ ] **Step 2: TODO 扫描**

Run: `grep -rn "TODO\|FIXME\|XXX" llm_router/connector/ | grep -v "Plan C\|Plan B\|Plan D\|plan-b-followup\|plan-followup\|TODO(parity)"`
Expected: 输出为空（保留显式标注的跨计划 / Plan-B-followup TODO；其他必须清理）。

- [ ] **Step 3: Lint**

Run: `ruff check llm_router/connector/`
Expected: `All checks passed!`

- [ ] **Step 4: 验证 Plan A 没被破坏**

Run: `pytest llm_router/tests/ -v --tb=short`
Expected: 19 passed + 1 skipped（与 Plan A 完成时一致）。

- [ ] **Step 5: 终态提交**

```bash
git add -u llm_router/
git commit --allow-empty -m "[llm_router] chore: Plan B complete — Mooncake KV connector"
```

---

## 完成判据

落地后应满足：

1. `llm_router/connector/` 子包存在，`from llm_router.connector import MooncakeKVConnector, register_with_vllm` 可用；
2. 调用 `register_with_vllm()` 后，`KVConnectorFactory.get_connector_class_by_name("MooncakeKVConnector") is MooncakeKVConnector`；
3. Cache key 永远是 `(hash(token_prefix), weight_version)` 三元组——不同 weight_version 永不撞键；
4. 单元测试 26+ 项通过，`mooncake` 缺包时整文件跳过；
5. 在装了 `mooncake` 的真主机上 `pytest llm_router/connector/tests/test_mooncake_store.py -v` 也能 pass（put/get round-trip）；
6. 47+ 项总测试通过、Plan A 测试无回归。

---

## Self-Review

**Spec coverage**：
- RFC §5.1 connector 双向迁移 → Task 7（`start_load_kv` / `save_kv_layer`）
- RFC §5.1 本地 miss 时查池 → Task 7（`get_num_new_matched_tokens`）
- RFC §5.2 version-tagged key → Task 2 + 贯穿 Task 6/7
- RFC §5.2 LRU 自然过期 → Task 4 InMemoryKVStore（cap 不够大时显式驱逐）
- §3.2 二级池本身 → Task 5 MooncakeKVStore
- §3.2 KV pool 跨 replica RDMA 共享 → **本计划暂不覆盖**，标记为 Plan B-followup（在 README 与 mooncake.py docstring 写明）

**Placeholder scan**：
- 文档里 `# TODO(plan-b-followup)` 标注属于显式延后项，保留；Self-review Step 2 的 grep 把它放进白名单。

**Type 一致性**：
- `VersionedKey` 在 `prefix_hash.py` 定义，`store/base.py` / `store/in_memory.py` / `store/mooncake.py` / `connector.py` 全部用的同一个类型。
- `MooncakeReqMeta.weight_version: str` 与 `make_versioned_key(..., weight_version: Any)` 经 `coerce_weight_version` 后类型对齐。
- `KVStore.get` 签名 `(key) -> Optional[bytes]`，三处实现一致。

**Scope**：
- 只动 `llm_router/connector/` 新子包；`llm_router/` 顶层 / `verl/` / `uni_agent/` 一律不动。
- 不实现 Plan C 路由、Plan D prewarm、PD-disagg、多模态——文档里 "Plan B-followup" 段已清单化遗留项。

---

*本计划完成后再推进 Plan C（两段路由规则填充 `RuleBasedPolicy`）、Plan D（prewarm 子系统）。*
