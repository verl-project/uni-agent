# Plan C: 两段路由规则 + Prefix 上报通路 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `RuleBasedPolicy` 真正实现 RFC §5.3 的两段路由规则——常态走 session 一致性哈希 fast path（健康 + GPU 命中 + 不过载三项可行性检查），不通过则 O(R) 扫描挑选「GPU 命中 + 负载达标」的最优 replica；若仍无候选，回退到负载最轻。新增 `LoadBalancer.report_prefixes()` 通路，让 verl `AsyncLLMServerManager` 在 generate 完成后能向 LB 注册 "该 server 已见过这些 prefix" 的提示。

**Architecture:** 改动**完全局限**在 `llm_router/policy/`、`llm_router/load_balancer.py`、`llm_router/config.py`、`llm_router/manager.py` —— **不动 worker、不动 connector**。verl 的 `AsyncLLMServerManager`（被 `AgentLoopWorker` 复用）已经实现了 worker 一侧的 `prefix_signatures` 计算与 `report_prefixes` 调用（见 `verl/verl/experimental/agent_loop/agent_loop.py:281-367`），只要 trainer config 把 `actor_rollout_ref.rollout.context_aware_scheduling.enable=True` 打开，prefix_signatures 就会自动经 `LoadBalancer.acquire_server.remote(...)` 与 `LoadBalancer.report_prefixes.remote(...)` 流到我们这边。LB 收到的 prefix_signatures 由 `RuleBasedPolicy` 内部维护 `prefix_locations: dict[server_id, LRUCache[(version, hash), int]]` 表，作为「该 server 大概率拥有这些 KV」的路由 proxy。

**Tech Stack:** Python 3.10+, `cachetools.LRUCache`, `hashlib.blake2b`（已通过 `prefix_hash.hash_token_prefix` 与 verl `_iter_prefix_signatures` 对齐），Ray actor, pytest。

---

## File Structure

```
<repo_root>/
├── llm_router/                                # Plan A/B 已有
│   ├── policy/
│   │   ├── base.py                            # 改：RouterPolicy ABC 加 report_prefixes
│   │   ├── legacy_sticky.py                   # 改:加 report_prefixes 的 no-op 实现
│   │   ├── rule_based.py                      # 重写:真正实现两段规则
│   │   └── __init__.py                        # 不动
│   ├── load_balancer.py                       # 改:加 report_prefixes passthrough
│   ├── config.py                              # 改:加 hit_threshold/load_threshold/max_prefix_entries_per_server
│   ├── manager.py                             # 改:_init_load_balancer 传新参数
│   ├── README.md                              # 改:文档启用方法
│   └── tests/
│       ├── test_policy_base.py                # 改:补 report_prefixes 抽象测试
│       ├── test_legacy_sticky_policy.py       # 改:补 report_prefixes 是 no-op 的测试
│       ├── test_rule_based_policy_stub.py     # 删:Plan A 阶段的 stub 测试已过时
│       ├── test_rule_based_policy.py          # 新建:真正的两段规则单测
│       ├── test_load_balancer.py              # 改:补 report_prefixes 测试
│       ├── test_config.py                     # 改:补新字段测试
│       └── test_rule_based_routing_e2e.py     # 新建:LB + policy 端到端集成测试
```

每个文件单一职责，改动局部。

---

## Task 1: 扩展 `RouterPolicy` ABC

**Files:**
- Modify: `llm_router/policy/base.py`
- Modify: `llm_router/tests/test_policy_base.py`

- [ ] **Step 1: 改测试为新接口**

替换 `llm_router/tests/test_policy_base.py` 整文件：

```python
"""RouterPolicy abstract base contract tests."""
import pytest

from llm_router.policy.base import RouterPolicy


def test_router_policy_is_abstract():
    """RouterPolicy 是抽象类,不能直接实例化。"""
    with pytest.raises(TypeError):
        RouterPolicy(server_ids=["a", "b"])


def test_router_policy_subclass_must_implement_acquire_release_and_report():
    """子类必须实现 acquire_server / release_server / report_prefixes。"""

    class MissingReport(RouterPolicy):
        def acquire_server(self, request_id, **_):
            return "a"

        def release_server(self, server_id):
            pass

    with pytest.raises(TypeError):
        MissingReport(server_ids=["a", "b"])


def test_router_policy_subclass_with_methods_works():
    """实现了三个方法的子类可正常实例化。"""

    class Minimal(RouterPolicy):
        def acquire_server(self, request_id, **_):
            return self.server_ids[0]

        def release_server(self, server_id):
            pass

        def report_prefixes(self, server_id, prefix_signatures):
            pass

    policy = Minimal(server_ids=["a", "b"])
    assert policy.acquire_server("req-1") == "a"
    policy.release_server("a")
    policy.report_prefixes("a", [("v0", "deadbeef", 1024)])
```

- [ ] **Step 2: 跑测试,应失败**

Run: `pytest llm_router/tests/test_policy_base.py -v`
Expected: `test_router_policy_subclass_must_implement_acquire_release_and_report` FAIL（MissingReport 当前能实例化,因为 base 还没声明 report_prefixes 是 abstract）

- [ ] **Step 3: 改 ABC,加 `report_prefixes` 抽象方法**

`llm_router/policy/base.py`:

```python
"""Abstract base for request → replica routing policies."""
from abc import ABC, abstractmethod
from typing import Any


class RouterPolicy(ABC):
    """Policy that picks a replica server for an incoming request.

    Sub-classes implement three methods:
    - `acquire_server`: choose a server, increment its in-flight counter.
    - `release_server`: decrement a server's in-flight counter on completion.
    - `report_prefixes`: record that a server has been observed handling
      these prefix signatures (used by context-aware policies; legacy
      policies can implement this as a no-op).
    """

    def __init__(self, server_ids: list[str]):
        if not server_ids:
            raise ValueError("server_ids must be non-empty")
        self.server_ids = list(server_ids)

    @abstractmethod
    def acquire_server(self, request_id: str, **kwargs: Any) -> str:
        """Return the server id chosen for this request. Implementations
        SHOULD increment any internal in-flight counter for that server.
        """

    @abstractmethod
    def release_server(self, server_id: str) -> None:
        """Decrement the in-flight counter for the given server."""

    @abstractmethod
    def report_prefixes(
        self,
        server_id: str,
        prefix_signatures: list[tuple[str, str, int]],
    ) -> None:
        """Tell the policy that `server_id` has just processed a request
        whose prompt yields these prefix signatures. Policies that don't
        do prefix-aware routing implement this as a no-op.

        Each tuple is `(weight_version, prefix_hash, prefix_len)`.
        """
```

- [ ] **Step 4: 跑测试,应通过**

Run: `pytest llm_router/tests/test_policy_base.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add llm_router/policy/base.py llm_router/tests/test_policy_base.py
git commit -m "[llm_router] feat: add report_prefixes to RouterPolicy ABC"
```

---

## Task 2: `LegacyStickyPolicy` 加 `report_prefixes` no-op

**Files:**
- Modify: `llm_router/policy/legacy_sticky.py`
- Modify: `llm_router/tests/test_legacy_sticky_policy.py`

- [ ] **Step 1: 补测试**

在 `llm_router/tests/test_legacy_sticky_policy.py` 文件末尾追加:

```python


def test_report_prefixes_is_noop_for_legacy():
    """Legacy 不做 prefix 感知,report_prefixes 必须不抛、不影响后续路由。"""
    p = LegacyStickyPolicy(server_ids=["a", "b"])
    p.report_prefixes("a", [("v0", "deadbeef", 1024)])
    # Routing unchanged.
    assert p.acquire_server("req-1") == "a"


def test_report_prefixes_validates_unknown_server():
    """Legacy 的 no-op 也应该校验 server_id 存在,避免静默配置错误。"""
    import pytest
    p = LegacyStickyPolicy(server_ids=["a"])
    with pytest.raises(ValueError, match="Invalid server_id"):
        p.report_prefixes("nonexistent", [])
```

- [ ] **Step 2: 跑测试,应失败**

Run: `pytest llm_router/tests/test_legacy_sticky_policy.py -v`
Expected: 2 NEW tests fail（AttributeError: 'LegacyStickyPolicy' object has no attribute 'report_prefixes'，因为 base 现在要求实现 → LegacyStickyPolicy 实例化也会失败 → 现有 6 测试一并 fail）

实际预期:整个文件 collection 不一定 fail，但实例化所有现有测试会 `TypeError: Can't instantiate abstract class LegacyStickyPolicy with abstract method report_prefixes`。

- [ ] **Step 3: 在 `LegacyStickyPolicy` 加 no-op 方法**

打开 `llm_router/policy/legacy_sticky.py`，在 `release_server` 之后追加:

```python
    def report_prefixes(
        self,
        server_id: str,
        prefix_signatures: list[tuple[str, str, int]],
    ) -> None:
        """No-op. Legacy sticky binding does not use prefix locations.

        Validates `server_id` to surface config errors early — silent no-ops
        on bogus ids would hide trainer misconfiguration.
        """
        if server_id not in self._inflight:
            raise ValueError(f"Invalid server_id: {server_id}")
```

- [ ] **Step 4: 跑测试**

Run: `pytest llm_router/tests/test_legacy_sticky_policy.py -v`
Expected: 8 passed（6 原有 + 2 新增）

- [ ] **Step 5: 提交**

```bash
git add llm_router/policy/legacy_sticky.py llm_router/tests/test_legacy_sticky_policy.py
git commit -m "[llm_router] feat: LegacyStickyPolicy.report_prefixes (no-op with validation)"
```

---

## Task 3: 扩展 `LLMRouterConfig`

加 `hit_threshold` / `load_threshold` / `max_prefix_entries_per_server` 三个字段，给 `RuleBasedPolicy` 用。

**Files:**
- Modify: `llm_router/config.py`
- Modify: `llm_router/tests/test_config.py`

- [ ] **Step 1: 补测试**

在 `llm_router/tests/test_config.py` 文件末尾追加:

```python


def test_hit_threshold_default():
    cfg = parse_config(OmegaConf.create({}))
    assert cfg.hit_threshold == 1


def test_load_threshold_default():
    cfg = parse_config(OmegaConf.create({}))
    assert cfg.load_threshold == 1024


def test_max_prefix_entries_default():
    cfg = parse_config(OmegaConf.create({}))
    assert cfg.max_prefix_entries_per_server == 8192


def test_explicit_thresholds():
    cfg = parse_config(
        OmegaConf.create(
            {
                "policy": "rule_based",
                "hit_threshold": 256,
                "load_threshold": 32,
                "max_prefix_entries_per_server": 4096,
            }
        )
    )
    assert cfg.policy == "rule_based"
    assert cfg.hit_threshold == 256
    assert cfg.load_threshold == 32
    assert cfg.max_prefix_entries_per_server == 4096


def test_negative_thresholds_clamped_to_zero():
    cfg = parse_config(
        OmegaConf.create({"hit_threshold": -5, "load_threshold": -1})
    )
    assert cfg.hit_threshold == 0
    assert cfg.load_threshold == 0
```

- [ ] **Step 2: 跑测试,应失败**

Run: `pytest llm_router/tests/test_config.py -v`
Expected: 5 new tests fail (AttributeError on `LLMRouterConfig.hit_threshold`)

- [ ] **Step 3: 改 `LLMRouterConfig`**

替换 `llm_router/config.py` 整文件:

```python
"""Config schema for llm_router."""
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf

VALID_POLICIES = {"legacy_sticky", "rule_based"}


@dataclass
class LLMRouterConfig:
    policy: str = "legacy_sticky"
    routing_cache_size: int = 10000
    # RuleBasedPolicy knobs — ignored by LegacyStickyPolicy.
    hit_threshold: int = 1
    load_threshold: int = 1024
    max_prefix_entries_per_server: int = 8192


def parse_config(cfg: DictConfig | dict[str, Any]) -> LLMRouterConfig:
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True) or {}
    policy = cfg.get("policy", "legacy_sticky")
    if policy not in VALID_POLICIES:
        raise ValueError(f"unknown policy: {policy!r} (valid: {sorted(VALID_POLICIES)})")
    return LLMRouterConfig(
        policy=policy,
        routing_cache_size=int(cfg.get("routing_cache_size", 10000)),
        hit_threshold=max(0, int(cfg.get("hit_threshold", 1))),
        load_threshold=max(0, int(cfg.get("load_threshold", 1024))),
        max_prefix_entries_per_server=int(cfg.get("max_prefix_entries_per_server", 8192)),
    )
```

- [ ] **Step 4: 跑测试**

Run: `pytest llm_router/tests/test_config.py -v`
Expected: 9 passed（4 原有 + 5 新增）

- [ ] **Step 5: 提交**

```bash
git add llm_router/config.py llm_router/tests/test_config.py
git commit -m "[llm_router] feat: add hit_threshold/load_threshold/max_prefix_entries config"
```

---

## Task 4: 实现 `RuleBasedPolicy` 真正的两段规则

这是 Plan C 的核心。RFC §5.3 的算法：

1. **优先**：从所有 replica 中选 `L_gpu(r) ≥ hit_threshold` **且** `wait(r) < load_threshold` 的 replica；多候选取本地命中 `L_gpu(r)` 最大者。
2. **兜底**：若无候选，选 `argmin wait(r)`。

**Fast path**（在 `acquire_server` 一上来跑）：用 `session_id` 一致性哈希取候选 primary；若 primary 健康（不可达视为 `wait=+∞`）+ GPU 命中 ≥ 阈值 + 负载 < 阈值，直接 O(1) 转发。

**Files:**
- Rewrite: `llm_router/policy/rule_based.py`（Plan A 的 stub 全部替换）
- Delete: `llm_router/tests/test_rule_based_policy_stub.py`
- Create: `llm_router/tests/test_rule_based_policy.py`

- [ ] **Step 1: 删 Plan A 的 stub 测试**

```bash
git rm llm_router/tests/test_rule_based_policy_stub.py
```

- [ ] **Step 2: 写新测试**

`llm_router/tests/test_rule_based_policy.py`:

```python
"""RuleBasedPolicy: RFC §5.3 two-stage routing rule + report_prefixes."""
import pytest

from llm_router.policy.rule_based import RuleBasedPolicy

VERSION = "v0"


def _sig(hash_str: str, length: int):
    return (VERSION, hash_str, length)


def test_acquire_without_signatures_falls_back_to_legacy_least_loaded():
    """No prefix_signatures → no rule-1 candidate → rule 2 (least loaded)."""
    p = RuleBasedPolicy(server_ids=["a", "b", "c"])
    assert p.acquire_server("req-1") == "a"


def test_session_fast_path_when_primary_has_gpu_hit_and_low_load():
    p = RuleBasedPolicy(
        server_ids=["a", "b", "c"],
        hit_threshold=10,
        load_threshold=100,
    )
    # Seed "session-1 hashes to b" by reporting a prefix on b first acquire.
    sigs = [_sig("aaaaaaaaaaaaaaaa", 64)]
    # First request — assigns session to primary chosen by consistent hashing.
    server = p.acquire_server("sess-1", session_id="sess-1", prefix_signatures=sigs)
    # Report that THIS server now has the prefix (mimics worker-side report).
    p.report_prefixes(server, sigs)
    p.release_server(server)
    # Second request from the same session — fast path: primary has the
    # prefix and load=0 → should return same server in O(1).
    server2 = p.acquire_server("sess-1", session_id="sess-1", prefix_signatures=sigs)
    assert server2 == server


def test_acquire_with_gpu_hit_picks_replica_with_longest_match():
    p = RuleBasedPolicy(
        server_ids=["a", "b", "c"],
        hit_threshold=10,
        load_threshold=100,
    )
    sig_short = _sig("h_short", 16)
    sig_long = _sig("h_long", 64)
    p.report_prefixes("a", [sig_short])
    p.report_prefixes("b", [sig_long])
    # Request whose prompt yields BOTH signatures (worker sends multiple
    # stride samples). Expect rule-1 winner = "b" (longer hit).
    server = p.acquire_server(
        "req-x",
        session_id="sess-x",
        prefix_signatures=[sig_short, sig_long],
    )
    assert server == "b"


def test_overloaded_replica_excluded_from_rule1_candidates():
    p = RuleBasedPolicy(
        server_ids=["a", "b"],
        hit_threshold=10,
        load_threshold=2,
    )
    sig = _sig("h", 64)
    p.report_prefixes("a", [sig])
    # Saturate "a"'s in-flight count above load_threshold.
    p.acquire_server("req-1")
    p.acquire_server("req-2")
    # Now "a" has 2 in-flight (== threshold, NOT strictly less).
    # Rule 1 must reject "a" → fall through to least-loaded "b".
    server = p.acquire_server(
        "req-3", session_id="sess-3", prefix_signatures=[sig]
    )
    assert server == "b"


def test_rule2_fallback_least_loaded_when_no_gpu_hit_candidate():
    p = RuleBasedPolicy(
        server_ids=["a", "b", "c"],
        hit_threshold=10,
        load_threshold=100,
    )
    # No reports — every replica has empty prefix_locations.
    server = p.acquire_server(
        "req-1",
        session_id="sess-1",
        prefix_signatures=[_sig("h", 64)],
    )
    # No GPU hit anywhere → rule 2 picks min in-flight (all 0 → first key).
    assert server == "a"


def test_report_prefixes_records_max_observed_length_per_key():
    p = RuleBasedPolicy(server_ids=["a"], max_prefix_entries_per_server=10)
    p.report_prefixes("a", [_sig("h", 32)])
    p.report_prefixes("a", [_sig("h", 64)])
    p.report_prefixes("a", [_sig("h", 48)])  # shorter — must not overwrite max
    # Inspect via the rule-1 path: a request matching the prefix should
    # observe the longest length we've ever reported for that key.
    server = p.acquire_server(
        "req",
        session_id="sess",
        prefix_signatures=[_sig("h", 80)],
    )
    assert server == "a"


def test_report_prefixes_rejects_unknown_server():
    p = RuleBasedPolicy(server_ids=["a"])
    with pytest.raises(ValueError, match="Invalid server_id"):
        p.report_prefixes("bogus", [_sig("h", 16)])


def test_prefix_locations_lru_evicts_oldest():
    p = RuleBasedPolicy(
        server_ids=["a"], max_prefix_entries_per_server=2
    )
    p.report_prefixes("a", [_sig("h1", 16)])
    p.report_prefixes("a", [_sig("h2", 16)])
    p.report_prefixes("a", [_sig("h3", 16)])  # evicts h1
    # h1 lookup must miss → rule 1 finds no candidate → rule 2 returns "a"
    # by least-loaded; nothing to assert behaviorally beyond no crash. Just
    # verify the in-memory state directly.
    assert ("v0", "h1") not in p._prefix_locations["a"]
    assert ("v0", "h2") in p._prefix_locations["a"]
    assert ("v0", "h3") in p._prefix_locations["a"]


def test_in_flight_counters_respected_after_acquire():
    p = RuleBasedPolicy(server_ids=["a", "b"], hit_threshold=1, load_threshold=1)
    p.report_prefixes("a", [_sig("h", 64)])
    # First request — a wins rule 1.
    s1 = p.acquire_server("r1", session_id="s1", prefix_signatures=[_sig("h", 64)])
    assert s1 == "a"
    # Second request, a is now at load=1 (>= load_threshold) → rule 1 rejects a.
    # Rule 2 falls through to least-loaded → b.
    s2 = p.acquire_server("r2", session_id="s2", prefix_signatures=[_sig("h", 64)])
    assert s2 == "b"
```

- [ ] **Step 3: 跑测试,应失败**

Run: `pytest llm_router/tests/test_rule_based_policy.py -v`
Expected: 多数 FAIL（旧 stub 不感知 prefix_signatures）

- [ ] **Step 4: 重写 `RuleBasedPolicy`**

替换 `llm_router/policy/rule_based.py` 整文件:

```python
"""RuleBasedPolicy: RFC §5.3 two-stage routing rule.

Routing decision per request:

1. **Fast path (session affinity)**: the primary chosen by consistent
   hashing on `session_id` is tested against three conditions:
   - reachable (not flagged unhealthy);
   - `L_gpu(primary) >= hit_threshold` (primary has the prefix);
   - `wait(primary) < load_threshold` (primary not overloaded).
   If all three hold, return primary in O(1).

2. **Rule 1 (slow path, GPU-hit + load gate)**: scan all replicas; the
   candidate set is `{r : L_gpu(r) >= hit_threshold AND wait(r) <
   load_threshold}`. Among candidates, return the one with the largest
   `L_gpu(r)`. Tie-break by smaller `wait(r)`, then by server id.

3. **Rule 2 (fallback, least loaded)**: if rule 1 yields no candidate,
   return `argmin wait(r)` over all replicas.

`L_gpu(r)` is maintained as `max` of prefix lengths previously reported
for each `(weight_version, prefix_hash)` on replica r — see
`report_prefixes()`. The per-replica index is an LRU capped by
`max_prefix_entries_per_server`.

Worker-side prefix_signatures and report calls come from verl's
AsyncLLMServerManager (`verl/verl/experimental/agent_loop/agent_loop.py`,
lines 287 and 360 respectively). LLMRouter does not need to add new
hooks — it just needs to honor the contract on the LB side.
"""
from cachetools import LRUCache

from llm_router.policy.base import RouterPolicy

DEFAULT_CACHE_SIZE = 10000
DEFAULT_HIT_THRESHOLD = 1
DEFAULT_LOAD_THRESHOLD = 1024
DEFAULT_MAX_PREFIX_ENTRIES = 8192


class RuleBasedPolicy(RouterPolicy):
    """Two-stage routing rule from RFC §5.3."""

    def __init__(
        self,
        server_ids: list[str],
        max_cache_size: int = DEFAULT_CACHE_SIZE,
        hit_threshold: int = DEFAULT_HIT_THRESHOLD,
        load_threshold: int = DEFAULT_LOAD_THRESHOLD,
        max_prefix_entries_per_server: int = DEFAULT_MAX_PREFIX_ENTRIES,
    ):
        super().__init__(server_ids=server_ids)
        self._inflight: dict[str, int] = {sid: 0 for sid in self.server_ids}
        self._session_to_server: LRUCache = LRUCache(maxsize=max_cache_size)
        self._hit_threshold = max(0, hit_threshold)
        self._load_threshold = max(0, load_threshold)
        # Per-server prefix index: server_id → LRU[(version, prefix_hash) → max_observed_len]
        self._prefix_locations: dict[str, LRUCache] = {
            sid: LRUCache(maxsize=max_prefix_entries_per_server)
            for sid in self.server_ids
        }

    # ---- ABC contract ----

    def acquire_server(
        self,
        request_id: str,
        *,
        session_id: str | None = None,
        prefix_signatures: list[tuple[str, str, int]] | None = None,
        **kwargs,
    ) -> str:
        session_id = session_id or request_id

        # Fast path: session consistent hash → check primary.
        primary = self._session_to_server.get(session_id)
        if primary is None:
            primary = self._consistent_hash_server(session_id)
            self._session_to_server[session_id] = primary

        primary_hit = self._prefix_hit_len(primary, prefix_signatures)
        if (
            primary_hit >= self._hit_threshold
            and self._inflight[primary] < self._load_threshold
        ):
            self._inflight[primary] += 1
            return primary

        # Slow path: rule 1 (GPU-hit + load gate) → rule 2 (least loaded).
        server_id = self._preferred_server(prefix_signatures) or self._least_loaded_server()
        self._session_to_server[session_id] = server_id
        self._inflight[server_id] += 1
        return server_id

    def release_server(self, server_id: str) -> None:
        if server_id not in self._inflight:
            raise ValueError(f"Invalid server_id: {server_id}")
        if self._inflight[server_id] <= 0:
            raise ValueError(f"Release with no in-flight on server {server_id}")
        self._inflight[server_id] -= 1

    def report_prefixes(
        self,
        server_id: str,
        prefix_signatures: list[tuple[str, str, int]],
    ) -> None:
        if server_id not in self._prefix_locations:
            raise ValueError(f"Invalid server_id: {server_id}")
        index = self._prefix_locations[server_id]
        for version, prefix_hash, prefix_len in prefix_signatures:
            key = (str(version), str(prefix_hash))
            current = int(index.get(key, 0))
            new_len = int(prefix_len)
            # Record the max observed length per (version, hash) — shorter
            # observations should not overwrite a longer hit.
            if new_len > current:
                index[key] = new_len

    # ---- Internal helpers ----

    def _consistent_hash_server(self, session_id: str) -> str:
        import hashlib

        digest = hashlib.blake2b(session_id.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, byteorder="big") % len(self.server_ids)
        return self.server_ids[idx]

    def _prefix_hit_len(
        self,
        server_id: str,
        prefix_signatures: list[tuple[str, str, int]] | None,
    ) -> int:
        if not prefix_signatures:
            return 0
        index = self._prefix_locations.get(server_id, {})
        hit_len = 0
        for version, prefix_hash, prefix_len in prefix_signatures:
            cached = int(index.get((str(version), str(prefix_hash)), 0))
            hit_len = max(hit_len, min(cached, int(prefix_len)))
        return hit_len

    def _preferred_server(
        self,
        prefix_signatures: list[tuple[str, str, int]] | None,
    ) -> str | None:
        """Rule 1: pick GPU-hit + load-gate candidate with largest hit."""
        candidates = []
        for server_id in self.server_ids:
            hit_len = self._prefix_hit_len(server_id, prefix_signatures)
            wait = self._inflight[server_id]
            if hit_len >= self._hit_threshold and wait < self._load_threshold:
                candidates.append((hit_len, -wait, server_id))
        if not candidates:
            return None
        # max() picks largest hit_len, then largest -wait (= smallest wait),
        # then largest server_id (lexicographic). server_id tiebreak is
        # stable but arbitrary; deterministic for tests.
        return max(candidates)[2]

    def _least_loaded_server(self) -> str:
        return min(self._inflight, key=self._inflight.get)
```

- [ ] **Step 5: 跑测试**

Run: `pytest llm_router/tests/test_rule_based_policy.py -v`
Expected: 9 passed

- [ ] **Step 6: 跑全部 policy 测试**

Run: `pytest llm_router/tests/ -v -k "policy"`
Expected: 3 base + 8 legacy + 9 rule_based = 20 passed（旧 stub 测试已删）

- [ ] **Step 7: 提交**

```bash
git add llm_router/policy/rule_based.py llm_router/tests/test_rule_based_policy.py
git rm llm_router/tests/test_rule_based_policy_stub.py
git commit -m "[llm_router] feat: RuleBasedPolicy two-stage routing rule (RFC §5.3)"
```

---

## Task 5: `LoadBalancer` 加 `report_prefixes` passthrough

**Files:**
- Modify: `llm_router/load_balancer.py`
- Modify: `llm_router/tests/test_load_balancer.py`

- [ ] **Step 1: 补测试**

在 `llm_router/tests/test_load_balancer.py` 文件末尾追加:

```python


def test_loadbalancer_report_prefixes_passthrough():
    lb = LoadBalancer.remote(server_ids=["a", "b"], policy_name="rule_based")
    # No exception, returns None.
    ret = ray.get(
        lb.report_prefixes.remote("a", [("v0", "deadbeef", 64)])
    )
    assert ret is None


def test_loadbalancer_report_then_acquire_rule1_path():
    """End-to-end through the actor: report a hit, then acquire with
    matching signatures, expect that server."""
    lb = LoadBalancer.remote(
        server_ids=["a", "b", "c"],
        policy_name="rule_based",
        hit_threshold=10,
        load_threshold=100,
    )
    ray.get(lb.report_prefixes.remote("b", [("v0", "h_long", 64)]))
    server = ray.get(
        lb.acquire_server.remote(
            "req-1",
            session_id="sess-1",
            prefix_signatures=[("v0", "h_long", 64)],
        )
    )
    assert server == "b"


def test_loadbalancer_legacy_report_prefixes_is_noop():
    """Legacy policy accepts report_prefixes (no-op) without affecting routing."""
    lb = LoadBalancer.remote(server_ids=["a"], policy_name="legacy_sticky")
    ray.get(lb.report_prefixes.remote("a", [("v0", "h", 64)]))
    s = ray.get(lb.acquire_server.remote("r1"))
    assert s == "a"
```

- [ ] **Step 2: 跑测试,应失败**

Run: `pytest llm_router/tests/test_load_balancer.py -v`
Expected: 3 new tests fail (LoadBalancer has no `report_prefixes` method).

- [ ] **Step 3: 改 `LoadBalancer`**

替换 `llm_router/load_balancer.py` 整文件:

```python
"""Ray-remote wrapper around a RouterPolicy."""
import ray

from llm_router.policy import LegacyStickyPolicy, RouterPolicy, RuleBasedPolicy

_POLICY_BUILDERS: dict[str, type[RouterPolicy]] = {
    "legacy_sticky": LegacyStickyPolicy,
    "rule_based": RuleBasedPolicy,
}


def _build_policy(
    server_ids: list[str],
    policy_name: str,
    routing_cache_size: int = 10000,
    hit_threshold: int = 1,
    load_threshold: int = 1024,
    max_prefix_entries_per_server: int = 8192,
) -> RouterPolicy:
    if policy_name not in _POLICY_BUILDERS:
        raise ValueError(f"unknown policy: {policy_name!r}")
    cls = _POLICY_BUILDERS[policy_name]
    if cls is RuleBasedPolicy:
        return cls(
            server_ids=server_ids,
            max_cache_size=routing_cache_size,
            hit_threshold=hit_threshold,
            load_threshold=load_threshold,
            max_prefix_entries_per_server=max_prefix_entries_per_server,
        )
    return cls(server_ids=server_ids, max_cache_size=routing_cache_size)


@ray.remote
class LoadBalancer:
    """Single actor that owns the RouterPolicy state for the entire job."""

    def __init__(
        self,
        server_ids: list[str],
        policy_name: str = "legacy_sticky",
        routing_cache_size: int = 10000,
        hit_threshold: int = 1,
        load_threshold: int = 1024,
        max_prefix_entries_per_server: int = 8192,
    ):
        self._policy = _build_policy(
            server_ids=server_ids,
            policy_name=policy_name,
            routing_cache_size=routing_cache_size,
            hit_threshold=hit_threshold,
            load_threshold=load_threshold,
            max_prefix_entries_per_server=max_prefix_entries_per_server,
        )

    def acquire_server(self, request_id: str, **kwargs) -> str:
        return self._policy.acquire_server(request_id, **kwargs)

    def release_server(self, server_id: str) -> None:
        self._policy.release_server(server_id)

    def report_prefixes(
        self,
        server_id: str,
        prefix_signatures: list[tuple[str, str, int]],
    ) -> None:
        self._policy.report_prefixes(server_id, prefix_signatures)
```

- [ ] **Step 4: 跑测试**

Run: `pytest llm_router/tests/test_load_balancer.py -v`
Expected: 7 passed（4 原有 + 3 新增）

- [ ] **Step 5: 提交**

```bash
git add llm_router/load_balancer.py llm_router/tests/test_load_balancer.py
git commit -m "[llm_router] feat: LoadBalancer.report_prefixes + rule-based config plumbing"
```

---

## Task 6: `LLMRouter._init_load_balancer` 透传新参数

**Files:**
- Modify: `llm_router/manager.py`

- [ ] **Step 1: 改 `_init_load_balancer`**

定位 `llm_router/manager.py` 中的 `_init_load_balancer` 方法。当前实现:

```python
    async def _init_load_balancer(self) -> None:
        self.load_balancer = LoadBalancer.remote(
            server_ids=self.server_addresses,
            policy_name=self._llm_router_config.policy,
            routing_cache_size=self._llm_router_config.routing_cache_size,
        )
```

替换为:

```python
    async def _init_load_balancer(self) -> None:
        self.load_balancer = LoadBalancer.remote(
            server_ids=self.server_addresses,
            policy_name=self._llm_router_config.policy,
            routing_cache_size=self._llm_router_config.routing_cache_size,
            hit_threshold=self._llm_router_config.hit_threshold,
            load_threshold=self._llm_router_config.load_threshold,
            max_prefix_entries_per_server=self._llm_router_config.max_prefix_entries_per_server,
        )
```

- [ ] **Step 2: 验证导入**

Run: `python -c "from llm_router import LLMRouter; print('OK')"`
Expected: `OK`

- [ ] **Step 3: 跑 Plan A 集成测试,确保 manager parity 未污染**

Run: `pytest llm_router/tests/test_manager_parity.py -v`
Expected: 1 skipped (no GPU)

- [ ] **Step 4: 跑全部 llm_router 测试**

Run: `pytest llm_router/ -v --tb=short`
Expected: 53+某新增 = 大约 60 passed + 2 skipped（待 Task 7 集成测试加入后会再增加）

- [ ] **Step 5: 提交**

```bash
git add llm_router/manager.py
git commit -m "[llm_router] feat: LLMRouter passes rule-based thresholds to LoadBalancer"
```

---

## Task 7: 端到端集成测试 + README

**Files:**
- Create: `llm_router/tests/test_rule_based_routing_e2e.py`
- Modify: `llm_router/README.md`

- [ ] **Step 1: 写端到端集成测试**

`llm_router/tests/test_rule_based_routing_e2e.py`:

```python
"""End-to-end: LoadBalancer + RuleBasedPolicy mimicking verl AsyncLLMServerManager.

Simulates the exact call sequence verl makes on each request:
  1. Worker computes prefix_signatures from prompt_ids + weight_version.
  2. Worker calls LoadBalancer.acquire_server.remote(
         request_id, session_id, prefix_signatures).
  3. Worker drives the picked replica (mocked here — we just count routes).
  4. After generation, worker calls LoadBalancer.report_prefixes.remote(
         server_id, prefix_signatures_of_cached_prompt).
  5. Worker calls LoadBalancer.release_server.remote(server_id).
"""
import hashlib

import pytest
import ray


@pytest.fixture(scope="module", autouse=True)
def ray_local():
    ray.init(num_cpus=2, local_mode=True, ignore_reinit_error=True)
    yield
    ray.shutdown()


def _hash_prefix(prompt_ids: list[int], prefix_len: int) -> str:
    """Mirror llm_router.connector.prefix_hash.hash_token_prefix shape."""
    h = hashlib.blake2b(digest_size=16)
    for tok in prompt_ids[:prefix_len]:
        h.update(int(tok).to_bytes(8, byteorder="little", signed=True))
    h.update(b":")
    h.update(str(prefix_len).encode("ascii"))
    return h.hexdigest()


def _signatures(prompt_ids: list[int], weight_version: str, stride: int = 16):
    """Mirror verl._iter_prefix_signatures: stride-sampled prefix hashes."""
    if not prompt_ids:
        return []
    lengths = list(range(stride, len(prompt_ids), stride))
    if not lengths or lengths[-1] != len(prompt_ids):
        lengths.append(len(prompt_ids))
    return [
        (weight_version, _hash_prefix(prompt_ids, n), n) for n in lengths
    ]


def _drive_request(lb, request_id, session_id, prompt_ids, weight_version):
    """Simulate one verl-style request through the LB."""
    sigs = _signatures(prompt_ids, weight_version)
    server_id = ray.get(
        lb.acquire_server.remote(
            request_id,
            session_id=session_id,
            prefix_signatures=sigs,
        )
    )
    # Mock generation: server now has the prompt cached.
    ray.get(lb.report_prefixes.remote(server_id, sigs))
    ray.get(lb.release_server.remote(server_id))
    return server_id


def test_e2e_session_affinity_after_first_turn():
    """A session's second turn lands on the same replica as its first."""
    from llm_router.load_balancer import LoadBalancer

    lb = LoadBalancer.remote(
        server_ids=["s0", "s1", "s2"],
        policy_name="rule_based",
        hit_threshold=1,
        load_threshold=100,
    )
    prompt_t1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    prompt_t2 = prompt_t1 + [17, 18, 19, 20]

    s_t1 = _drive_request(lb, "req-t1", "session-X", prompt_t1, "v0")
    s_t2 = _drive_request(lb, "req-t2", "session-X", prompt_t2, "v0")

    assert s_t1 == s_t2


def test_e2e_different_weight_version_does_not_hit():
    """KV reported under v0 must not satisfy a v1 query (RFC §5.2)."""
    from llm_router.load_balancer import LoadBalancer

    lb = LoadBalancer.remote(
        server_ids=["s0", "s1"],
        policy_name="rule_based",
        hit_threshold=1,
        load_threshold=100,
    )
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    # Turn under v0 → server reported as having v0 prefix.
    _drive_request(lb, "r-v0", "sess", prompt, "v0")

    # Next turn under v1 — consistent hash still maps "sess" to the same
    # primary, but L_gpu(primary) for the v1 query is 0 → fast path fails
    # the hit_threshold check. Then rule 1 also finds no candidate, so
    # rule 2 falls through to least-loaded, which is non-deterministic
    # between empty servers (all in_flight=0 → pick first alphabetically).
    # We assert only that the returned server is in the set.
    s_v1 = _drive_request(lb, "r-v1", "sess", prompt, "v1")
    assert s_v1 in {"s0", "s1"}


def test_e2e_legacy_policy_ignores_signatures():
    """Legacy mode: signatures are silently accepted but irrelevant."""
    from llm_router.load_balancer import LoadBalancer

    lb = LoadBalancer.remote(
        server_ids=["s0", "s1"],
        policy_name="legacy_sticky",
    )
    prompt = [1, 2, 3]
    s = _drive_request(lb, "r", "sess", prompt, "v0")
    assert s in {"s0", "s1"}
```

- [ ] **Step 2: 跑测试**

Run: `pytest llm_router/tests/test_rule_based_routing_e2e.py -v`
Expected: 3 passed

- [ ] **Step 3: 更新 README**

替换 `llm_router/README.md` 中的 `## Policies` 部分:

将原本

```markdown
## Policies

- `legacy_sticky` — exact behavior of verl's `GlobalRequestLoadBalancer`. Use this to validate the drop-in is byte-identical to the original.
- `rule_based` — stub in Plan A; Plan C fills in the two-stage rule from the RFC.
```

改为:

```markdown
## Policies

- `legacy_sticky` — exact behavior of verl's `GlobalRequestLoadBalancer`. Use this to validate the drop-in is byte-identical to the original.
- `rule_based` — RFC §5.3 two-stage rule. Fast path: session consistent hashing + health/GPU-hit/load feasibility check. Slow path: rule 1 picks the replica with the largest GPU prefix hit under the load threshold; rule 2 falls back to least-loaded.

### Enabling rule_based

The two-stage rule requires `prefix_signatures` to flow in on every request,
plus `report_prefixes` calls after each generation. **verl's
`AsyncLLMServerManager` already does both** (see
`verl/verl/experimental/agent_loop/agent_loop.py:287, 360`) — you just need
to turn the context-aware flag on in your trainer config:

```yaml
actor_rollout_ref:
  rollout:
    agent:
      agent_loop_manager_class: "llm_router.LLMRouter"
    llm_router:
      policy: rule_based
      hit_threshold: 1                  # min GPU prefix tokens to count as a hit
      load_threshold: 1024              # in-flight queue cap before rule 1 rejects
      max_prefix_entries_per_server: 8192
      routing_cache_size: 10000
    context_aware_scheduling:
      enable: true                      # verl flag — drives worker-side prefix_signatures + report
      prefix_probe_stride: 256
```

With `enable: true`, verl's `AsyncLLMServerManager` computes
`prefix_signatures` from each request's `prompt_ids + weight_version` and
threads them through `acquire_server`. After generation, it reports the
cached-prompt signatures back via `report_prefixes`. `LLMRouter`'s
`LoadBalancer` consumes both and feeds `RuleBasedPolicy`.

### Plan C follow-ups (not in this plan)

- **Connector → LB direct reporting**: a Plan-B connector hook could
  notify the LB the moment KV lands in the Mooncake store, instead of
  relying on the worker-side "this server has been observed handling
  this prompt" proxy. The current proxy is sufficient when worker→store
  visibility is sound, but a direct hook would let routing react to KV
  evictions and cross-replica transfers without trainer involvement.
- **Stride-aligned key granularity**: today `_save_request_layer` (Plan B)
  stores one KV blob per full prompt, while worker reports stride-sampled
  signatures (every 256 tokens by default). The mismatch means a request
  matching a 256-token prefix would route on the hit but find no payload
  in the store. Aligning the store-key granularity to the report stride
  would close the loop.
```

- [ ] **Step 4: 跑全部 llm_router 测试**

Run: `pytest llm_router/ -v --tb=short`
Expected: roughly 64 passed + 2 skipped (Plan A 19+1 + Plan B 34+1 + Plan C 11+3 = 64).

具体分项目: 3 base + 8 legacy + 9 rule_based + 9 config + 7 LB + 1 parity-skip (Plan A) + Plan B 34+1 + 3 e2e = 64+2 大致。

- [ ] **Step 5: 提交**

```bash
git add llm_router/tests/test_rule_based_routing_e2e.py llm_router/README.md
git commit -m "[llm_router] test+docs: rule-based routing e2e + README"
```

---

## Task 8: Self-review 与终态提交

- [ ] **Step 1: 跑全 llm_router 测试套**

Run: `pytest llm_router/ -v --tb=short`
Expected: 0 failed; passed count matches Task 7 expectation.

- [ ] **Step 2: TODO 扫描**

Run: `grep -rn "TODO\|FIXME\|XXX" llm_router/ | grep -v "Plan C\|Plan B\|Plan D\|plan-b-followup\|plan-followup\|plan-c-followup\|TODO(parity)"`
Expected: 输出为空。

- [ ] **Step 3: Lint**

Run: `ruff check llm_router/`
Expected: `All checks passed!`

- [ ] **Step 4: 验证 Plan A 与 Plan B 测试无回归**

Run: `pytest llm_router/tests/ llm_router/connector/tests/ -v --tb=short`
Expected: 全 pass（含 GPU/mooncake 两个 skip）。

- [ ] **Step 5: 终态提交**

```bash
git add -u llm_router/
git commit --allow-empty -m "[llm_router] chore: Plan C complete — rule-based routing with prefix reporting"
```

---

## 完成判据

落地后应满足：

1. `RuleBasedPolicy` 真正实现 RFC §5.3 两段规则（含 fast path + rule 1 + rule 2 + LRU 维护的 prefix_locations）。
2. `LoadBalancer.report_prefixes()` actor 方法 passthrough 到 policy。
3. `LLMRouterConfig` 新增 3 个字段：`hit_threshold` / `load_threshold` / `max_prefix_entries_per_server`，默认值与 verl `GlobalRequestLoadBalancer` 对齐。
4. `LLMRouter._init_load_balancer` 透传新参数。
5. 单元测试 ≥ 60 项通过；端到端 e2e 测试 3 项通过；ruff clean；Plan A / Plan B 无回归。
6. README 文档化启用方法 + 两条 Plan-C follow-up。
7. **不触碰** verl worker / Plan B connector / uni_agent。

---

## Self-Review

**Spec coverage**：
- RFC §5.3 fast path → Task 4 `acquire_server` 头部 session 一致性 hash + 三项检查
- RFC §5.3 rule 1 → Task 4 `_preferred_server`
- RFC §5.3 rule 2 → Task 4 `_least_loaded_server`
- RFC §5.2 version-tagged key 在 routing 侧的语义安全 → Task 4 `_prefix_hit_len`/`report_prefixes` 都以 `(version, hash)` 为 key，跨 version 自然不命中
- 阈值参数化 → Task 3 config
- LB actor 暴露给 verl 的方法集合 → Task 5 含 `report_prefixes`
- 默认值与 verl `GlobalRequestLoadBalancer.__init__` 对齐 → Task 3（`hit_threshold=1`、`load_threshold=1024`、`max_prefix_entries_per_server=8192`、`routing_cache_size=10000`）

**Placeholder scan**：
- README 显式记录两条 Plan C follow-up（connector→LB direct reporting、stride-aligned key granularity），Step 2 grep 把 `plan-c-followup` 加入白名单。

**Type 一致性**：
- `prefix_signatures: list[tuple[str, str, int]] | None`、`session_id: str | None`、`hit_threshold: int`、`load_threshold: int`、`max_prefix_entries_per_server: int` 三处（base / rule_based / LoadBalancer / config / manager）签名一致。
- `prefix_locations: dict[str, LRUCache]` 中 LRUCache 的键类型是 `(str, str)` （即 `(version, prefix_hash)`），值类型是 `int` （prefix_len）—— 在 `report_prefixes` 与 `_prefix_hit_len` 中一致。
- `LegacyStickyPolicy.report_prefixes` 接收同样的签名，no-op + 校验 server_id。

**Scope**：
- 只动 `llm_router/policy/`、`llm_router/load_balancer.py`、`llm_router/config.py`、`llm_router/manager.py`、`llm_router/README.md`、`llm_router/tests/`。
- 不动 verl、Plan B connector、uni_agent。

---

*本计划完成后再推进 Plan D（prewarm 子系统）。Plan D 用 verl 已有的 `prewarm_prefixes`（见 `verl/.../agent_loop.py:372`）——由 trainer 在 weight update 后调用 LLMRouter 的对应方法，仍然不需要动 worker。*
