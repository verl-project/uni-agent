from __future__ import annotations

import ctypes
import importlib.util
import logging
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from uni_agent.gateway.agent_hint import AgentRuntimeHint
from uni_agent.gateway.kv_offload.hints import AgentHintConfig

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def _load(monkeypatch):
    base = types.ModuleType("vllm.v1.kv_offload.base")
    base.OffloadKey = bytes

    @dataclass
    class ReqContext:
        req_id: str
        kv_transfer_params: dict | None = None

    base.ReqContext = ReqContext
    manager_module = types.ModuleType("vllm.v1.kv_offload.cpu.manager")
    manager_module.CPUOffloadingManager = type(
        "CPUOffloadingManager",
        (),
        {
            "on_new_request": lambda self, context: None,
            "on_request_finished": lambda self, context: None,
        },
    )
    policy_module = types.ModuleType("vllm.v1.kv_offload.cpu.policies.base")

    class BlockStatus(ctypes.Structure):
        _fields_ = [("ref_cnt", ctypes.c_int32), ("block_id", ctypes.c_int64)]

        def __init__(self, block_id):
            super().__init__()
            self.ref_cnt = 0
            self.block_id = block_id

    class CachePolicy:
        def __init__(self, cache_capacity):
            self.cache_capacity = cache_capacity

    policy_module.BlockStatus = BlockStatus
    policy_module.CachePolicy = CachePolicy
    spec_module = types.ModuleType("vllm.v1.kv_offload.cpu.spec")
    spec_module.CPUOffloadingSpec = type("CPUOffloadingSpec", (), {})
    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.kv_offload": types.ModuleType("vllm.v1.kv_offload"),
        "vllm.v1.kv_offload.base": base,
        "vllm.v1.kv_offload.cpu": types.ModuleType("vllm.v1.kv_offload.cpu"),
        "vllm.v1.kv_offload.cpu.manager": manager_module,
        "vllm.v1.kv_offload.cpu.policies": types.ModuleType("vllm.v1.kv_offload.cpu.policies"),
        "vllm.v1.kv_offload.cpu.policies.base": policy_module,
        "vllm.v1.kv_offload.cpu.spec": spec_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    # Load under a private name so stub-based tests cannot poison the module
    # later imported by real-vLLM integration tests in the same pytest run.
    spec = importlib.util.spec_from_file_location(
        "_test_priority_policy", Path(__file__).resolve().parents[4] / "uni_agent/gateway/kv_offload/priority_policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, BlockStatus, ReqContext


def _runtime(**overrides):
    fields = dict(
        tools_available=True,
        has_active_chain=True,
        received_tool_result=True,
        context_tokens=40_000,
        trajectory_capacity=65_536,
        remaining_capacity=20_000,
        rollback_applied=False,
        active_chain_count=1,
        trajectory_id="trajectory-1",
    )
    fields.update(overrides)
    return AgentRuntimeHint(**fields).to_dict()


@pytest.mark.parametrize("deadline", [199, 200, 201])
@pytest.mark.parametrize("allow_missing, threshold", [(False, 0), (True, 0), (True, 50)])
def test_admission_applies_no_hint_rules_to_expired_claims(monkeypatch, deadline, allow_missing, threshold):
    module, _, ReqContext = _load(monkeypatch)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100)
    monkeypatch.setattr(module.CPUOffloadingManager, "__init__", lambda self, **kwargs: None)
    calls = []

    def prepare_store(self, keys, context):
        calls.append((list(keys), context))
        return keys

    monkeypatch.setattr(module.CPUOffloadingManager, "prepare_store", prepare_store, raising=False)
    manager = module.PriorityCPUOffloadingManager(
        1,
        agent_hint_config=AgentHintConfig(
            active_lease_seconds=deadline - 100, min_offload_priority=threshold, admit_without_hint=allow_missing
        ),
    )
    # A previous request must not leak its high priority into fallback admission.
    manager._policy.set_request_hint(module._Hint(priority=100, lease_until=1000))
    context = ReqContext("request", {"agent_hint": _runtime()})
    manager._request_hint(context)
    monkeypatch.setattr(module.time, "monotonic", lambda: 200)
    admitted = deadline > 200 or (allow_missing and threshold == 0)
    assert manager.prepare_store([b"new"], context) == ([b"new"] if admitted else [])
    assert calls == [([b"new"] if admitted else [], context)]
    assert manager.accepted_store_calls == int(admitted)
    assert manager.rejected_store_calls == int(not admitted)
    assert manager._policy._current_hint is None


@pytest.mark.parametrize("target", ["manager", "spec"])
@pytest.mark.parametrize(
    "field, value",
    [("min_offload_priority", v) for v in [True, False, 1.9, "50", -1, 101, None]]
    + [("admit_without_hint", v) for v in ["false", "0", 0, 1, None]],
)
def test_admission_config_rejects_invalid_types_before_native_init(monkeypatch, target, field, value):
    module, _, _ = _load(monkeypatch)

    def unexpected_init(*args, **kwargs):
        pytest.fail("invalid admission config must fail before native initialization")

    monkeypatch.setattr(module.CPUOffloadingManager, "__init__", unexpected_init)
    monkeypatch.setattr(module.CPUOffloadingSpec, "__init__", unexpected_init)
    with pytest.raises(ValueError, match=field):
        if target == "manager":
            module.PriorityCPUOffloadingManager(1, agent_hint_config=AgentHintConfig(**{field: value}))
        else:
            config = types.SimpleNamespace(
                kv_transfer_config=types.SimpleNamespace(
                    kv_connector_extra_config={"agent_hint_config": {field: value}}
                )
            )
            module.PriorityGPUOffloadingSpec(config, None)


@pytest.mark.parametrize("priority, allow_missing", [(0, False), (50, True), (100, False)])
def test_admission_config_preserves_valid_values(monkeypatch, priority, allow_missing):
    module, _, _ = _load(monkeypatch)
    monkeypatch.setattr(module.CPUOffloadingManager, "__init__", lambda self, **kwargs: None)
    monkeypatch.setattr(module.CPUOffloadingSpec, "__init__", lambda self, config, cache: None)
    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(
            kv_connector_extra_config={
                "agent_hint_config": {"min_offload_priority": priority, "admit_without_hint": allow_missing},
            }
        )
    )
    for instance in [
        module.PriorityCPUOffloadingManager(
            1, agent_hint_config=AgentHintConfig(min_offload_priority=priority, admit_without_hint=allow_missing)
        ),
        module.PriorityGPUOffloadingSpec(config, None),
    ]:
        assert instance.agent_hint_config.min_offload_priority == priority
        assert instance.agent_hint_config.admit_without_hint is allow_missing


def test_priority_policy_evicts_expired_then_lower_priority(monkeypatch):
    module, BlockStatus, _ = _load(monkeypatch)
    policy = module.PriorityCachePolicy(3)
    entries = [
        (b"leased-high", module._Hint(priority=90, lease_until=10**12)),
        (b"leased-low", module._Hint(priority=10, lease_until=10**12)),
        (b"expired-high", module._Hint(priority=100, lease_until=1)),
    ]
    for index, (key, hint) in enumerate(entries):
        policy.set_request_hint(hint)
        policy.insert(key, BlockStatus(index))

    evicted = policy.evict(2, protected=set())

    assert [key for key, _ in evicted] == [b"expired-high", b"leased-low"]
    assert policy.get(b"leased-high") is not None


def test_hint_parser_reads_gateway_agent_hint(monkeypatch):
    module, _, ReqContext = _load(monkeypatch)
    hint = module._parse_hint(ReqContext("request", {"agent_hint": _runtime()}))
    assert hint is not None
    assert hint.trajectory_id == "trajectory-1"
    assert hint.context_tokens == 40_000
    assert module.compute_dynamic_priority(hint).priority == 87


def test_touch_refreshes_hint_without_downgrading_live_shared_prefix(monkeypatch):
    module, BlockStatus, _ = _load(monkeypatch)
    policy = module.PriorityCachePolicy(1)
    policy.set_request_hint(module._Hint(priority=90, lease_until=10**12, trajectory_id="hot"))
    policy.insert(b"shared", BlockStatus(0))

    policy.set_request_hint(module._Hint(priority=30, lease_until=10**11, trajectory_id="cold"))
    policy.touch([b"shared"])

    assert policy.meta[b"shared"].priority == 90
    assert policy.meta[b"shared"].lease_until == 10**12
    assert policy.meta[b"shared"].trajectory_id == "hot"


def test_touch_replaces_expired_hint(monkeypatch):
    module, BlockStatus, _ = _load(monkeypatch)
    policy = module.PriorityCachePolicy(1)
    policy.set_request_hint(module._Hint(priority=90, lease_until=1, trajectory_id="expired"))
    policy.insert(b"shared", BlockStatus(0))

    policy.set_request_hint(module._Hint(priority=40, lease_until=10**12, trajectory_id="new"))
    policy.touch([b"shared"])

    assert policy.meta[b"shared"].priority == 40
    assert policy.meta[b"shared"].lease_until == 10**12
    assert policy.meta[b"shared"].trajectory_id == "new"


@pytest.mark.parametrize("lease_until", [float("nan"), float("inf"), float("-inf"), "NaN", "Infinity", 0, -1, None])
def test_hint_parser_rejects_legacy_policy_results(monkeypatch, lease_until):
    module, _, ReqContext = _load(monkeypatch)
    hint = module._parse_hint(ReqContext("request", {"agent_hint": {"kv_priority": 90, "lease_until": lease_until}}))

    assert hint is None


def test_hint_parser_requires_runtime_schema_without_changing_no_hint_default(monkeypatch):
    module, _, ReqContext = _load(monkeypatch)
    for params in [{"agent_hint": {"kv_priority": 90}}, None, {}]:
        assert module._parse_hint(ReqContext("request", params)) is None


def test_request_scored_once_and_lease_not_renewed_by_block_operations(monkeypatch):
    module, BlockStatus, ReqContext = _load(monkeypatch)
    monkeypatch.setattr(module.CPUOffloadingManager, "__init__", lambda self, **kwargs: None)
    monkeypatch.setattr(
        module.CPUOffloadingManager, "touch", lambda self, keys, context: self._policy.touch(keys), raising=False
    )
    monkeypatch.setattr(module.CPUOffloadingManager, "lookup", lambda self, key, context: True, raising=False)
    monkeypatch.setattr(
        module.CPUOffloadingManager, "prepare_store", lambda self, keys, context: list(keys), raising=False
    )
    scored = []
    compute = module.compute_dynamic_priority

    def score(runtime):
        scored.append(runtime)
        return compute(runtime)

    monkeypatch.setattr(module, "compute_dynamic_priority", score)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100)
    manager = module.PriorityCPUOffloadingManager(1, agent_hint_config=AgentHintConfig(active_lease_seconds=50))
    context = ReqContext("request", {"agent_hint": _runtime()})
    manager.on_new_request(context)
    hint = manager._request_hint(context)
    assert (hint.priority, hint.lease_until) == (87, 150)
    manager._policy.set_request_hint(hint)
    manager._policy.insert(b"shared", BlockStatus(0))
    monkeypatch.setattr(module.time, "monotonic", lambda: 140)
    for _ in range(10):
        manager.touch([b"shared"], ReqContext("request", context.kv_transfer_params))
        assert manager.lookup(b"shared", context) is True
        assert manager.prepare_store([b"new"], context) == [b"new"]
    assert len(scored) == 1
    assert manager._policy.meta[b"shared"].lease_until == 150
    manager.on_request_finished(context)
    assert not manager._request_hints
    manager.on_new_request(ReqContext("next-turn", context.kv_transfer_params))
    assert len(scored) == 2
    assert manager._request_hints["next-turn"].lease_until == 190


@pytest.mark.parametrize("key", ["min_offload_priority", "admit_without_hint", "priority_mode", "tool_priority"])
def test_spec_rejects_old_top_level_policy_options(monkeypatch, key):
    module, _, _ = _load(monkeypatch)
    config = types.SimpleNamespace(kv_transfer_config=types.SimpleNamespace(kv_connector_extra_config={key: 1}))
    with pytest.raises(ValueError, match="Put policy settings in agent_hint_config"):
        module.PriorityGPUOffloadingSpec(config, None)


@pytest.mark.parametrize("deadline", [150, 200])
@pytest.mark.parametrize("priority", [10, 30, 90])
def test_expired_hint_preserves_live_claim_and_updates_recency(monkeypatch, deadline, priority):
    module, BlockStatus, _ = _load(monkeypatch)
    monkeypatch.setattr(module.time, "monotonic", lambda: 200)
    policy = module.PriorityCachePolicy(2)
    policy.set_request_hint(module._Hint(priority=30, lease_until=500, trajectory_id="live"))
    policy.insert(b"shared", BlockStatus(0))
    policy.insert(b"other", BlockStatus(1))
    old_seq = policy.meta[b"shared"].touch_seq
    policy.set_request_hint(module._Hint(priority=priority, lease_until=deadline, trajectory_id="stale"))
    policy.touch([b"shared"])
    meta = policy.meta[b"shared"]
    assert (meta.priority, meta.lease_until, meta.trajectory_id) == (30, 500, "live")
    assert meta.touch_seq > old_seq
    assert list(policy.blocks) == [b"other", b"shared"]
    assert [key for key, _ in policy.evict(1, set())] == [b"other"]


def test_low_priority_touches_cannot_renew_high_priority_lease(monkeypatch):
    module, BlockStatus, _ = _load(monkeypatch)
    policy = module.PriorityCachePolicy(1)
    policy.set_request_hint(module._Hint(priority=90, lease_until=300, trajectory_id="hot"))
    policy.insert(b"shared", BlockStatus(0))
    for now, priority, deadline in [(200, 90, 300), (300, 30, 600), (400, 30, 700), (600, 30, 900)]:
        monkeypatch.setattr(module.time, "monotonic", lambda now=now: now)
        policy.set_request_hint(module._Hint(priority=30, lease_until=now + 300, trajectory_id="cold"))
        policy.touch([b"shared"])
        assert policy.meta[b"shared"].priority == priority
        assert policy.meta[b"shared"].lease_until == deadline
        assert policy.meta[b"shared"].trajectory_id == ("hot" if priority == 90 else "cold")


def test_upgrade_does_not_inherit_lower_priority_deadline(monkeypatch):
    module, BlockStatus, _ = _load(monkeypatch)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100)
    policy = module.PriorityCachePolicy(1)
    policy.set_request_hint(module._Hint(priority=30, lease_until=900, trajectory_id="cold"))
    policy.insert(b"shared", BlockStatus(0))
    policy.set_request_hint(module._Hint(priority=90, lease_until=300, trajectory_id="hot"))
    policy.touch([b"shared"])
    assert policy.meta[b"shared"].priority == 90
    assert policy.meta[b"shared"].lease_until == 300
    assert policy.meta[b"shared"].trajectory_id == "hot"
    policy.set_request_hint(module._Hint(priority=90, lease_until=400))
    policy.touch([b"shared"])
    assert policy.meta[b"shared"].lease_until == 400


def test_expired_high_priority_claim_is_evictable_after_cold_touch(monkeypatch):
    module, BlockStatus, _ = _load(monkeypatch)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100)
    policy = module.PriorityCachePolicy(2)
    for index, (key, priority, deadline) in enumerate([(b"shared", 90, 300), (b"live", 50, 900)]):
        policy.set_request_hint(module._Hint(priority=priority, lease_until=deadline))
        policy.insert(key, BlockStatus(index))
    monkeypatch.setattr(module.time, "monotonic", lambda: 200)
    policy.set_request_hint(module._Hint(priority=30, lease_until=500))
    policy.touch([b"shared"])
    monkeypatch.setattr(module.time, "monotonic", lambda: 301)
    assert [key for key, _ in policy.evict(1, set())] == [b"shared"]


def test_eviction_is_atomic_and_respects_busy_and_protected_blocks(monkeypatch):
    module, BlockStatus, _ = _load(monkeypatch)
    policy = module.PriorityCachePolicy(3)
    for index, key in enumerate([b"free", b"busy", b"protected"]):
        policy.insert(key, BlockStatus(index))
    policy.get(b"busy").ref_cnt = 1
    assert policy.evict(2, {b"protected"}) is None
    assert len(policy.blocks) == 3
    assert [key for key, _ in policy.evict(1, {b"protected"})] == [b"free"]


@pytest.mark.parametrize("as_iterator", [False, True])
def test_equal_priority_touch_evicts_suffix_before_prefix(monkeypatch, as_iterator):
    module, BlockStatus, _ = _load(monkeypatch)
    policy = module.PriorityCachePolicy(3)
    keys = [b"prefix0", b"prefix1", b"prefix2"]
    for index, key in enumerate(keys):
        policy.insert(key, BlockStatus(index))
    policy.touch(iter(keys) if as_iterator else keys)
    assert list(policy.blocks) == list(reversed(keys))
    assert [key for key, _ in policy.evict(2, set())] == [b"prefix2", b"prefix1"]
    assert policy.get(b"prefix0") is not None


def test_reverse_touch_preserves_lru_between_requests(monkeypatch):
    module, BlockStatus, _ = _load(monkeypatch)
    policy = module.PriorityCachePolicy(4)
    older, newer = [b"a0", b"a1"], [b"b0", b"b1"]
    for index, key in enumerate(older + newer):
        policy.insert(key, BlockStatus(index))
    policy.touch(older)
    policy.touch(newer)
    assert [key for key, _ in policy.evict(3, set())] == [b"a1", b"a0", b"b1"]


@pytest.mark.parametrize("seed", [0, 179, 2026])
def test_indexed_eviction_matches_full_sort_after_mixed_operations(monkeypatch, seed):
    module, BlockStatus, _ = _load(monkeypatch)
    rng = random.Random(seed)
    now = 100
    monkeypatch.setattr(module.time, "monotonic", lambda: now)
    policy = module.PriorityCachePolicy(64)
    keys = [bytes([index]) for index in range(64)]
    for _ in range(2000):
        operation = rng.randrange(100)
        if operation < 35:
            key = rng.choice(keys)
            policy.set_request_hint(
                module._Hint(priority=rng.randrange(101), lease_until=rng.choice([0, now + 5, now + 30]))
            )
            policy.insert(key, BlockStatus(keys.index(key)))
        elif operation < 55:
            policy.set_request_hint(
                rng.choice([None, module._Hint(priority=rng.randrange(101), lease_until=now + rng.randrange(-5, 40))])
            )
            policy.touch(rng.sample(keys, rng.randrange(8)))
        elif operation < 65:
            policy.remove(rng.choice(keys))
        elif operation < 75:
            block = policy.get(rng.choice(keys))
            if block is not None:
                block.ref_cnt = rng.choice([-1, 0, 1, 2])
        elif operation < 85:
            now += rng.randrange(15)
        elif operation < 99:
            protected = set(rng.sample(keys, rng.randrange(8)))
            n = rng.randrange(8)
            before = dict(policy.blocks)
            candidates = [key for key, block in before.items() if block.ref_cnt == 0 and key not in protected]
            candidates.sort(
                key=lambda key: (
                    0 if policy.meta[key].expired(now) else 1,
                    policy.meta[key].priority,
                    policy.meta[key].touch_seq,
                )
            )
            evicted = policy.evict(n, protected)
            if len(candidates) < n:
                assert evicted is None
                assert dict(policy.blocks) == before
            else:
                assert [key for key, _ in evicted] == candidates[:n]
                assert dict(policy.blocks) == {key: block for key, block in before.items() if key not in candidates[:n]}
        else:
            policy.clear()
        # Repeated renewals, removals and reinserts must not leave stale records.
        assert len(policy._eviction_index) == len(policy.blocks)
        assert len(policy._lease_index) <= len(policy.blocks)


def test_single_eviction_checks_only_required_candidates(monkeypatch):
    module, _, _ = _load(monkeypatch)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100)
    policy = module.PriorityCachePolicy(10_000)
    checks = []

    class Block:
        def __init__(self, index):
            self.block_id = index
            self.busy = False

        @property
        def ref_cnt(self):
            checks.append(self.block_id)
            return int(self.busy)

    policy.set_request_hint(module._Hint(priority=50, lease_until=300))
    for index in range(10_000):
        policy.insert(index, Block(index))
    policy.get(0).busy = True
    assert [key for key, _ in policy.evict(1, {1})] == [2]
    assert checks == [0, 2]
    # Busy/protected entries must remain available without a touch callback.
    policy.get(0).busy = False
    checks.clear()
    assert [key for key, _ in policy.evict(1, set())] == [0]
    assert checks == [0]


def test_indexed_deadlines_follow_renewal_reset_and_key_reuse(monkeypatch):
    module, BlockStatus, _ = _load(monkeypatch)
    now = 100
    monkeypatch.setattr(module.time, "monotonic", lambda: now)
    policy = module.PriorityCachePolicy(2)
    policy.set_request_hint(module._Hint(priority=90, lease_until=150))
    policy.insert(b"renewed", BlockStatus(0))
    policy.set_request_hint(module._Hint(priority=10, lease_until=300))
    policy.insert(b"cold", BlockStatus(1))
    policy.set_request_hint(module._Hint(priority=90, lease_until=250))
    policy.touch([b"renewed"])
    now = 160
    assert [key for key, _ in policy.evict(1, set())] == [b"cold"]
    policy.clear()
    assert len(policy._eviction_index) == len(policy._lease_index) == 0
    policy.set_request_hint(module._Hint(priority=90, lease_until=400))
    policy.insert(b"renewed", BlockStatus(0))
    policy.set_request_hint(module._Hint(priority=10, lease_until=400))
    policy.insert(b"cold", BlockStatus(1))
    now = 260
    assert [key for key, _ in policy.evict(1, set())] == [b"cold"]
    now = 400
    assert [key for key, _ in policy.evict(1, set())] == [b"renewed"]


@pytest.mark.parametrize("result, state", [(True, "hit"), (None, "pending"), (False, "miss")])
@pytest.mark.parametrize("log_level", [logging.DEBUG, logging.WARNING])
def test_manager_hot_paths_do_not_scan_cache(monkeypatch, result, state, log_level):
    module, _, ReqContext = _load(monkeypatch)
    monkeypatch.setattr(module.logger, "level", log_level)
    manager = module.PriorityCPUOffloadingManager.__new__(module.PriorityCPUOffloadingManager)
    manager._policy = module.PriorityCachePolicy(1)
    manager._stats = dict.fromkeys(
        ["lookup_hit", "lookup_pending", "lookup_miss", "load_complete", "store_complete"], 0
    )
    manager.accepted_store_calls = 0
    manager.rejected_store_calls = 0
    manager.min_offload_priority = 0
    manager.admit_without_hint = True
    context = ReqContext("request")
    keys = [b"block"]
    calls = []

    def lookup(self, key, req_context):
        calls.append(("lookup", key, req_context))
        return result

    def complete_load(self, keys, req_context):
        calls.append(("load", keys, req_context))
        return "loaded"

    def complete_store(self, keys, req_context, success=True):
        calls.append(("store", keys, req_context, success))
        return "stored"

    def unexpected_snapshot():
        pytest.fail("normal cache operations must not scan the cache")

    monkeypatch.setattr(module.CPUOffloadingManager, "lookup", lookup, raising=False)
    monkeypatch.setattr(module.CPUOffloadingManager, "complete_load", complete_load, raising=False)
    monkeypatch.setattr(module.CPUOffloadingManager, "complete_store", complete_store, raising=False)
    monkeypatch.setattr(manager._policy, "snapshot", unexpected_snapshot)

    assert manager.lookup(keys[0], context) is result
    assert manager.complete_load(keys, context) == "loaded"
    assert manager.complete_store(keys, context, success=False) == "stored"
    assert calls == [("lookup", keys[0], context), ("load", keys, context), ("store", keys, context, False)]
    assert manager._stats[f"lookup_{state}"] == 1
    assert sum(manager._stats[f"lookup_{name}"] for name in ["hit", "pending", "miss"]) == 1
    assert manager._stats["load_complete"] == manager._stats["store_complete"] == 1

    # Full diagnostics remain available when explicitly requested.
    monkeypatch.setattr(manager._policy, "snapshot", lambda: {"entries": 1})
    assert manager.debug_snapshot()["policy"] == {"entries": 1}
