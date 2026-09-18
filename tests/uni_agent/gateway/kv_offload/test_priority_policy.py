from __future__ import annotations

import ctypes
import importlib
import sys
import types
from dataclasses import dataclass


def _load(monkeypatch):
    base = types.ModuleType("vllm.v1.kv_offload.base")
    base.OffloadKey = bytes

    @dataclass
    class ReqContext:
        req_id: str
        kv_transfer_params: dict | None = None

    base.ReqContext = ReqContext
    manager_module = types.ModuleType("vllm.v1.kv_offload.cpu.manager")
    manager_module.CPUOffloadingManager = type("CPUOffloadingManager", (), {})
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
    sys.modules.pop("uni_agent.gateway.kv_offload.priority_policy", None)
    module = importlib.import_module("uni_agent.gateway.kv_offload.priority_policy")
    return module, BlockStatus, ReqContext


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
    hint, present = module._parse_hint(
        ReqContext(
            "request",
            {
                "agent_hint": {
                    "schema_version": 1,
                    "trajectory_id": "trajectory-1",
                    "kv_priority": 87,
                    "lease_until": 1234.5,
                    "tools_available": True,
                }
            },
        )
    )

    assert present is True
    assert hint.trajectory_id == "trajectory-1"
    assert hint.priority == 87
    assert hint.lease_until == 1234.5


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
