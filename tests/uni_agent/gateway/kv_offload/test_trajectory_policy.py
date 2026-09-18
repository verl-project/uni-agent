from __future__ import annotations

import ctypes
import importlib
import sys
import types
from dataclasses import dataclass


def _load_policy(monkeypatch):
    base = types.ModuleType("vllm.v1.kv_offload.base")
    base.OffloadKey = bytes

    @dataclass
    class ReqContext:
        req_id: str
        kv_transfer_params: dict | None = None

    base.ReqContext = ReqContext
    policy_base = types.ModuleType("vllm.v1.kv_offload.cpu.policies.base")

    class BlockStatus(ctypes.Structure):
        _fields_ = [("ref_cnt", ctypes.c_int32), ("block_id", ctypes.c_int64)]

        def __init__(self, block_id):
            super().__init__()
            self.ref_cnt = 0
            self.block_id = block_id

    class CachePolicy:
        def __init__(self, cache_capacity):
            self.cache_capacity = cache_capacity

    policy_base.BlockStatus = BlockStatus
    policy_base.CachePolicy = CachePolicy
    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.kv_offload": types.ModuleType("vllm.v1.kv_offload"),
        "vllm.v1.kv_offload.base": base,
        "vllm.v1.kv_offload.cpu": types.ModuleType("vllm.v1.kv_offload.cpu"),
        "vllm.v1.kv_offload.cpu.policies": types.ModuleType("vllm.v1.kv_offload.cpu.policies"),
        "vllm.v1.kv_offload.cpu.policies.base": policy_base,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("uni_agent.gateway.kv_offload", None)
    sys.modules.pop("uni_agent.gateway.kv_offload.trajectory_policy", None)
    module = importlib.import_module("uni_agent.gateway.kv_offload.trajectory_policy")
    return module, BlockStatus, ReqContext


def test_policy_preserves_leased_chain_and_evicts_expired_suffix_first(monkeypatch):
    module, BlockStatus, ReqContext = _load_policy(monkeypatch)
    policy = module.TrajectoryAwareCachePolicy(6)
    leased = [b"a0", b"a1", b"a2"]
    expired = [b"b0", b"b1", b"b2"]
    for index, key in enumerate([*leased, *expired]):
        policy.insert(key, BlockStatus(index))
    policy.touch(
        leased,
        ReqContext(
            "a",
            {
                "uni_agent": {
                    "schema_version": 1,
                    "trajectory_id": "a",
                    "lease_until": 10**12,
                    "reuse_probability": 1.0,
                }
            },
        ),
    )
    policy.touch(
        expired,
        ReqContext(
            "b",
            {
                "uni_agent": {
                    "schema_version": 1,
                    "trajectory_id": "b",
                    "lease_until": 0,
                    "reuse_probability": 1.0,
                }
            },
        ),
    )

    evicted = policy.evict(2, protected=set())

    assert [key for key, _ in evicted] == [b"b2", b"b1"]
    assert policy.get(b"b0") is not None
    assert all(policy.get(key) is not None for key in leased)


def test_policy_eviction_is_atomic_when_blocks_are_busy(monkeypatch):
    module, BlockStatus, _ = _load_policy(monkeypatch)
    policy = module.TrajectoryAwareCachePolicy(2)
    first, second = BlockStatus(0), BlockStatus(1)
    second.ref_cnt = 1
    policy.insert(b"first", first)
    policy.insert(b"second", second)

    assert policy.evict(2, protected=set()) is None
    assert policy.get(b"first") is first
    assert policy.get(b"second") is second
