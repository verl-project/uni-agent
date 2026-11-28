"""Model-free inputs for real vLLM 0.23 offloading integration tests."""

import sys
from importlib.metadata import version
from types import SimpleNamespace

import pytest


@pytest.fixture
def api():
    if sys.platform == "win32":
        pytest.skip("vLLM 0.23.0 requires Linux; exercised in the Linux CPU CI job")
    # A missing or different engine in Linux CI must fail rather than skip.
    assert version("vllm") == "0.23.0"
    from vllm.v1.kv_offload.base import ReqContext, make_offload_key
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
    from vllm.v1.kv_offload.factory import OffloadingSpecFactory

    from uni_agent.gateway.kv_offload.priority_policy import PriorityCPUOffloadingManager, PriorityGPUOffloadingSpec

    assert PriorityCPUOffloadingManager.__bases__ == (CPUOffloadingManager,)
    assert PriorityGPUOffloadingSpec.__bases__ == (CPUOffloadingSpec,)
    return SimpleNamespace(
        manager=PriorityCPUOffloadingManager,
        spec=PriorityGPUOffloadingSpec,
        native_spec=CPUOffloadingSpec,
        factory=OffloadingSpecFactory,
        context=ReqContext,
        key=make_offload_key,
    )


@pytest.fixture
def offload_configs(api, request):
    import torch
    from vllm.config import KVTransferConfig
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

    factor = getattr(request, "param", 1)
    attention = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16)
    page_size = attention.page_size_bytes
    transfer = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "spec_name": "PriorityGPUOffloadingSpec",
            "spec_module_path": "uni_agent.gateway.kv_offload.priority_policy",
            "cpu_bytes_to_use": 4 * page_size * factor,
            "block_size": 16 * factor,
            "store_threshold": 2,
            "max_tracker_size": 16,
            "agent_hint_config": {
                "active_lease_seconds": 45.0,
                "min_offload_priority": 50,
                "admit_without_hint": False,
            },
        },
    )
    # Only the outer model-free config containers are structural inputs.
    # Factory, spec constructors, cache geometry and manager are all native.
    config = SimpleNamespace(
        kv_transfer_config=transfer,
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(world_size=1, decode_context_parallel_size=1, prefill_context_parallel_size=1),
        kv_events_config=SimpleNamespace(enable_kv_cache_events=True),
    )
    caches = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[KVCacheTensor(size=4 * page_size, shared_by=["layer"])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer"], kv_cache_spec=attention)],
    )
    return config, caches
