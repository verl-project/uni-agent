"""Real CUDA transfer lifecycle, selected by the GPU CI job."""

import pytest

from uni_agent.gateway.agent_hint import AgentRuntimeHint


@pytest.mark.gpu
@pytest.mark.level0
@pytest.mark.parametrize("offload_configs", [1, 2], indirect=True)
def test_native_handlers_roundtrip_kv_data(api, offload_configs):
    import torch
    from vllm.v1.kv_offload.base import CanonicalKVCacheRef, CanonicalKVCaches, CanonicalKVCacheTensor, GPULoadStoreSpec
    from vllm.v1.kv_offload.worker.worker import OffloadingWorker

    if not torch.cuda.is_available():
        pytest.skip("Native GPU/CPU transfer validation requires a CUDA-alike GPU")
    config, caches = offload_configs
    config.kv_transfer_config.kv_connector_extra_config["store_threshold"] = 0
    spec = api.factory.create_spec(config, caches)
    manager = spec.get_manager()
    factor = spec.block_size_factor
    page_size = caches.kv_cache_groups[0].kv_cache_spec.page_size_bytes
    tensor = torch.arange(4 * page_size, dtype=torch.int64, device="cuda").to(torch.int8).reshape(4, page_size)
    expected = tensor[:factor].clone()
    canonical = CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(tensor=tensor, page_size_bytes=page_size)],
        group_data_refs=[[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page_size)]],
    )
    runtime = AgentRuntimeHint(True, True, True, 40_000, 65_536, 20_000, False, 1)
    context = api.context("transfer", {"agent_hint": runtime.to_dict()})
    key = api.key(b"roundtrip", 0)
    worker = OffloadingWorker()
    manager.on_new_request(context)
    try:
        handlers = list(spec.get_handlers(canonical))
        assert list(spec.get_handlers(canonical)) == handlers
        for source, destination, handler in handlers:
            worker.register_handler(source, destination, handler)
        store = manager.prepare_store([key], context)
        assert store.keys_to_store == [key]
        assert worker.transfer_async(1, (GPULoadStoreSpec(list(range(factor)), [factor], [0]), store.store_spec))
        worker.wait({1})
        finished = worker.get_finished()
        assert [(result.job_id, result.success) for result in finished] == [(1, True)]
        assert finished[0].transfer_size == page_size * factor
        manager.complete_store([key], context)
        assert manager.lookup(key, context) is True

        tensor[:factor].zero_()
        load = manager.prepare_load([key], context)
        assert worker.transfer_async(2, (load, GPULoadStoreSpec(list(range(factor)), [factor], [0])))
        worker.wait({2})
        finished = worker.get_finished()
        assert [(result.job_id, result.success) for result in finished] == [(2, True)]
        assert finished[0].transfer_size == page_size * factor
        manager.complete_load([key], context)
        torch.testing.assert_close(tensor[:factor], expected)
        assert manager._policy.get(key).ref_cnt == 0
        assert worker.get_finished() == []
    finally:
        worker.shutdown()
        manager.on_request_finished(context)
