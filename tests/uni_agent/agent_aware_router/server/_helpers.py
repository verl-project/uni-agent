"""Small vLLM cache layout shared by the engine metadata contract tests."""


def hybrid_cache_config():
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, SlidingWindowSpec

    specs = [
        SlidingWindowSpec(block_size=4, num_kv_heads=1, head_size=8, dtype=torch.float16, sliding_window=9),
        FullAttentionSpec(block_size=8, num_kv_heads=1, head_size=8, dtype=torch.float16),
    ]
    return KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([f"layer.{i}"], spec) for i, spec in enumerate(specs)],
    )
