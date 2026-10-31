"""Shared cache metadata and event fixtures for router tests."""

import msgpack
import pytest

from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser


@pytest.fixture
def metadata():
    def make_metadata():
        return {
            "schema_version": 1,
            "status": "ok",
            "vllm_version": "0.29.0",
            "groups": [
                {"group_idx": 0, "kind": "full_attention", "block_size": 64, "sliding_window": None},
                {"group_idx": 3, "kind": "mamba", "block_size": 96, "sliding_window": None},
            ],
            "cache_config": {"enable_prefix_caching": True, "mamba_cache_mode": "align", "prefix_match_unit": None},
            "parallel_config": {
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "decode_context_parallel_size": 1,
                "prefill_context_parallel_size": 1,
            },
            "model_config": {"model": "test-model", "attention_chunk_size": 8192},
        }

    return make_metadata


@pytest.fixture
def layout(metadata):
    def make_layout(groups=(("full_attention", 4), ("mamba", 8)), unit=2, window=5, chunk=8, mode="align"):
        raw = metadata()
        raw["groups"] = [
            dict(group_idx=i, kind=kind, block_size=size, sliding_window=window if "sliding" in kind else None)
            for i, (kind, size) in enumerate(groups)
        ]
        raw["cache_config"].update(prefix_match_unit=unit, mamba_cache_mode=mode)
        raw["model_config"]["attention_chunk_size"] = chunk
        return raw

    return make_layout


@pytest.fixture
def stored():
    def make_stored(group=0, kind="full_attention", block=4, tokens=None, hashes=None, parent=None, **kwargs):
        tokens = list(range(16)) if tokens is None else tokens
        hashes = [f"r{i}" for i in range(block, len(tokens) + 1, block)] if hashes is None else hashes
        event = dict(
            type="BlockStored",
            group_idx=group,
            kv_cache_spec_kind=kind,
            block_size=block,
            token_ids=tokens,
            block_hashes=hashes,
            parent_block_hash=parent,
            medium="GPU",
        )
        event.update(kwargs)
        return event

    return make_stored


@pytest.fixture
def removed():
    def make_removed(group, hashes, **kwargs):
        return dict(type="BlockRemoved", group_idx=group, block_hashes=hashes, medium="GPU", **kwargs)

    return make_removed


@pytest.fixture
def parse():
    def make_parse(events, node="a"):
        parser = VLLMKVParser(group_aware=True)
        update = parser.parse(msgpack.packb(dict(ts=0, events=events, data_parallel_rank=0)), node)
        assert update is not None
        return update.group_events

    return make_parse
