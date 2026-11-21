"""Shared cache metadata and event fixtures for router tests."""

import math

import msgpack
import pytest

from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.store.data_store import DataStore


@pytest.fixture
def install_metadata():
    def install(store, replica_id, metadata):
        cache = store._kv if isinstance(store, DataStore) else store
        cache.refresh_replica_metadata(replica_id, metadata)
        cache.mark_replica_ready(replica_id, revision=cache.replay_revision(replica_id))
        return cache.registry_state(replica_id)

    return install


@pytest.fixture
def metadata():
    def make_metadata():
        return {
            "schema_version": 1,
            "status": "ok",
            "vllm_version": "0.29.0",
            "native_hash_config": {"schema_version": 1, "vllm_version": "0.29.0", "unit": 32, "hit_alignment": 32},
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
    def make_layout(
        groups=(("full_attention", 4), ("mamba", 8)), unit=2, window=5, chunk=8, mode="align", *, alignment=None
    ):
        raw = metadata()
        raw["groups"] = [
            dict(group_idx=i, kind=kind, block_size=size, sliding_window=window if "sliding" in kind else None)
            for i, (kind, size) in enumerate(groups)
        ]
        raw["cache_config"].update(prefix_match_unit=unit, mamba_cache_mode=mode)
        raw["native_hash_config"].update(unit=unit, hit_alignment=unit if alignment is None else alignment)
        raw["model_config"]["attention_chunk_size"] = chunk
        # Static native budgets, separate from matching policy and token capacity.
        raw["cache_reservation_profile"] = {
            "schema_version": 1,
            "max_model_len": 64,
            "groups": [
                {
                    "group_idx": i,
                    "max_blocks_per_request": (2 if kind == "mamba" and mode != "all" else math.ceil(64 / size)),
                }
                for i, (kind, size) in enumerate(groups)
            ],
        }
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
        parser = VLLMKVParser()
        update = parser.parse(msgpack.packb([0, events, 0]), node)
        assert update is not None
        return update.events

    return make_parse
