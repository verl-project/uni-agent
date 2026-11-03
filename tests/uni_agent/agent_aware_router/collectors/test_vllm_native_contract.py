# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Installed vLLM CPU producers → native msgspec wire → router cache index.

These tests allocate Python block descriptors only; they do not start an engine,
load a model, or claim GPU integration coverage. Plugin discovery and native
utility dispatch run in-process against controlled EngineCoreProc state, followed
by the production HTTP metadata adapter. No subprocess IPC or HTTP network runs.
"""

import asyncio
from importlib.metadata import entry_points
from queue import Queue
from types import SimpleNamespace

import pytest

from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.collectors.parse.vllm.native_hash import VLLMHashCodec
from uni_agent.agent_aware_router.collectors.transport.base import TransportMessage
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.fixture
def native(monkeypatch, request):
    vllm = pytest.importorskip("vllm")
    import msgspec
    import torch
    import vllm.plugins as plugins
    from vllm.distributed import kv_events
    from vllm.utils.hashing import sha256
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.core import EngineCore, EngineCoreProc, EngineShutdownState
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, SlidingWindowSpec
    from vllm.v1.request import Request

    from uni_agent.agent_aware_router.server.http_server import KVMetadataHttpServer

    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", str(int(request.param)))
    monkeypatch.setattr(kv_cache_utils, "NONE_HASH", None, raising=False)
    monkeypatch.setattr(kv_cache_utils, "_NONE_HASH_SEED", None, raising=False)
    plugin_name = "uni_agent_native_hash"
    profile_method = "uni_agent_get_native_hash_profile"
    entries = [entry for entry in entry_points(group="vllm.general_plugins") if entry.name == plugin_name]
    assert len(entries) == 1, "Install the current uni-agent distribution to register its vLLM entry point"
    assert entries[0].dist.metadata["Name"] == "uni-agent"
    monkeypatch.setenv("UNI_AGENT_ENABLE_NATIVE_HASH_METADATA", "1")
    monkeypatch.setenv("VLLM_PLUGINS", plugin_name)
    # Exercise actual discovery even if another test already loaded plugins.
    # monkeypatch restores both the loader cache and the original class method.
    monkeypatch.setattr(plugins, "plugins_loaded", False)
    monkeypatch.setattr(EngineCore, profile_method, None, raising=False)
    plugins.load_general_plugins()
    assert callable(getattr(EngineCore, profile_method))
    assert kv_cache_utils.NONE_HASH is None  # Registration must not initialize a root.
    kv_cache_utils.init_none_hash(sha256)
    root = bytes(kv_cache_utils.NONE_HASH)
    specs = [
        SlidingWindowSpec(block_size=4, num_kv_heads=1, head_size=8, dtype=torch.float16, sliding_window=9),
        FullAttentionSpec(block_size=8, num_kv_heads=1, head_size=8, dtype=torch.float16),
    ]
    config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([f"layer.{i}"], spec) for i, spec in enumerate(specs)],
    )
    engine_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=8,
            enable_prefix_caching=True,
            prefix_caching_hash_algo="sha256",
            hash_block_size=None,
            prefix_match_unit=None,
            mamba_cache_mode="align",
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(model="native-contract", max_model_len=64),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        max_in_flight_tokens=16,
        kv_transfer_config=None,
    )
    scheduler_block_size, hash_block_size = kv_cache_utils.resolve_kv_cache_block_sizes(config, engine_config)
    assert (scheduler_block_size, hash_block_size) == (8, 4)
    manager = KVCacheManager(
        config,
        max_model_len=64,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        enable_kv_cache_events=True,
    )
    core = EngineCoreProc.__new__(EngineCoreProc)
    core.vllm_config = engine_config
    core.scheduler = SimpleNamespace(kv_cache_config=config, num_lookahead_tokens=0)
    core.request_block_hasher = kv_cache_utils.get_request_block_hasher(hash_block_size, sha256)
    core.output_queue = Queue()
    core.shutdown_state = EngineShutdownState.RUNNING
    utility_calls = []

    async def call_utility(method):
        # Replace IPC only: use the actual native dispatcher and result wrapper.
        call_id = len(utility_calls)
        utility_calls.append(method)
        core._handle_client_request(EngineCoreRequestType.UTILITY, (0, call_id, method, ()))
        client_id, output = core.output_queue.get_nowait()
        utility = output.utility_output
        assert client_id == 0 and utility.call_id == call_id
        assert utility.failure_message is None, utility.failure_message
        return utility.result.result

    server = KVMetadataHttpServer.__new__(KVMetadataHttpServer)
    server.node_rank = 0
    server.engine = SimpleNamespace(
        vllm_config=engine_config, engine_core=SimpleNamespace(call_utility_async=call_utility)
    )
    metadata = asyncio.run(server.get_kv_cache_metadata())
    assert metadata["status"] == "ok", metadata
    profile = metadata["native_hash_config"]
    assert profile is not None, metadata
    assert profile["root_hash_hex"] == root.hex() and profile["unit"] == hash_block_size
    assert profile["int_hashes"] is request.param
    assert utility_calls == ["get_kv_cache_group_metadata", profile_method, "uni_agent_get_cache_reservation_profile"]
    budget = metadata["cache_reservation_profile"]
    assert budget is not None and budget["num_blocks"] == config.num_blocks
    # The real pool reserves a null block once for the shared pool, not per group.
    assert budget["capacity_blocks"] == config.num_blocks - 1 == manager.block_pool.get_num_free_blocks()
    assert [g["group_idx"] for g in budget["groups"]] == [0, 1]
    prompt = Request(
        "native-contract",
        list(range(17)),
        vllm.SamplingParams(max_tokens=1),
        None,
        block_hasher=core.request_block_hasher,
    )
    return SimpleNamespace(
        manager=manager,
        prompt=prompt,
        metadata=metadata,
        events=kv_events,
        encode=msgspec.msgpack.encode,
        int_hashes=request.param,
    )


@pytest.mark.parametrize("native", [False, True], indirect=True, ids=["bytes-hashes", "integer-hashes"])
def test_native_hybrid_events_restore_identity_and_track_residency(native):
    """Sparse native SHA hashes need Full identities and each group's residency."""
    metadata = native.metadata
    store = DataStore()
    store.set_native_hash_codec_factory(VLLMHashCodec.from_metadata)
    state = store.configure_cache_registry(metadata, replica_metadata={"a": metadata, "b": metadata})
    assert state.status == "registered"
    assert [(group.match_kind, group.block_size) for group in state.registry.groups] == [("sliding", 4), ("full", 8)]
    assert state.registry.local_hash_unit == 4
    parser = VLLMKVParser()

    def apply(events, node):
        batch = native.events.KVEventBatch(ts=1.0, events=events, data_parallel_rank=0)
        payload = native.encode(batch)
        update = parser.parse(TransportMessage(payload), node)
        assert update is not None and len(update.events) == len(events)
        store.apply_kv_update(update)

    hashes, _ = get_prefix_hashes_incremental(list(range(17)), 4, 0, 0)

    def hits():
        estimates = store.estimate_prefixes(["a", "b"], [str(h) for h in hashes], 17)
        return {node: estimate.estimated_hit_tokens for node, estimate in estimates.items()}

    manager = native.manager
    pool = manager.block_pool
    sliding_blocks = pool.get_new_blocks(4)
    full_blocks = pool.get_new_blocks(2)
    pool.cache_full_blocks(native.prompt, sliding_blocks, 0, 4, 4, 0, block_mask=[False, True, True, True])
    sparse = manager.take_events()
    pool.cache_full_blocks(native.prompt, full_blocks, 0, 2, 8, 1)
    full = manager.take_events()
    assert len(sparse) == len(full) == 1
    assert isinstance(sparse[0], native.events.BlockStored)
    assert sparse[0].kv_cache_spec_kind == "sliding_window" and len(sparse[0].block_hashes) == 3
    assert full[0].kv_cache_spec_kind == "full_attention" and len(full[0].block_hashes) == 2
    assert isinstance(full[0].block_hashes[0], int if native.int_hashes else bytes)

    apply(sparse, "a")
    assert hits() == {"a": 0, "b": 0}
    # Identities can cross replicas; Full residency cannot.
    apply(full, "b")
    assert hits() == {"a": 0, "b": 0}
    apply(full, "a")
    assert hits() == {"a": 16, "b": 0}

    pool.evict_blocks({sliding_blocks[2].block_id})
    removed = manager.take_events()
    assert len(removed) == 1 and isinstance(removed[0], native.events.BlockRemoved)
    apply(removed, "a")
    assert hits() == {"a": 0, "b": 0}
    assert store.per_replica_block_counts()["a"] > 0

    pool.free_blocks(sliding_blocks + full_blocks)
    assert pool.reset_prefix_cache() is True
    cleared = manager.take_events()
    assert len(cleared) == 1 and isinstance(cleared[0], native.events.AllBlocksCleared)
    apply(cleared, "a")
    # A native Clear affects only its publisher, including Full-group entries.
    assert store.per_replica_block_counts().get("a", 0) == 0
    apply(sparse, "b")
    assert hits() == {"a": 0, "b": 16}
