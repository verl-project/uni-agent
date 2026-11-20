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
import inspect
from importlib.metadata import entry_points
from queue import Queue
from types import SimpleNamespace

import pytest

from uni_agent.agent_aware_router.collectors.parse.vllm.kv import VLLMKVParser
from uni_agent.agent_aware_router.collectors.parse.vllm.native_hash import VLLMHashCodec
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental


@pytest.fixture
def single_full_drop():
    return None  # Use the hybrid cache fixture unless this parameter selects a single Full group.


@pytest.fixture
def hash_lookahead():
    return 0


@pytest.fixture
def native(monkeypatch, request, hash_lookahead, single_full_drop):
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

    if hash_lookahead and "num_hash_lookahead_tokens" not in inspect.signature(KVCacheManager).parameters:
        pytest.skip("Installed vLLM does not support lookahead block hashes")
    hash_options = {"num_hash_lookahead_tokens": hash_lookahead} if hash_lookahead else {}
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
    monkeypatch.setattr(plugins, "plugins_loaded", False)
    for name in (profile_method, "uni_agent_get_cache_reservation_profile"):
        monkeypatch.setattr(EngineCore, name, None, raising=False)
    plugins.load_general_plugins()
    assert callable(getattr(EngineCore, profile_method))
    assert kv_cache_utils.NONE_HASH is None  # Registration must not initialize a root.
    kv_cache_utils.init_none_hash(sha256)
    root = bytes(kv_cache_utils.NONE_HASH)
    specs = [
        SlidingWindowSpec(block_size=4, num_kv_heads=1, head_size=8, dtype=torch.float16, sliding_window=9),
        FullAttentionSpec(block_size=8, num_kv_heads=1, head_size=8, dtype=torch.float16),
    ]
    if single_full_drop is not None:
        specs = specs[1:]  # Unitary Full cache with 8-token blocks.
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
        num_hash_lookahead_tokens=hash_lookahead,
        kv_transfer_config=None,
    )
    scheduler_block_size, hash_block_size = kv_cache_utils.resolve_kv_cache_block_sizes(config, engine_config)
    assert (scheduler_block_size, hash_block_size) == (8, min(spec.block_size for spec in specs))
    manager = KVCacheManager(
        config,
        max_model_len=64,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        enable_kv_cache_events=True,
        use_eagle=bool(single_full_drop),
        **hash_options,
    )
    core = EngineCoreProc.__new__(EngineCoreProc)
    core.vllm_config = engine_config
    core.scheduler = SimpleNamespace(kv_cache_config=config, kv_cache_manager=manager, num_lookahead_tokens=0)
    core.request_block_hasher = kv_cache_utils.get_request_block_hasher(hash_block_size, sha256, **hash_options)
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
    assert profile["hit_alignment"] == scheduler_block_size
    assert profile["int_hashes"] is request.param
    assert utility_calls == ["get_kv_cache_group_metadata", profile_method, "uni_agent_get_cache_reservation_profile"]

    def make_request(token_ids):
        return Request(
            "native-contract",
            token_ids,
            vllm.SamplingParams(max_tokens=1),
            None,
            block_hasher=core.request_block_hasher,
        )

    prompt = make_request(list(range(16 + max(1, hash_lookahead))))
    return SimpleNamespace(
        manager=manager,
        make_request=make_request,
        prompt=prompt,
        metadata=metadata,
        events=kv_events,
        encode=msgspec.msgpack.encode,
        int_hashes=request.param,
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("native", [False, True], indirect=True, ids=["bytes-hashes", "integer-hashes"])
def test_native_hybrid_events_restore_identity_and_track_residency(native, install_metadata):
    """Sparse native SHA hashes need Full identities and each group's residency."""
    metadata = native.metadata
    store = DataStore()
    store.set_native_hash_codec_factory(VLLMHashCodec.from_metadata)
    install_metadata(store, "a", metadata)
    state = install_metadata(store, "b", metadata)
    assert state.status == "registered"
    assert [(group.match_kind, group.block_size) for group in state.registry.groups] == [("sliding", 4), ("full", 8)]
    assert state.registry.local_hash_unit == 4
    parser = VLLMKVParser()

    def apply(events, node):
        batch = native.events.KVEventBatch(ts=1.0, events=events, data_parallel_rank=0)
        payload = native.encode(batch)
        update = parser.parse(payload, node)
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


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("native", [False], indirect=True)
@pytest.mark.parametrize(("hash_lookahead", "single_full_drop"), [(1, None), (2, None), (0, False), (0, True)])
def test_prefix_hit_matches_vllm_coordinator(native, hash_lookahead, single_full_drop, install_metadata):
    """Compare lookahead and EAGLE drop policies against the installed vLLM coordinator."""
    from uni_agent.agent_aware_router.utils.prefix_cache import resolve_prefix_hashes

    store = DataStore()
    store.set_native_hash_codec_factory(VLLMHashCodec.from_metadata)
    assert native.metadata["native_hash_config"]["lookahead_tokens"] == hash_lookahead
    install_metadata(store, "a", native.metadata)
    manager = native.manager
    pool = manager.block_pool
    for group, group_meta in enumerate(native.metadata["groups"]):
        block_size = group_meta["block_size"]
        n = 16 // block_size
        pool.cache_full_blocks(native.prompt, pool.get_new_blocks(n), 0, n, block_size, group)
    events = manager.take_events()
    if hash_lookahead:
        assert all(len(event.lookahead_token_ids) == len(event.block_hashes) for event in events)
    batch = native.events.KVEventBatch(ts=1.0, events=events, data_parallel_rank=0)
    store.apply_kv_update(VLLMKVParser().parse(native.encode(batch), "a"))

    original = list(native.prompt.all_token_ids)
    first_fork = original.copy()
    first_fork[8 + hash_lookahead - 1] += 100
    cases = ((original, 16), (original[:-1], 8), (original[:-1] + [999], 8), (first_fork, 0))
    if single_full_drop is not None:
        assert native.metadata["native_hash_config"]["drop_eagle_group_ids"] == ([0] if single_full_drop else [])
        cases = ((original[:9], 0 if single_full_drop else 8), (original, 8 if single_full_drop else 16))
    for prompt, expected in cases:
        query = native.make_request(prompt)
        vllm_hits = manager.get_computed_blocks(query)[1]
        hashes = resolve_prefix_hashes(prompt, None, store)
        estimate = store.estimate_prefixes(["a"], hashes, len(prompt))["a"]
        assert vllm_hits == expected
        assert estimate.estimated_hit_tokens == vllm_hits
