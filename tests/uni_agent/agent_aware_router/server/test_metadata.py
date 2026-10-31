"""Native metadata forwarding without starting GPU engines."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("vllm")

from uni_agent.agent_aware_router.server.http_server import KVMetadataHttpServer

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.fixture
def server():
    instance = object.__new__(KVMetadataHttpServer)
    instance.node_rank = 0
    instance.engine = SimpleNamespace(
        engine_core=SimpleNamespace(
            call_utility_async=AsyncMock(
                return_value=[
                    {"group_idx": 0, "kind": "full_attention", "block_size": 80, "sliding_window": None},
                    {"group_idx": 1, "kind": "mamba", "block_size": 80, "sliding_window": None},
                ]
            )
        ),
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(
                enable_prefix_caching=True,
                mamba_cache_mode="align",
                prefix_match_unit=16,
                kv_cache_size_tokens=20480,
            ),
            parallel_config=SimpleNamespace(tensor_parallel_size=2, data_parallel_size=1),
            model_config=SimpleNamespace(
                model="resolved-model",
                attention_chunk_size=4096,
                hf_text_config=SimpleNamespace(attention_chunk_size=8192),
            ),
        ),
    )
    return instance


def read(server):
    return asyncio.run(server.get_kv_cache_metadata())


@pytest.mark.parametrize("prefix_policy", [False, True], ids=["original-rpc", "prefix-policy-rpc"])
def test_resolved_config_and_native_groups_are_preserved(server, prefix_policy):
    groups = server.engine.engine_core.call_utility_async.return_value
    if prefix_policy:
        groups[0].update(prefix_cacheable=True, prefix_replay_tokens=0)
        groups[1].update(kind="sliding_window_mla", prefix_cacheable=False, prefix_replay_tokens=128)
    expected_groups = deepcopy(groups)
    result = read(server)
    assert result["status"] == "ok"
    assert result["schema_version"] == 1
    assert [g["group_idx"] for g in result["groups"]] == [0, 1]
    assert result["model_config"]["attention_chunk_size"] == 4096
    assert result["cache_config"]["hash_block_size"] is None
    assert result["cache_config"]["prefix_match_unit"] == 16
    assert result["parallel_config"]["prefill_context_parallel_size"] is None
    assert result["kv_cache_capacity_tokens"] == 20480
    assert result["capacity_source"] == "native_config"
    assert result["groups"] == expected_groups
    server.engine.engine_core.call_utility_async.assert_awaited_once_with("get_kv_cache_group_metadata")


def test_chunk_fallback_only_when_property_is_missing(server):
    model = server.engine.vllm_config.model_config
    model.attention_chunk_size = None
    assert read(server)["model_config"]["attention_chunk_size"] is None
    del model.attention_chunk_size
    result = read(server)["model_config"]
    assert result["attention_chunk_size"] == 8192
    assert result["attention_chunk_size_source"] == "model_config.hf_text_config.attention_chunk_size"


@pytest.mark.parametrize("rank,status", [(0, "not_ready"), (1, "headless")])
def test_no_engine_is_not_an_empty_group_list(server, rank, status):
    del server.engine
    server.node_rank = rank
    result = read(server)
    assert result["status"] == status
    assert "groups" not in result


def test_missing_config(server):
    del server.engine.vllm_config
    assert read(server)["reason"] == "engine_config_unavailable"


def test_rpc_failure_and_timeout(server):
    server.engine.engine_core.call_utility_async.side_effect = AttributeError("missing native method")
    result = read(server)
    assert result["status"] == "error"
    assert result["error_type"] == "AttributeError"
    assert "groups" not in result

    async def slow(*args):
        await asyncio.sleep(10)

    server.engine.engine_core.call_utility_async.side_effect = slow
    server._METADATA_RPC_TIMEOUT = 0.001
    assert read(server)["reason"] == "rpc_timeout"


def test_native_hash_unit_excludes_unknown_auxiliary_groups_in_original_metadata(server, monkeypatch):
    import vllm

    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setattr(vllm.envs, "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", False, raising=False)
    cache = server.engine.vllm_config.cache_config
    cache.prefix_caching_hash_algo = "sha256"
    cache.prefix_match_unit = None
    groups = [
        dict(group_idx=0, kind="full_attention", block_size=32, sliding_window=None),
        dict(group_idx=1, kind="sliding_window", block_size=16, sliding_window=1024),
        dict(group_idx=2, kind="unknown", block_size=8, sliding_window=None),
    ]
    server.engine.engine_core.call_utility_async.return_value = deepcopy(groups)
    result = read(server)
    assert result["native_hash_config"]["unit"] == 16
    assert result["groups"] == groups


def test_missing_capacity_keeps_group_metadata(server):
    del server.engine.vllm_config.cache_config.kv_cache_size_tokens
    result = read(server)
    assert result["status"] == "ok" and len(result["groups"]) == 2
    assert result["kv_cache_capacity_tokens"] is None
    assert result["capacity_source"] is None


@pytest.mark.parametrize(
    "seed,version,unit_field",
    [
        ("42", "0.23.0", "hash_block_size"),
        ("42", "0.31.0", "prefix_match_unit"),
        (None, "0.29.0", "prefix_match_unit"),
        ("random", "0.29.0", "prefix_match_unit"),
    ],
)
def test_native_hash_profile_requires_explicit_deterministic_seed(seed, version, unit_field, server, monkeypatch):
    import vllm

    monkeypatch.setattr(vllm, "__version__", version)
    cache = server.engine.vllm_config.cache_config
    cache.prefix_caching_hash_algo = "sha256"
    cache.prefix_match_unit = None
    setattr(cache, unit_field, 16)
    if seed is None:
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    else:
        monkeypatch.setenv("PYTHONHASHSEED", seed)
    profile = read(server)["native_hash_config"]
    if seed == "42":
        assert profile["seed"] == "42" and profile["unit"] == 16
        assert profile["algorithm"] == "sha256"
    else:
        assert profile is None
