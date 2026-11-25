"""Native metadata forwarding without starting GPU engines."""

import asyncio
import os
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

pytest.importorskip("vllm")

from vllm.v1.engine.core import EngineCore

from tests.uni_agent.agent_aware_router.server._helpers import hybrid_cache_config
from uni_agent.agent_aware_router.server.http_server import KVMetadataHttpServer


@pytest.fixture
def server(monkeypatch):
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
            ),
            parallel_config=SimpleNamespace(tensor_parallel_size=2, data_parallel_size=1),
            model_config=SimpleNamespace(
                model="resolved-model",
                attention_chunk_size=4096,
                hf_text_config=SimpleNamespace(attention_chunk_size=8192),
            ),
        ),
    )
    instance.native_profile = {
        "schema_version": 1,
        "vllm_version": "engine-version",
        "enable_prefix_caching": True,
        "algorithm": "sha256",
        "root_hash_hex": "ab" * 32,
        "unit": 16,
        "hit_alignment": 32,
        "pickle_protocol": 5,
        "int_hashes": False,
        "lookahead_tokens": 0,
    }
    instance.reservation_profile = {
        "schema_version": 1,
        "num_blocks": 2048,
        "capacity_blocks": 2047,
        "max_model_len": 4096,
        "groups": [{"group_idx": 0, "max_blocks_per_request": 52}, {"group_idx": 1, "max_blocks_per_request": 2}],
    }
    rpc = instance.engine.engine_core.call_utility_async
    monkeypatch.setattr(EngineCore, "get_kv_cache_group_metadata", lambda self: rpc.return_value, raising=False)

    async def query(method):
        if method == "get_kv_cache_group_metadata":
            return rpc.return_value
        return {
            "uni_agent_get_native_hash_profile": instance.native_profile,
            "uni_agent_get_cache_reservation_profile": instance.reservation_profile,
        }[method]

    rpc.side_effect = query
    return instance


def read(server):
    return asyncio.run(server.get_kv_cache_metadata())


@pytest.mark.cpu
@pytest.mark.level0
def test_installed_engine_exports_cache_groups():
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    if not hasattr(EngineCore, "get_kv_cache_group_metadata"):
        pytest.skip("Installed vLLM does not expose cache group metadata")
    config = hybrid_cache_config()
    manager = KVCacheManager(config, max_model_len=64, scheduler_block_size=8, hash_block_size=4)
    core = EngineCore.__new__(EngineCore)
    core.scheduler = SimpleNamespace(kv_cache_config=config, kv_cache_manager=manager)

    groups = core.get_kv_cache_group_metadata()
    assert [(g["group_idx"], g["kind"], g["block_size"], g["sliding_window"]) for g in groups] == [
        (0, "sliding_window", 4, 9),
        (1, "full_attention", 8, None),
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_resolved_config_and_native_groups_are_preserved(server):
    groups = server.engine.engine_core.call_utility_async.return_value
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
    assert result["groups"] == expected_groups
    server.engine.engine_core.call_utility_async.assert_has_awaits(
        [
            call("get_kv_cache_group_metadata"),
            call("uni_agent_get_native_hash_profile"),
            call("uni_agent_get_cache_reservation_profile"),
        ]
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("engine_fields", "expected"),
    [({"enable_prefix_caching": False}, False), ({}, None), ({"enable_prefix_caching": "false"}, None)],
    ids=["disabled-by-engine", "older-plugin", "invalid-flag"],
)
def test_prefix_caching_uses_engine_value(server, engine_fields, expected):
    assert server.engine.vllm_config.cache_config.enable_prefix_caching is True
    server.native_profile = engine_fields
    assert read(server)["cache_config"]["enable_prefix_caching"] is expected


@pytest.mark.cpu
@pytest.mark.level0
def test_chunk_fallback_only_when_property_is_missing(server):
    model = server.engine.vllm_config.model_config
    model.attention_chunk_size = None
    assert read(server)["model_config"]["attention_chunk_size"] is None
    del model.attention_chunk_size
    result = read(server)["model_config"]
    assert result["attention_chunk_size"] == 8192


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("rank,status", [(0, "not_ready"), (1, "headless")])
def test_no_engine_is_not_an_empty_group_list(server, rank, status):
    del server.engine
    server.node_rank = rank
    result = read(server)
    assert result["status"] == status
    assert "groups" not in result


@pytest.mark.cpu
@pytest.mark.level0
def test_missing_config(server):
    del server.engine.vllm_config
    assert read(server)["reason"] == "engine_config_unavailable"


@pytest.mark.cpu
@pytest.mark.level0
def test_missing_group_metadata_method_is_unsupported(server, monkeypatch):
    monkeypatch.delattr(EngineCore, "get_kv_cache_group_metadata")
    result = read(server)
    assert result["status"] == "unsupported"
    assert result["reason"] == "group_metadata_method_unavailable"
    server.engine.engine_core.call_utility_async.assert_not_called()


@pytest.mark.cpu
@pytest.mark.level0
def test_rpc_failure_and_timeout(server):
    # vLLM's utility client wraps the remote failure message in Exception.
    server.engine.engine_core.call_utility_async.side_effect = Exception("group query failed")
    result = read(server)
    assert result["status"] == "error"
    assert result["reason"] == "metadata_query_failed"
    assert result["message"] == "group query failed"
    assert "groups" not in result

    async def slow(*args):
        await asyncio.sleep(10)

    server.engine.engine_core.call_utility_async.side_effect = slow
    server._METADATA_RPC_TIMEOUT = 0.001
    assert read(server)["reason"] == "rpc_timeout"


@pytest.mark.cpu
@pytest.mark.level0
def test_native_hash_profile_is_forwarded_independently_of_http_environment(server, monkeypatch):
    import vllm

    monkeypatch.setenv("PYTHONHASHSEED", "731")
    monkeypatch.setattr(vllm.envs, "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", True, raising=False)
    result = read(server)
    assert result["native_hash_config"] == server.native_profile
    assert result["native_hash_config"] is not server.native_profile
    assert "seed" not in result["native_hash_config"]
    assert result["cache_reservation_profile"] == server.reservation_profile
    assert result["cache_reservation_profile"] is not server.reservation_profile


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("raw,expected", [(2048, 2047), (1, 0), (True, None), (None, None)])
def test_old_reservation_profile_normalizes_pool_capacity_at_vllm_boundary(server, raw, expected):
    del server.reservation_profile["capacity_blocks"]
    if raw is None:
        del server.reservation_profile["num_blocks"]
    else:
        server.reservation_profile["num_blocks"] = raw
    result = read(server)
    assert result["status"] == "ok"
    assert result["cache_reservation_profile"]["capacity_blocks"] == expected
    assert "capacity_blocks" not in server.reservation_profile  # Detached output only.


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("failure", ["missing", "unsupported", "not_json"])
@pytest.mark.parametrize("method", ["uni_agent_get_native_hash_profile", "uni_agent_get_cache_reservation_profile"])
def test_optional_profile_rpc_failure_keeps_groups_and_other_profile(server, failure, method):
    rpc = server.engine.engine_core.call_utility_async
    groups = deepcopy(rpc.return_value)

    async def query(requested):
        if requested == "get_kv_cache_group_metadata":
            return groups
        if requested != method:
            return (
                server.native_profile
                if requested == "uni_agent_get_native_hash_profile"
                else server.reservation_profile
            )
        if failure == "missing":
            raise Exception("plugin is absent or filtered")
        if failure == "not_json":
            return {"root_hash_hex": b"invalid wire value"}
        return None

    rpc.side_effect = query
    result = read(server)
    assert result["status"] == "ok"
    assert result["groups"] == groups
    failed_key = "native_hash_config" if method == "uni_agent_get_native_hash_profile" else "cache_reservation_profile"
    assert result[failed_key] is None
    if failed_key == "native_hash_config":
        assert result["cache_config"]["enable_prefix_caching"] is None
        assert result["cache_reservation_profile"] == server.reservation_profile
    else:
        assert result["native_hash_config"] == server.native_profile


@pytest.mark.cpu
@pytest.mark.level0
def test_hanging_optional_profile_returns_groups_within_discovery_budget(server):
    rpc = server.engine.engine_core.call_utility_async
    groups = deepcopy(rpc.return_value)

    async def query(method):
        if method == "get_kv_cache_group_metadata":
            return groups
        await asyncio.Future()

    async def discover():
        return await asyncio.wait_for(server.get_kv_cache_metadata(), timeout=0.5)

    rpc.side_effect = query
    server._METADATA_RPC_TIMEOUT = 0.01
    result = asyncio.run(discover())
    assert result["status"] == "ok"
    assert result["groups"] == groups
    assert result["native_hash_config"] is None
    assert result["cache_reservation_profile"] is None
    assert {entry.args[0] for entry in rpc.await_args_list} == {
        "get_kv_cache_group_metadata",
        "uni_agent_get_native_hash_profile",
        "uni_agent_get_cache_reservation_profile",
    }


@pytest.mark.cpu
@pytest.mark.level0
def test_late_group_result_does_not_delay_optional_profiles(server):
    rpc = server.engine.engine_core.call_utility_async
    groups = deepcopy(rpc.return_value)

    async def query(method):
        if method == "get_kv_cache_group_metadata":
            await asyncio.sleep(0.01)
            return groups
        if method == "uni_agent_get_native_hash_profile":
            return server.native_profile
        return server.reservation_profile

    rpc.side_effect = query
    result = read(server)
    assert result["status"] == "ok"
    assert result["groups"] == groups
    assert result["native_hash_config"] == server.native_profile
    assert result["cache_reservation_profile"] == server.reservation_profile


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "extra_vars", ["USER_ENGINE_SETTING", "USER_ENGINE_SETTING, UNI_AGENT_ENABLE_NATIVE_HASH_METADATA"]
)
def test_server_forwards_plugin_flag_before_engine_initialization(monkeypatch, extra_vars):
    from vllm.ray.ray_env import get_env_vars_to_copy

    from uni_agent.agent_aware_router.server.http_server import vLLMHttpServer

    flag = "UNI_AGENT_ENABLE_NATIVE_HASH_METADATA"
    copy_key = "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"
    monkeypatch.setenv(flag, "0")
    monkeypatch.setenv(copy_key, extra_vars)
    monkeypatch.setenv("USER_ENGINE_SETTING", "custom")
    monkeypatch.setenv("VLLM_PLUGINS", "user_plugin")
    initialized = []

    def parent_init(self):
        copied = get_env_vars_to_copy()
        assert {flag, "USER_ENGINE_SETTING"} <= copied
        assert os.environ["USER_ENGINE_SETTING"] == "custom"
        assert [name.strip() for name in os.environ[copy_key].split(",")].count(flag) == 1
        assert os.environ["VLLM_PLUGINS"] == "user_plugin"
        initialized.append(os.environ[flag])

    monkeypatch.setattr(vLLMHttpServer, "__init__", parent_init)
    KVMetadataHttpServer()
    KVMetadataHttpServer()
    assert initialized == ["1", "1"]


@pytest.mark.cpu
@pytest.mark.level0
def test_group_timeout_preserves_profiles_and_resolved_config(server):
    rpc = server.engine.engine_core.call_utility_async

    async def query(method):
        if method == "get_kv_cache_group_metadata":
            await asyncio.Future()
        if method == "uni_agent_get_native_hash_profile":
            return server.native_profile
        return server.reservation_profile

    rpc.side_effect = query
    server._METADATA_RPC_TIMEOUT = 0.01
    result = read(server)
    assert result["status"] == "error"
    assert result["reason"] == "rpc_timeout"
    assert result["native_hash_config"] == server.native_profile
    assert result["cache_reservation_profile"] == server.reservation_profile
    assert result["cache_config"]["enable_prefix_caching"] is True
    assert result["metadata_errors"]["groups"]["reason"] == "rpc_timeout"
