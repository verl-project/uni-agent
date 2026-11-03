"""Native metadata forwarding without starting GPU engines."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

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
    instance.native_profile = {
        "schema_version": 1,
        "vllm_version": "engine-version",
        "algorithm": "sha256",
        "root_hash_hex": "ab" * 32,
        "unit": 16,
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
    assert result["kv_cache_capacity_tokens"] == 20480
    assert result["capacity_source"] == "native_config"
    assert result["groups"] == expected_groups
    server.engine.engine_core.call_utility_async.assert_has_awaits(
        [
            call("get_kv_cache_group_metadata"),
            call("uni_agent_get_native_hash_profile"),
            call("uni_agent_get_cache_reservation_profile"),
        ]
    )


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


def test_missing_capacity_keeps_group_metadata(server):
    del server.engine.vllm_config.cache_config.kv_cache_size_tokens
    result = read(server)
    assert result["status"] == "ok" and len(result["groups"]) == 2
    assert result["kv_cache_capacity_tokens"] is None
    assert result["capacity_source"] is None


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
            raise AttributeError("plugin is absent or filtered")
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
        assert result["cache_reservation_profile"] == server.reservation_profile
    else:
        assert result["native_hash_config"] == server.native_profile


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
    server._NATIVE_HASH_PROFILE_TIMEOUT = 1
    server._NATIVE_HASH_METADATA_BUDGET = 0.01
    result = asyncio.run(discover())
    assert result["status"] == "ok"
    assert result["groups"] == groups
    assert result["native_hash_config"] is None
    assert result["cache_reservation_profile"] is None
    # The first optional timeout exhausts the shared deadline; the second
    # profile must not start a fresh timeout and delay discovery again.
    assert rpc.await_args_list == [call("get_kv_cache_group_metadata"), call("uni_agent_get_native_hash_profile")]


def test_late_group_result_skips_optional_profile_rpc(server):
    rpc = server.engine.engine_core.call_utility_async
    groups = deepcopy(rpc.return_value)

    async def query(method):
        assert method == "get_kv_cache_group_metadata"
        await asyncio.sleep(0.01)
        return groups

    rpc.side_effect = query
    server._NATIVE_HASH_METADATA_BUDGET = 0.001
    result = read(server)
    assert result["status"] == "ok"
    assert result["groups"] == groups
    assert result["native_hash_config"] is None
    assert result["cache_reservation_profile"] is None
    rpc.assert_awaited_once_with("get_kv_cache_group_metadata")


def test_server_enables_plugin_before_parent_initialization_without_changing_allowlist(monkeypatch):
    from uni_agent.agent_aware_router.server.http_server import vLLMHttpServer

    monkeypatch.delenv("UNI_AGENT_ENABLE_NATIVE_HASH_METADATA", raising=False)
    monkeypatch.setenv("VLLM_PLUGINS", "user_plugin")
    initialized = []

    def parent_init(self):
        import os

        initialized.append(os.environ["UNI_AGENT_ENABLE_NATIVE_HASH_METADATA"])
        assert os.environ["VLLM_PLUGINS"] == "user_plugin"

    monkeypatch.setattr(vLLMHttpServer, "__init__", parent_init)
    KVMetadataHttpServer()
    assert initialized == ["1"]
