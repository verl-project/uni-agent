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

"""Native cache metadata on top of the existing verl server."""

import asyncio
import json
import os

import ray
import vllm

from uni_agent.vllm_plugins.native_hash import cache_capacity_blocks
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica


class KVMetadataHttpServer(vLLMHttpServer):
    """Expose native groups and resolved engine configuration to the router.

    Install uni-agent in every engine environment (for example, ``pip install
    -e .``) so vLLM discovers its ``uni_agent_native_hash`` general-plugin entry
    point. An explicit ``VLLM_PLUGINS`` allowlist must include that name to
    enable native hash recovery; the server preserves the user's allowlist.
    """

    # All three RPCs share this deadline, below the router's five-second wait.
    _METADATA_RPC_TIMEOUT = 4.0
    _NATIVE_HASH_PROFILE_TIMEOUT = 0.5

    def __init__(self, *args, **kwargs):
        os.environ["UNI_AGENT_ENABLE_NATIVE_HASH_METADATA"] = "1"
        super().__init__(*args, **kwargs)

    async def _engine_metadata_rpc(self, engine, method, timeout):
        try:
            value = await asyncio.wait_for(engine.engine_core.call_utility_async(method), timeout=timeout)
            if value is None:
                return None, {"reason": "profile_unavailable"}
            return json.loads(json.dumps(value, allow_nan=False)), None
        except asyncio.TimeoutError:
            return None, {"reason": "rpc_timeout"}
        except Exception as exc:
            return None, {"reason": "metadata_query_failed", "error_type": type(exc).__name__, "message": str(exc)}

    async def get_kv_cache_metadata(self) -> dict:
        result = {"schema_version": 1, "vllm_version": vllm.__version__}
        engine = getattr(self, "engine", None)
        if engine is None:
            status = "not_ready" if self.node_rank == 0 else "headless"
            return {**result, "status": status}
        config = getattr(engine, "vllm_config", None)
        if config is None:
            return {**result, "status": "unsupported", "reason": "engine_config_unavailable"}
        budget = self._METADATA_RPC_TIMEOUT
        groups_result, native_result, reservation_result = await asyncio.gather(
            self._engine_metadata_rpc(engine, "get_kv_cache_group_metadata", budget),
            self._engine_metadata_rpc(
                engine, "uni_agent_get_native_hash_profile", min(self._NATIVE_HASH_PROFILE_TIMEOUT, budget)
            ),
            self._engine_metadata_rpc(
                engine, "uni_agent_get_cache_reservation_profile", min(self._NATIVE_HASH_PROFILE_TIMEOUT, budget)
            ),
        )
        groups, group_error = groups_result
        native, native_error = native_result
        reservation, reservation_error = reservation_result
        if native is not None and not isinstance(native, dict):
            native, native_error = None, {"reason": "invalid_profile"}
        if reservation is not None and not isinstance(reservation, dict):
            reservation, reservation_error = None, {"reason": "invalid_profile"}
        if reservation is not None and "capacity_blocks" not in reservation:
            reservation["capacity_blocks"] = cache_capacity_blocks(reservation.get("num_blocks"))
        cache = config.cache_config
        capacity = getattr(cache, "kv_cache_size_tokens", None)
        if type(capacity) is not int or capacity <= 0:
            capacity = None
        parallel = config.parallel_config
        model = config.model_config
        if hasattr(model, "attention_chunk_size"):
            chunk_size = model.attention_chunk_size
            chunk_source = "model_config.attention_chunk_size"
        else:
            chunk_size = getattr(getattr(model, "hf_text_config", None), "attention_chunk_size", None)
            chunk_source = "model_config.hf_text_config.attention_chunk_size"
        metadata = {
            **result,
            "status": "error" if group_error else "ok",
            "native_hash_config": native,
            "cache_reservation_profile": reservation,
            "metadata_errors": {
                key: error
                for key, error in (
                    ("groups", group_error),
                    ("native_hash_config", native_error),
                    ("cache_reservation_profile", reservation_error),
                )
                if error is not None
            },
            "kv_cache_capacity_tokens": capacity,
            "capacity_source": "native_config" if capacity is not None else None,
            "cache_config": {
                key: getattr(cache, key, None)
                for key in ("enable_prefix_caching", "mamba_cache_mode", "hash_block_size", "prefix_match_unit")
            },
            "parallel_config": {
                key: getattr(parallel, key, None)
                for key in (
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "data_parallel_size",
                    "decode_context_parallel_size",
                    "prefill_context_parallel_size",
                )
            },
            "model_config": {
                "model": getattr(model, "model", None),
                "attention_chunk_size": chunk_size,
                "attention_chunk_size_source": chunk_source if chunk_size is not None else None,
            },
        }
        if group_error:
            metadata.update(group_error)
        else:
            metadata["groups"] = groups
        return json.loads(json.dumps(metadata, allow_nan=False))


class KVMetadataReplica(vLLMReplica):
    """Use metadata-enabled actors with the original names and lifecycle."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(KVMetadataHttpServer)
