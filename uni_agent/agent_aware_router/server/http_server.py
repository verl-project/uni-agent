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

"""Cache metadata and synchronized lifecycle on top of the existing verl server."""

import asyncio
import json
import os
import pickle
from math import gcd

import ray
import vllm

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

from .lifecycle_server import CacheLifecycleServerMixin


class KVMetadataHttpServer(CacheLifecycleServerMixin, vLLMHttpServer):
    """Expose native groups and resolved engine configuration to the router."""

    _METADATA_RPC_TIMEOUT = 30.0

    @staticmethod
    def _native_hash_config(config, groups):
        cache = config.cache_config
        seed = os.environ.get("PYTHONHASHSEED")
        int_hashes = getattr(vllm.envs, "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", None)
        if (
            getattr(cache, "prefix_caching_hash_algo", None) != "sha256"
            or getattr(cache, "mamba_cache_mode", None) == "all"
            or getattr(config, "kv_transfer_config", None) is not None
            or type(int_hashes) is not bool
            or seed is None
            or not seed.isdecimal()
            or not 0 <= int(seed) < 2**32
        ):
            return None
        sizes = [
            group["block_size"]
            for group in groups
            if group.get("prefix_cacheable") is not False
            and group["kind"] not in {"encoder_only_attention", "cross_attention"}
            and (group["kind"] != "unknown" or group.get("prefix_cacheable") is True)
        ]
        if not sizes:
            return None
        return {
            "algorithm": "sha256",
            "seed": seed,
            "unit": getattr(cache, "prefix_match_unit", None) or getattr(cache, "hash_block_size", None) or gcd(*sizes),
            "pickle_protocol": pickle.HIGHEST_PROTOCOL,
            "int_hashes": int_hashes,
        }

    async def get_kv_cache_metadata(self) -> dict:
        result = {"schema_version": 1, "vllm_version": vllm.__version__}
        engine = getattr(self, "engine", None)
        if engine is None:
            status = "not_ready" if self.node_rank == 0 else "headless"
            return {**result, "status": status}
        config = getattr(engine, "vllm_config", None)
        if config is None:
            return {**result, "status": "unsupported", "reason": "engine_config_unavailable"}
        try:
            groups = await asyncio.wait_for(
                engine.engine_core.call_utility_async("get_kv_cache_group_metadata"),
                timeout=self._METADATA_RPC_TIMEOUT,
            )
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
                "status": "ok",
                "groups": groups,
                "native_hash_config": self._native_hash_config(config, groups),
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
            # Send only detached, serializable fields, never an engine/config object.
            return json.loads(json.dumps(metadata, allow_nan=False))
        except asyncio.TimeoutError:
            return {**result, "status": "error", "reason": "rpc_timeout"}
        except Exception as exc:
            # Unsupported RPCs also arrive as remote engine exceptions. Preserve
            # their type and message instead of guessing from a version string.
            return {
                **result,
                "status": "error",
                "reason": "metadata_query_failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }


class KVMetadataReplica(vLLMReplica):
    """Use metadata-enabled actors with the original names and lifecycle."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(KVMetadataHttpServer)
