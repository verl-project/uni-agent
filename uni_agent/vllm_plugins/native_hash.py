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

"""Opt-in, read-only metadata from the process that owns native cache hashes."""

import os
import pickle

METHOD = "uni_agent_get_native_hash_profile"
RESERVATION_METHOD = "uni_agent_get_cache_reservation_profile"
NATIVE_HASH_METADATA_ENV = "UNI_AGENT_ENABLE_NATIVE_HASH_METADATA"
_OWNER = "uni_agent.native_hash.v1"


def cache_capacity_blocks(num_blocks):
    """Normalize vLLM's configured pool size, excluding its one null block."""
    if type(num_blocks) is not int or num_blocks < 0:
        return None
    return max(0, num_blocks - 1)


def _get_hit_alignment(scheduler, version):
    """Read the alignment used by the running coordinator's prefix lookup."""
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator, UnitaryKVCacheCoordinator

    manager = getattr(scheduler, "kv_cache_manager", None)
    coordinator = getattr(manager, "coordinator", None)
    alignment = None
    if isinstance(coordinator, UnitaryKVCacheCoordinator):
        alignment = coordinator.block_size
    elif isinstance(coordinator, HybridKVCacheCoordinator):
        alignment = getattr(coordinator, "_cache_hit_alignment_tokens", None)
        if alignment is None and version in {"0.23.0", "0.24.0", "0.25.0"}:
            alignment = coordinator.scheduler_block_size
    return alignment if type(alignment) is int and alignment > 0 else None


def _get_native_hash_profile(self):
    import vllm
    import vllm.envs as envs
    from vllm.v1.core import kv_cache_utils

    config = self.vllm_config
    cache = config.cache_config
    scheduler = self.scheduler
    lookahead = getattr(scheduler, "num_lookahead_tokens", None)
    prefill_lookahead = getattr(scheduler, "num_prefill_lookahead", 0)
    hash_lookahead = getattr(config, "num_hash_lookahead_tokens", 0)
    root = getattr(kv_cache_utils, "NONE_HASH", None)
    resolve = getattr(kv_cache_utils, "resolve_kv_cache_block_sizes", None)
    int_hashes = getattr(envs, "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", None)
    profile = {"schema_version": 1, "vllm_version": vllm.__version__, "unit": None}
    if not callable(resolve):
        return {**profile, "codec_unavailable_reason": "hash_unit_resolver_unavailable"}
    # Use the same resolver that initializes EngineCore's request hasher.
    _, unit = resolve(scheduler.kv_cache_config, config)
    if type(unit) is not int or unit <= 0:
        return {**profile, "codec_unavailable_reason": "invalid_native_hash_unit"}
    profile["unit"] = unit
    profile["hit_alignment"] = _get_hit_alignment(scheduler, vllm.__version__)
    unsupported = (
        (getattr(self, "request_block_hasher", None) is None, "hasher_uninitialized"),
        (not getattr(cache, "enable_prefix_caching", False), "prefix_caching_disabled"),
        (getattr(cache, "prefix_caching_hash_algo", None) != "sha256", "unsupported_hash_algorithm"),
        (getattr(cache, "mamba_cache_mode", None) == "all", "unsupported_mamba_all_codec"),
        (getattr(config, "kv_transfer_config", None) is not None, "unsupported_kv_transfer"),
        (type(lookahead) is not int or lookahead != 0, "unsupported_lookahead"),
        (type(prefill_lookahead) is not int or prefill_lookahead != 0, "unsupported_prefill_lookahead"),
        (type(hash_lookahead) is not int or hash_lookahead != 0, "unsupported_hash_lookahead"),
        (type(int_hashes) is not bool, "invalid_int_hashes"),
        (not isinstance(root, bytes) or len(root) != 32, "invalid_root_hash"),
    )
    for rejected, reason in unsupported:
        if rejected:
            return {**profile, "codec_unavailable_reason": reason}
    return {
        **profile,
        "algorithm": "sha256",
        "root_hash_hex": root.hex(),
        "pickle_protocol": pickle.HIGHEST_PROTOCOL,
        "int_hashes": int_hashes,
        "lookahead_tokens": 0,
    }


_get_native_hash_profile.__uni_agent_owner__ = _OWNER


def _get_cache_reservation_profile(self):
    """Read native static request budgets, independently of hash recovery.

    Each group draws block IDs from the scheduler's shared pool. These are
    the native specs' peak budgets, not samples of current allocations.
    """
    try:
        config = self.vllm_config
        cache = self.scheduler.kv_cache_config
        num_blocks = cache.num_blocks
        max_model_len = config.model_config.max_model_len
        if (
            type(num_blocks) is not int
            or num_blocks <= 0
            or type(max_model_len) is not int
            or max_model_len <= 0
            or not cache.kv_cache_groups
        ):
            return None
        groups = []
        for idx, group in enumerate(cache.kv_cache_groups):
            spec = group.kv_cache_spec
            page_bytes = spec.page_size_bytes
            peak_bytes = spec.max_memory_usage_bytes(config)
            if type(page_bytes) is not int or page_bytes <= 0 or type(peak_bytes) is not int or peak_bytes < 0:
                return None
            groups.append({"group_idx": idx, "max_blocks_per_request": (peak_bytes + page_bytes - 1) // page_bytes})
        return {
            "schema_version": 1,
            "num_blocks": num_blocks,
            "capacity_blocks": cache_capacity_blocks(num_blocks),
            "max_model_len": max_model_len,
            "groups": groups,
        }
    except Exception:
        # Optional metadata must not disrupt engines with unsupported specs.
        return None


_get_cache_reservation_profile.__uni_agent_owner__ = _OWNER


def register():
    """Install read-only utilities when a metadata-enabled server opts in."""
    if os.environ.get(NATIVE_HASH_METADATA_ENV) != "1":
        return

    from vllm.v1.engine.core import EngineCore

    methods = {METHOD: _get_native_hash_profile, RESERVATION_METHOD: _get_cache_reservation_profile}
    # Check the complete set first: a conflict must not leave a partial install.
    for name in methods:
        existing = getattr(EngineCore, name, None)
        if existing is not None and getattr(existing, "__uni_agent_owner__", None) != _OWNER:
            raise RuntimeError(f"Refusing to replace existing EngineCore.{name}")
    for name, method in methods.items():
        if getattr(EngineCore, name, None) is None:
            setattr(EngineCore, name, method)
