"""The optional engine plugin reads native state without initializing it."""

import math
import os
import pickle
import subprocess
import sys
from types import SimpleNamespace
from typing import get_args
from unittest.mock import Mock

import pytest

from uni_agent.vllm_plugins import native_hash


@pytest.mark.cpu
@pytest.mark.level0
def test_disabled_entry_point_does_not_import_vllm():
    env = {**os.environ, "UNI_AGENT_ENABLE_NATIVE_HASH_METADATA": "0"}
    script = """
import sys
from importlib.metadata import entry_points
entry, = entry_points(group="vllm.general_plugins", name="uni_agent_native_hash")
entry.load()()
assert "vllm" not in sys.modules
assert "uni_agent.agent_aware_router" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], env=env, check=True)


@pytest.mark.cpu
@pytest.mark.level0
def test_registration_is_idempotent_and_does_not_replace_other_methods(monkeypatch):
    import vllm.plugins as plugins
    from vllm.v1.engine.core import EngineCore

    monkeypatch.setenv("UNI_AGENT_ENABLE_NATIVE_HASH_METADATA", "1")
    monkeypatch.setenv("VLLM_PLUGINS", "uni_agent_native_hash")
    monkeypatch.setattr(plugins, "plugins_loaded", False)
    names = (native_hash.METHOD, native_hash.RESERVATION_METHOD)
    for name in names:
        monkeypatch.setattr(EngineCore, name, None, raising=False)
    plugins.load_general_plugins()
    installed = {name: getattr(EngineCore, name) for name in names}
    assert all(callable(method) for method in installed.values())
    native_hash.register()
    assert all(getattr(EngineCore, name) is method for name, method in installed.items())

    existing = lambda self: None
    for conflict in names:
        for name in names:
            monkeypatch.setattr(EngineCore, name, existing if name == conflict else None)
        with pytest.raises(RuntimeError, match="Refusing to replace"):
            native_hash.register()
        assert getattr(EngineCore, conflict) is existing
        assert getattr(EngineCore, next(name for name in names if name != conflict)) is None


@pytest.fixture
def core(monkeypatch):
    import vllm.envs as envs
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

    from tests.uni_agent.agent_aware_router.server._helpers import hybrid_cache_config

    monkeypatch.setattr(kv_cache_utils, "NONE_HASH", bytes(range(32)), raising=False)
    monkeypatch.setattr(envs, "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", False)
    coordinator = Mock(spec=HybridKVCacheCoordinator, scheduler_block_size=8, _cache_hit_alignment_tokens=8)
    return SimpleNamespace(
        request_block_hasher=lambda _: [],
        scheduler=SimpleNamespace(
            kv_cache_config=hybrid_cache_config(),
            num_lookahead_tokens=0,
            kv_cache_manager=SimpleNamespace(coordinator=coordinator),
        ),
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(
                enable_prefix_caching=True,
                prefix_caching_hash_algo="sha256",
                block_size=8,
                hash_block_size=None,
                prefix_match_unit=None,
            ),
            parallel_config=SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1),
            kv_transfer_config=None,
        ),
    )


@pytest.mark.cpu
@pytest.mark.level0
def test_profile_reads_engine_root_and_resolved_hybrid_hash_unit(core, monkeypatch):
    import vllm
    import vllm.envs as envs
    from vllm.v1.core import kv_cache_utils

    monkeypatch.setattr(envs, "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", True)
    root = kv_cache_utils.NONE_HASH
    result = native_hash._get_native_hash_profile(core)
    assert result == {
        "schema_version": 1,
        "vllm_version": vllm.__version__,
        "enable_prefix_caching": True,
        "algorithm": "sha256",
        "root_hash_hex": root.hex(),
        "unit": 4,
        "hit_alignment": 8,
        "pickle_protocol": pickle.HIGHEST_PROTOCOL,
        "int_hashes": True,
        "lookahead_tokens": 0,
    }
    assert kv_cache_utils.NONE_HASH is root
    assert native_hash._get_native_hash_profile(core) == result


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "version,resolved,expected",
    [
        ("0.23.0", None, 16),
        ("0.25.0", None, 16),
        ("0.25.1", None, 16),
        ("0.26.0", 4, 4),
        ("0.33.0", None, None),
        ("dev", None, None),
    ],
)
def test_hybrid_profile_only_uses_scheduler_alignment_for_known_legacy_versions(
    core, monkeypatch, version, resolved, expected
):
    import vllm
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

    monkeypatch.setattr(vllm, "__version__", version)
    coordinator = Mock(spec=HybridKVCacheCoordinator, scheduler_block_size=16)
    if resolved is None:
        del coordinator._cache_hit_alignment_tokens
    else:
        coordinator._cache_hit_alignment_tokens = resolved
    core.scheduler.kv_cache_manager.coordinator = coordinator
    assert native_hash._get_native_hash_profile(core)["hit_alignment"] == expected


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "unsupported",
    [
        "uninitialized",
        "disabled",
        "algorithm",
        "all_mamba",
        "transfer",
        "lookahead",
        "prefill",
        "hash_lookahead",
        "root",
        "resolver",
    ],
)
def test_unsupported_codec_keeps_unit_except_when_resolver_missing(core, monkeypatch, unsupported):
    from vllm.v1.core import kv_cache_utils

    if unsupported == "uninitialized":
        core.request_block_hasher = None
    elif unsupported == "disabled":
        core.vllm_config.cache_config.enable_prefix_caching = False
    elif unsupported == "algorithm":
        core.vllm_config.cache_config.prefix_caching_hash_algo = "sha256_cbor"
    elif unsupported == "all_mamba":
        core.vllm_config.cache_config.mamba_cache_mode = "all"
    elif unsupported == "transfer":
        core.vllm_config.kv_transfer_config = object()
    elif unsupported == "lookahead":
        core.scheduler.num_lookahead_tokens = 1
    elif unsupported == "prefill":
        core.scheduler.num_prefill_lookahead = 1
    elif unsupported == "hash_lookahead":
        core.vllm_config.num_hash_lookahead_tokens = 1
    elif unsupported == "root":
        monkeypatch.setattr(kv_cache_utils, "NONE_HASH", b"short")
    else:
        monkeypatch.delattr(kv_cache_utils, "resolve_kv_cache_block_sizes")
    profile = native_hash._get_native_hash_profile(core)
    assert profile is not None
    assert profile["enable_prefix_caching"] is (unsupported != "disabled")
    if unsupported == "resolver":
        assert profile["unit"] is None
        assert profile["codec_unavailable_reason"] == "hash_unit_resolver_unavailable"
    else:
        assert profile["unit"] == (8 if unsupported == "disabled" else 4)
        assert profile["hit_alignment"] == 8
        assert profile["codec_unavailable_reason"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("mamba_mode", ["align", "all"])
def test_reservation_profile_uses_native_full_window_and_mamba_budgets_without_hash_initialization(
    monkeypatch, mamba_mode
):
    import json

    import torch
    from vllm.config.cache import MambaCacheMode
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec, FullAttentionSpec, MambaSpec, SlidingWindowSpec

    if mamba_mode == "all" and mamba_mode not in get_args(MambaCacheMode):
        pytest.skip("Installed vLLM does not support Mamba cache mode 'all'")

    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=4096),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1),
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_mode),
        max_in_flight_tokens=256,
    )
    specs = [
        FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=32, dtype=torch.float16),
        SlidingWindowSpec(block_size=32, num_kv_heads=2, head_size=32, dtype=torch.float16, sliding_window=128),
        MambaSpec(block_size=64, shapes=((2, 16),), dtypes=(torch.float16,), num_speculative_blocks=2),
        EncoderOnlyAttentionSpec(block_size=16, num_kv_heads=2, head_size=32, dtype=torch.float16),
    ]
    cache = SimpleNamespace(num_blocks=2048, kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec) for spec in specs])
    # The independent budget utility works when native hash recovery is disabled.
    core = SimpleNamespace(vllm_config=config, scheduler=SimpleNamespace(kv_cache_config=cache))
    monkeypatch.setattr(kv_cache_utils, "NONE_HASH", b"not-initialized", raising=False)
    original = kv_cache_utils.NONE_HASH
    result = native_hash._get_cache_reservation_profile(core)
    budgets = [math.ceil(spec.max_memory_usage_bytes(config) / spec.page_size_bytes) for spec in specs]
    assert result == {
        "schema_version": 1,
        "num_blocks": 2048,
        "capacity_blocks": 2047,
        "max_model_len": 4096,
        "groups": [{"group_idx": i, "max_blocks_per_request": budget} for i, budget in enumerate(budgets)],
    }
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert native_hash._get_cache_reservation_profile(core) == result
    assert kv_cache_utils.NONE_HASH is original
    assert [group.kv_cache_spec for group in cache.kv_cache_groups] == specs


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "raw,expected", [(0, 0), (1, 0), (10, 9), (-1, None), (True, None), (10.0, None), (None, None)]
)
def test_pool_capacity_normalization_rejects_invalid_configured_counts(raw, expected):
    assert native_hash.cache_capacity_blocks(raw) == expected


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("invalid", ["missing", "pool", "page", "budget", "host_group", "raises"])
def test_incomplete_static_reservation_profile_is_unavailable(invalid):
    spec = SimpleNamespace(page_size_bytes=8, max_memory_usage_bytes=lambda _: 8)
    cache = SimpleNamespace(num_blocks=32, kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)])
    core = SimpleNamespace(
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=128)),
        scheduler=SimpleNamespace(kv_cache_config=cache),
    )
    if invalid == "missing":
        del cache.kv_cache_groups
    elif invalid == "pool":
        cache.num_blocks = True
    elif invalid == "page":
        spec.page_size_bytes = 0
    elif invalid == "budget":
        spec.max_memory_usage_bytes = lambda _: -1
    elif invalid == "host_group":
        cache.kv_cache_groups.append(SimpleNamespace(kv_cache_spec=spec, host_resident=True))
    else:

        def broken(_):
            raise NotImplementedError("unsupported spec")

        spec.max_memory_usage_bytes = broken
    assert native_hash._get_cache_reservation_profile(core) is None


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("unit", [None, 0, True])
def test_invalid_native_unit_has_an_explicit_reason(core, monkeypatch, unit):
    from vllm.v1.core import kv_cache_utils

    monkeypatch.setattr(kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_: (16, unit))
    profile = native_hash._get_native_hash_profile(core)
    assert profile is not None
    assert profile["unit"] is None
    assert profile["codec_unavailable_reason"] == "invalid_native_hash_unit"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("failure", ["resolver", "alignment"])
def test_profile_read_error_preserves_resolved_unit(core, monkeypatch, failure):
    from vllm.v1.core import kv_cache_utils

    if failure == "resolver":
        monkeypatch.setattr(kv_cache_utils, "resolve_kv_cache_block_sizes", Mock(side_effect=ValueError("unit")))
    else:
        monkeypatch.setattr(native_hash, "_get_hit_alignment", Mock(side_effect=ValueError("alignment")))
    profile = native_hash._get_native_hash_profile(core)
    assert profile["codec_unavailable_reason"] == "native_profile_error:ValueError"
    assert profile["enable_prefix_caching"] is True
    assert profile["unit"] == (None if failure == "resolver" else 4)
    assert profile.get("hit_alignment") is None
