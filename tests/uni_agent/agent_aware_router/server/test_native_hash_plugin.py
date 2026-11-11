"""The optional engine plugin reads native state without initializing it."""

import os
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest

from uni_agent.vllm_plugins import native_hash

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_import_and_disabled_registration_do_not_import_vllm():
    env = {**os.environ, "UNI_AGENT_ENABLE_NATIVE_HASH_METADATA": "0", "UNI_AGENT_ENABLE_CACHE_LIFECYCLE": "0"}
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from uni_agent.vllm_plugins.native_hash import register; "
            "register(); assert 'vllm' not in sys.modules; "
            "assert 'uni_agent.agent_aware_router' not in sys.modules",
        ],
        env=env,
        check=True,
    )


def test_registration_is_idempotent_and_does_not_replace_other_methods(monkeypatch):
    from vllm.v1.engine.core import EngineCore

    monkeypatch.setenv("UNI_AGENT_ENABLE_NATIVE_HASH_METADATA", "1")
    names = (native_hash.METHOD, native_hash.RESERVATION_METHOD)
    for name in names:
        monkeypatch.setattr(EngineCore, name, None, raising=False)
    native_hash.register()
    installed = {name: getattr(EngineCore, name) for name in names}
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

    monkeypatch.setattr(kv_cache_utils, "NONE_HASH", bytes(range(32)), raising=False)
    monkeypatch.setattr(kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_: (16, 4))
    monkeypatch.setattr(envs, "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", False)
    return SimpleNamespace(
        request_block_hasher=lambda _: [],
        scheduler=SimpleNamespace(kv_cache_config=object(), num_lookahead_tokens=0),
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=True, prefix_caching_hash_algo="sha256"),
            kv_transfer_config=None,
        ),
    )


def test_profile_reads_actual_root_unit_and_emitter_mode_without_seed_reconstruction(core, monkeypatch):
    import vllm
    import vllm.envs as envs
    from vllm.v1.core import kv_cache_utils

    monkeypatch.setenv("PYTHONHASHSEED", "731")
    monkeypatch.setattr(envs, "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", True)
    root = kv_cache_utils.NONE_HASH
    result = native_hash._get_native_hash_profile(core)
    assert result == {
        "schema_version": 1,
        "vllm_version": vllm.__version__,
        "algorithm": "sha256",
        "root_hash_hex": root.hex(),
        "unit": 4,
        "pickle_protocol": pickle.HIGHEST_PROTOCOL,
        "int_hashes": True,
        "lookahead_tokens": 0,
    }
    assert kv_cache_utils.NONE_HASH is root
    assert native_hash._get_native_hash_profile(core) == result


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
def test_unsupported_native_state_has_no_profile(core, monkeypatch, unsupported):
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
    assert native_hash._get_native_hash_profile(core) is None


@pytest.mark.parametrize("mamba_mode", ["align", "all"])
def test_reservation_profile_uses_native_full_window_and_mamba_budgets_without_hash_initialization(
    monkeypatch, mamba_mode
):
    import json

    import torch
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec, FullAttentionSpec, MambaSpec, SlidingWindowSpec

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
    assert result == {
        "schema_version": 1,
        "num_blocks": 2048,
        "max_model_len": 4096,
        "groups": [
            {"group_idx": 0, "max_blocks_per_request": 256},
            {"group_idx": 1, "max_blocks_per_request": 13},
            {"group_idx": 2, "max_blocks_per_request": 4 if mamba_mode == "align" else 66},
            {"group_idx": 3, "max_blocks_per_request": 0},
        ],
    }
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert native_hash._get_cache_reservation_profile(core) == result
    assert kv_cache_utils.NONE_HASH is original
    assert [group.kv_cache_spec for group in cache.kv_cache_groups] == specs


@pytest.mark.parametrize("invalid", ["missing", "pool", "page", "budget", "raises"])
def test_incomplete_static_reservation_profile_is_unavailable(invalid):
    spec = SimpleNamespace(page_size_bytes=8, max_memory_usage_bytes=lambda _: 0)
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
    else:

        def broken(_):
            raise NotImplementedError("unsupported spec")

        spec.max_memory_usage_bytes = broken
    assert native_hash._get_cache_reservation_profile(core) is None
