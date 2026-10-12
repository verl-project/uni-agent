from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_mini_swe_launcher_preserves_defaults_and_forwards_selected_runtime(tmp_path, monkeypatch, engine):
    if sys.platform == "win32":
        pytest.skip("the Ray inference launcher targets POSIX hosts")
    repo_root = Path(__file__).resolve().parents[3]
    ray_stub = tmp_path / "ray"
    ray_stub.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n", encoding="utf-8")
    ray_stub.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "TENSOR_PARALLEL_SIZE",
        "N_GPUS_PER_NODE",
        "LANGUAGE_MODEL_ONLY",
        "ENFORCE_EAGER",
        "GATEWAY_COUNT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENYUANRONG_SERVER_ADDRESS", "https://sandbox.example")
    monkeypatch.setenv("OPENYUANRONG_TOKEN", "synthetic-test-token")
    monkeypatch.setenv("PYTHON_BIN", sys.executable)
    monkeypatch.setenv("ENGINE", engine)
    monkeypatch.setenv("UNI_AGENT_SGLANG_PYTHON", "/selected/runtime/python")
    monkeypatch.setenv("UNI_AGENT_SGLANG_PYTHONPATH", "/selected/runtime/packages")
    if engine == "sglang":
        monkeypatch.setenv("LANGUAGE_MODEL_ONLY", "true")
    result = subprocess.run(
        ["bash", "examples/mini_swe_agent/run_infer_mini_swe_agent.sh"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    argv = json.loads(result.stdout)
    runtime = json.loads(argv[argv.index("--runtime-env-json") + 1])["env_vars"]
    assert argv[argv.index("--engine") + 1] == engine
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert argv[argv.index("--n-gpus-per-node") + 1] == "2"
    assert argv[argv.index("--gateway-count") + 1] == "4"
    assert runtime["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert ("--language-model-only" in argv) is (engine == "sglang")
    assert "--enforce-eager" not in argv
    assert ("UNI_AGENT_SGLANG_PYTHON" in runtime) is (engine == "sglang")
    if engine == "sglang":
        assert runtime["UNI_AGENT_SGLANG_PYTHONPATH"] == "/selected/runtime/packages"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("eos", [248046, [248046, 248047], (248046, 248047)])
def test_sglang_tokenizer_stops_preserve_explicit_controls_without_adding_string_stop(eos):
    from verl.workers.rollout.sglang_rollout.async_sglang_server import _add_tokenizer_stop_token_ids

    params = {"stop_token_ids": [17], "stop": ["user-supplied-stop"]}
    tokenizer = SimpleNamespace(eos_token_id=eos, additional_stop_token_ids=[18])
    _add_tokenizer_stop_token_ids(params, tokenizer)
    expected = {17, 18, 248046} | ({248047} if not isinstance(eos, int) else set())
    assert set(params["stop_token_ids"]) == expected
    assert params["stop"] == ["user-supplied-stop"]


@pytest.mark.cpu
@pytest.mark.level0
def test_sglang_tokenizer_stops_respect_ignore_eos():
    from verl.workers.rollout.sglang_rollout.async_sglang_server import _add_tokenizer_stop_token_ids

    params = {"ignore_eos": True, "stop_token_ids": [17]}
    _add_tokenizer_stop_token_ids(params, SimpleNamespace(eos_token_id=248046))
    assert params == {"ignore_eos": True, "stop_token_ids": [17]}


@pytest.mark.cpu
@pytest.mark.level0
def test_sglang_eos_reuses_configured_tokenizer():
    from verl.workers.rollout.sglang_rollout.async_sglang_server import _get_request_stop_tokenizer

    tokenizer = object()
    assert _get_request_stop_tokenizer(SimpleNamespace(tokenizer=tokenizer)) is tokenizer


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "local_tokenizer_path,tokenizer_path,expected_path",
    [
        ("/materialized/tokenizer", "/configured/tokenizer", "/materialized/tokenizer"),
        (None, "/configured/tokenizer", "/configured/tokenizer"),
        (None, None, "/weights"),
    ],
)
def test_sglang_eos_fallback_preserves_separate_tokenizer_path(
    monkeypatch, local_tokenizer_path, tokenizer_path, expected_path
):
    from verl.workers.rollout.sglang_rollout.async_sglang_server import _get_request_stop_tokenizer

    calls = []
    tokenizer = object()
    module = ModuleType("sglang.srt.utils.hf_transformers")

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return tokenizer

    module.get_tokenizer = load
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = SimpleNamespace(
        tokenizer=None,
        local_tokenizer_path=local_tokenizer_path,
        tokenizer_path=tokenizer_path,
        local_path="/weights",
        trust_remote_code=True,
    )
    assert _get_request_stop_tokenizer(config) is tokenizer
    assert calls == [(expected_path, {"trust_remote_code": True})]


@pytest.mark.cpu
@pytest.mark.level0
def test_sglang_replica_module_import_does_not_require_sglang_package():
    repo_root = Path(__file__).resolve().parents[3]
    child = """
import importlib.abc
import sys

class BlockSGLang(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sglang" or fullname.startswith("sglang."):
            raise ModuleNotFoundError("SGLang import blocked for lazy-import contract test")
        return None

sys.meta_path.insert(0, BlockSGLang())
from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangReplica
assert SGLangReplica.__name__ == "SGLangReplica"
assert not any(name == "sglang" or name.startswith("sglang.") for name in sys.modules)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(repo_root), str(repo_root / "verl")))
    subprocess.run([sys.executable, "-c", child], cwd=repo_root, env=env, capture_output=True, text=True, check=True)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("checkpoint_workers_enabled", "expected_worker_class"),
    [(False, "_StandalonePlacementWorker"), (True, "CheckpointEngineWorker")],
)
def test_rollout_worker_selection_preserves_training_default_and_supports_inference_opt_out(
    checkpoint_workers_enabled, expected_worker_class, monkeypatch
):
    from verl.workers.rollout import replica as replica_module

    class ConcreteReplica(replica_module.RolloutReplica):
        async def launch_servers(self):
            return None

    replica = object.__new__(ConcreteReplica)
    replica.config = SimpleNamespace(
        checkpoint_engine=SimpleNamespace(enabled=checkpoint_workers_enabled),
    )
    replica.model_config = object()
    replica.replica_rank = 0
    replica.rollout_mode = replica_module.RolloutMode.STANDALONE
    monkeypatch.setattr(replica_module.ray, "remote", lambda worker_class: worker_class)

    worker_spec = replica.get_ray_class_with_init_args()

    assert worker_spec.cls.__name__ == expected_worker_class


@pytest.mark.cpu
@pytest.mark.level0
def test_sglang_server_args_field_detection_supports_dataclass_and_msgspec_records():
    from verl.workers.rollout.sglang_rollout.async_sglang_server import _server_args_has_field

    @dataclass
    class DataclassArgs:
        enable_weights_cpu_backup: bool = False

    assert _server_args_has_field(DataclassArgs, "enable_weights_cpu_backup")

    msgspec = pytest.importorskip("msgspec")

    class MsgspecArgs(msgspec.Struct):
        enable_weights_cpu_backup: bool = False

    assert _server_args_has_field(MsgspecArgs, "enable_weights_cpu_backup")
