from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_mini_swe_launcher_preserves_defaults_without_forwarding_local_runtime(tmp_path, monkeypatch, engine):
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
    assert "UNI_AGENT_SGLANG_PYTHON" not in runtime
    assert "UNI_AGENT_SGLANG_PYTHONPATH" not in runtime


@pytest.mark.cpu
@pytest.mark.level0
def test_sglang_inference_driver_import_does_not_require_sglang_package():
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
from uni_agent.inference.sglang import SGLangInferenceReplica
assert SGLangInferenceReplica.__name__ == "SGLangInferenceReplica"
assert not any(name == "sglang" or name.startswith("sglang.") for name in sys.modules)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(repo_root), str(repo_root / "verl")))
    subprocess.run([sys.executable, "-c", child], cwd=repo_root, env=env, capture_output=True, text=True, check=True)


@pytest.mark.cpu
@pytest.mark.level0
def test_vllm_standalone_preserves_upstream_checkpoint_worker(monkeypatch):
    from verl.checkpoint_engine.base import CheckpointEngineWorker
    from verl.workers.rollout import replica as replica_module
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

    replica = object.__new__(vLLMReplica)
    replica.config = object()
    replica.model_config = object()
    replica.replica_rank = 0
    replica.rollout_mode = replica_module.RolloutMode.STANDALONE
    monkeypatch.setattr(replica_module.ray, "remote", lambda worker_class: worker_class)

    worker_spec = replica.get_ray_class_with_init_args()

    assert worker_spec.cls is CheckpointEngineWorker
