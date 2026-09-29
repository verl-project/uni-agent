from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.cpu
@pytest.mark.level0
def test_mini_swe_launcher_uses_validated_driver_python_for_ray_job():
    repo_root = Path(__file__).resolve().parents[3]
    launcher = (repo_root / "examples/mini_swe_agent/run_infer_mini_swe_agent.sh").read_text(encoding="utf-8")

    expected = '-- "${PYTHON_BIN}" "${REPO_ROOT}/examples/inference/parallel_infer_verl.py"'
    assert expected in launcher
    assert '\n    -- python "${REPO_ROOT}/examples/inference/parallel_infer_verl.py"' not in launcher


@pytest.mark.cpu
@pytest.mark.level0
def test_mini_swe_launcher_forwards_sglang_cuda_library_path_only_to_actor_runtime():
    repo_root = Path(__file__).resolve().parents[3]
    launcher = (repo_root / "examples/mini_swe_agent/run_infer_mini_swe_agent.sh").read_text(encoding="utf-8")

    assert "SGLANG_ACTOR_LD_LIBRARY_PATH" in launcher
    assert 'env["UNI_AGENT_SGLANG_LD_LIBRARY_PATH"]' in launcher
    assert 'LD_LIBRARY_PATH="${SGLANG_ACTOR_LD_LIBRARY_PATH}"' in launcher
    assert "examples/mini_swe_agent/sglang_actor_runtime" in launcher
    assert 'env["UNI_AGENT_SGLANG_CUDA_HOME"]' in launcher
    assert 'env["UNI_AGENT_SGLANG_CUDA_INCLUDE_PATH"]' in launcher
    assert 'env["UNI_AGENT_SGLANG_CPATH"]' in launcher
    assert 'env["UNI_AGENT_SGLANG_PATH"]' in launcher
    assert "SGLANG_CUDA_HOME=\"${REPO_ROOT}/.tmp/sglang-cuda13\"" in launcher


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
