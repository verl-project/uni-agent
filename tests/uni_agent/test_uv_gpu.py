"""GPU smoke coverage for the locked driver and Ray worker environments."""

import hashlib
import importlib.metadata
import os
import shlex
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.level0]
ROOT = Path(__file__).resolve().parents[2]
EXTRAS = ("vllm", "megatron", "swebench")


def assert_equivalent_environments(driver, worker):
    # Ray's UV hook ships working_dir and creates a .venv in the unpacked
    # project. Paths can differ; the lock, policy and installed stack cannot.
    for environment in (driver, worker):
        assert Path(environment["prefix"]) == Path(environment["project"]) / ".venv", "prefix"
    for field in ("lock_sha256", "policy_sha256", "python_version", "versions", "cuda"):
        assert worker[field] == driver[field], f"{field}: driver={driver[field]!r}, worker={worker[field]!r}"


def probe_environment():
    # Import inside the probe so CPU collection does not require GPU packages.
    import megatron.core
    import torch
    import transformer_engine.pytorch
    import vllm

    import uni_agent
    import verl

    assert torch.cuda.is_available(), "GPU CI requires a working CUDA device"
    value = torch.ones((32, 32), device="cuda")
    assert (value @ value)[0, 0].item() == 32
    torch.cuda.synchronize()
    project = Path(uni_agent.__file__).resolve().parents[1]
    prefix = Path(sys.prefix).resolve()
    assert Path(verl.__file__).resolve().is_relative_to(project / "verl")
    backend_paths = [megatron.core.__file__, transformer_engine.pytorch.__file__, vllm.__file__]
    assert all(Path(path).resolve().is_relative_to(prefix) for path in backend_paths), backend_paths
    return {
        "python": str(Path(sys.executable).resolve()),
        "python_version": tuple(sys.version_info[:3]),
        "project": str(project),
        "prefix": str(prefix),
        "lock_sha256": hashlib.sha256((project / "uv.lock").read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256(
            (project / "pyproject.toml").read_bytes() + (project / "verl/pyproject.toml").read_bytes()
        ).hexdigest(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "vllm", "megatron-core", "transformer-engine", "ray", "uni-agent", "verl")
        },
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "backend_paths": backend_paths,
    }


def test_locked_gpu_environment_reaches_ray_worker():
    import ray

    driver = probe_environment()
    assert Path(driver["prefix"]) == (ROOT / ".venv").resolve(), driver

    @ray.remote(num_gpus=1)
    def worker_probe():
        result = probe_environment()
        command = shlex.split(ray.get_runtime_context().runtime_env["py_executable"])
        assert Path(command[0]).name == "uv" and command[1] == "run", command
        assert "--locked" in command and "--all-packages" in command, command
        extras = [command[i + 1] for i, arg in enumerate(command[:-1]) if arg == "--extra"]
        assert set(EXTRAS).issubset(extras), command
        result["uv_command"] = command
        return result

    try:
        ray.init(
            address="local",
            num_cpus=2,
            num_gpus=1,
            include_dashboard=False,
            runtime_env={
                # Let Ray's native UV hook propagate the driver's locked
                # command and build an environment in its packaged working_dir.
                "working_dir": str(ROOT),
                # pytest imports this file as test_uv_gpu. Make that module
                # importable when Ray deserializes the probe in a new process.
                "env_vars": {"PYTHONPATH": os.pathsep.join([".", "tests/uni_agent"])},
            },
        )
        worker = ray.get(worker_probe.remote(), timeout=180)
        print("UV_GPU_DRIVER", driver)
        print("UV_GPU_RAY_WORKER", worker)
        assert_equivalent_environments(driver, worker)
        assert worker["gpu_count"] == 1, worker
    finally:
        ray.shutdown()
