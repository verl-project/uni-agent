"""GPU smoke coverage for the locked driver and Ray worker environments."""

import importlib.metadata
import os
import shlex
import shutil
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.level0]
ROOT = Path(__file__).resolve().parents[2]
EXTRAS = ("vllm", "megatron", "swebench")


def probe_environment():
    # Import inside the probe so CPU collection does not require GPU packages.
    import megatron.core
    import torch
    import transformer_engine.pytorch
    import vllm

    assert torch.cuda.is_available(), "GPU CI requires a working CUDA device"
    value = torch.ones((32, 32), device="cuda")
    assert (value @ value)[0, 0].item() == 32
    torch.cuda.synchronize()
    return {
        "python": str(Path(sys.executable).resolve()),
        "prefix": str(Path(sys.prefix).resolve()),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "vllm", "megatron-core", "transformer-engine", "ray", "uni-agent", "verl")
        },
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "backend_paths": [megatron.core.__file__, transformer_engine.pytorch.__file__, vllm.__file__],
    }


def test_locked_gpu_environment_reaches_ray_worker():
    import ray

    driver = probe_environment()
    assert Path(driver["prefix"]) == (ROOT / ".venv").resolve(), driver
    uv = shutil.which("uv")
    assert uv, "The GPU image must provide uv"
    command = [uv, "run", "--project", str(ROOT), "--locked", "--all-packages"]
    for extra in EXTRAS:
        command.extend(["--extra", extra])
    try:
        ray.init(
            address="local",
            num_cpus=2,
            num_gpus=1,
            include_dashboard=False,
            runtime_env={
                "py_executable": shlex.join(command),
                # pytest imports this file as test_uv_gpu. Make that module
                # importable when Ray deserializes the probe in a new process.
                "env_vars": {"PYTHONPATH": os.pathsep.join([str(ROOT), str(Path(__file__).parent)])},
            },
        )
        worker = ray.get(ray.remote(num_gpus=1)(probe_environment).remote(), timeout=180)
        assert worker["prefix"] == driver["prefix"], (driver, worker)
        assert worker["python"] == driver["python"], (driver, worker)
        assert worker["versions"] == driver["versions"], (driver, worker)
        assert worker["cuda"] == driver["cuda"], (driver, worker)
        assert worker["gpu_count"] == 1, worker
        print("UV_GPU_DRIVER", driver)
        print("UV_GPU_RAY_WORKER", worker)
    finally:
        ray.shutdown()
