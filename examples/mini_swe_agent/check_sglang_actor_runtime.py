#!/usr/bin/env python3
"""Verify the isolated SGLang Python runtime from inside an active Ray job."""

from __future__ import annotations

import json
import os
import sys

import ray


def _runtime_identity(stage: str) -> dict[str, str]:
    print("sglang actor probe: entered worker function", flush=True)
    import ray

    identity = {
        "python_executable": sys.executable,
        "ray_version": ray.__version__,
        "ray_module_file": ray.__file__,
    }
    print("sglang actor probe: imported ray", flush=True)
    if stage == "ray":
        return identity

    import sglang

    identity.update({"sglang_version": sglang.__version__, "sglang_module_file": sglang.__file__})
    print("sglang actor probe: imported sglang", flush=True)
    if stage == "sglang":
        return identity

    import torch

    identity.update({"torch_version": torch.__version__, "torch_module_file": torch.__file__})
    print("sglang actor probe: imported torch", flush=True)
    if stage == "torch":
        return identity

    import transformers

    identity.update({"transformers_version": transformers.__version__, "transformers_module_file": transformers.__file__})
    print("sglang actor probe: imported transformers", flush=True)
    return identity


class RuntimeIdentityProbe:
    def __init__(self, stage: str) -> None:
        self.identity = _runtime_identity(stage)

    def get_identity(self) -> dict[str, str]:
        return self.identity


def main() -> int:
    actor_python = os.environ.get("UNI_AGENT_SGLANG_PYTHON")
    actor_pythonpath = os.environ.get("UNI_AGENT_SGLANG_PYTHONPATH")
    if not actor_python or not actor_pythonpath:
        raise RuntimeError("the Ray job is missing the isolated SGLang actor environment")
    stage = os.environ.get("SGLANG_ACTOR_PROBE_STAGE", "all")
    if stage not in {"ray", "sglang", "torch", "all"}:
        raise ValueError(f"unknown actor probe stage: {stage}")

    ray.init()
    actor_env = {
        "py_executable": actor_python,
        "env_vars": {
            "PYTHONPATH": actor_pythonpath,
            "FLASHINFER_WORKSPACE_BASE": os.environ["FLASHINFER_WORKSPACE_BASE"],
            "FLASHINFER_CUBIN_DIR": os.environ["FLASHINFER_CUBIN_DIR"],
            "HF_HOME": os.environ["HF_HOME"],
            "TMPDIR": os.environ["TMPDIR"],
            "XDG_CACHE_HOME": os.environ["XDG_CACHE_HOME"],
            "SGLANG_ACTOR_PROBE_STAGE": stage,
        },
    }
    probe = ray.remote(RuntimeIdentityProbe).options(runtime_env=actor_env).remote(stage)
    identity = ray.get(probe.get_identity.remote())
    expected = {
        "python_executable": actor_python,
        "ray_version": ray.__version__,
    }
    if stage != "ray":
        expected["sglang_version"] = "0.5.20"
    if stage in {"torch", "all"}:
        expected["torch_version"] = "2.13.0+cu130"
    if stage == "all":
        expected["transformers_version"] = "5.12.1"
    observed = {key: identity.get(key) for key in expected}
    report = {"status": "PASS" if observed == expected else "FAIL", "stage": stage, "expected": expected, "observed": observed, "identity": identity}
    print(json.dumps(report, sort_keys=True))
    ray.kill(probe, no_restart=True)
    ray.shutdown()
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
