#!/usr/bin/env python3
"""Freeze a credential-free pre-run manifest for the single-instance rollout."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _git_diff_sha(repo: Path) -> str:
    result = subprocess.run(["git", "-C", str(repo), "diff", "--binary"], check=True, capture_output=True)
    return hashlib.sha256(result.stdout).hexdigest()


def _changed_file_hashes(repo: Path) -> dict[str, str]:
    tracked = _git(repo, "diff", "--name-only").splitlines()
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard").splitlines()
    hashes = {}
    for relative in sorted(set(tracked + untracked)):
        path = repo / relative
        if path.is_file():
            hashes[relative] = _digest(path) or ""
    return hashes


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"required environment value is missing: {name}")
    return value


def _write_manifest() -> Path:
    root = Path(_required_env("REPO_ROOT")).resolve()
    output_dir = Path(_required_env("OUTPUT_DIR")).resolve()
    baseline_path = root / ".research" / "baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    data_path = Path(os.path.expanduser(_required_env("DATA_PATH"))).resolve()
    model_path = Path(os.path.expanduser(_required_env("MODEL_PATH"))).resolve()
    credential_path = Path(_required_env("OPENYUANRONG_CREDENTIAL_FILE"))

    if output_dir.parent != (root / "artifacts").resolve():
        raise ValueError("OUTPUT_DIR must be an immediate child of the repository artifacts directory")
    if not output_dir.is_dir():
        raise FileNotFoundError("the unique run output directory must be created before manifest generation")
    if not data_path.is_file() or not model_path.is_dir():
        raise FileNotFoundError("the selected data or local model path is unavailable")

    selected = {
        "nnodes": int(_required_env("NNODES")),
        "n_gpus_per_node": int(_required_env("N_GPUS_PER_NODE")),
        "tensor_parallel_size": int(_required_env("TENSOR_PARALLEL_SIZE")),
        "n": int(_required_env("N")),
        "limit": int(_required_env("LIMIT")),
        "concurrency": int(_required_env("CONCURRENCY")),
        "gateway_count": int(_required_env("GATEWAY_COUNT")),
        "cuda_visible_devices": _required_env("CUDA_VISIBLE_DEVICES"),
        "language_model_only": _required_env("LANGUAGE_MODEL_ONLY").lower() in {"1", "true"},
    }
    expected_gpu_shape = {
        "nnodes": 1,
        "n_gpus_per_node": 8,
        "tensor_parallel_size": 8,
        "n": 1,
        "limit": 1,
        "concurrency": 1,
        "gateway_count": 1,
        "cuda_visible_devices": "0,1,2,3,4,5,6,7",
        "language_model_only": True,
    }
    if selected != expected_gpu_shape:
        raise ValueError(f"run parameters differ from the frozen single-node TP=8 acceptance shape: {selected}")
    if os.environ.get("ENGINE") not in {"vllm", "sglang"}:
        raise ValueError("ENGINE must explicitly select vllm or sglang")

    credential_stat = credential_path.lstat()
    if (
        not stat.S_ISREG(credential_stat.st_mode)
        or credential_stat.st_uid != os.geteuid()
        or stat.S_IMODE(credential_stat.st_mode) != 0o600
    ):
        raise PermissionError("credential file must be a regular, process-owned mode-600 file")

    data_sha = _digest(data_path)
    expected_data_sha = baseline["inputs"]["dataset_sha256"]
    if data_sha != expected_data_sha:
        raise ValueError("SWE-bench parquet SHA256 differs from the frozen W2 input")

    gpu_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    gpu_rows = list(csv.reader(gpu_result.stdout.splitlines()))
    process_result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    gpu_process_rows = list(csv.reader(process_result.stdout.splitlines()))
    if (
        len(gpu_rows) != 8
        or any(len(row) < 4 or not row[3].strip().isdigit() or int(row[3].strip()) > 16 for row in gpu_rows)
        or gpu_process_rows
    ):
        raise RuntimeError("all eight selected GPUs must be idle (at most 16 MiB driver overhead) with no compute processes before the run")

    task_config_path = Path(os.path.expanduser(_required_env("TASK_CONFIG")))
    if not task_config_path.is_absolute():
        task_config_path = root / task_config_path
    task_config_path = task_config_path.resolve()
    task_entries = yaml.safe_load(task_config_path.read_text(encoding="utf-8"))
    task = next(entry for entry in task_entries if entry.get("name") == "swe_bench")

    model_metadata = {
        name: _digest(model_path / name)
        for name in (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        )
        if (model_path / name).is_file()
    }
    patch_files = (
        root / "patches" / "sglang" / "qwen3_5_language_model_only.patch",
        root / "patches" / "verl" / "sglang-runtime-audit.patch",
    )
    mem_available_kib = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        if line.startswith("MemAvailable:")
    )
    candidate = {
        "uni_agent_base_sha": _git(root, "rev-parse", "HEAD"),
        "uni_agent_branch": _git(root, "branch", "--show-current"),
        "uni_agent_dirty_patch_sha256": _git_diff_sha(root),
        "uni_agent_status": _git(root, "status", "--short"),
        "verl_base_sha": _git(root / "verl", "rev-parse", "HEAD"),
        "verl_dirty_patch_sha256": _git_diff_sha(root / "verl"),
        "uni_agent_changed_file_sha256": _changed_file_hashes(root),
        "verl_changed_file_sha256": _changed_file_hashes(root / "verl"),
        "sglang_ref": baseline["repositories"]["sglang"]["ref"],
        "sglang_sha": baseline["repositories"]["sglang"]["sha"],
        "patch_sha256": {path.name: _digest(path) for path in patch_files},
    }
    manifest: dict[str, Any] = {
        "run_id": _required_env("RUN_ID"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"hostname": os.uname().nodename},
        "candidate": candidate,
        "driver": {
            "python_executable": _required_env("PYTHON_BIN"),
            "python_version": sys.version,
            "environment": baseline["driver_environment"],
        },
        "sglang_actor_environment": None
        if os.environ["ENGINE"] != "sglang"
        else {
            "python_executable": _required_env("SGLANG_PYTHON"),
            "pythonpath": _required_env("SGLANG_ACTOR_PYTHONPATH"),
            "venv_site_packages": _required_env("SGLANG_SITE_PACKAGES"),
            "inherited_site_packages": _required_env("SGLANG_SYSTEM_SITE_PACKAGES"),
            "flashinfer_cubin_dir": _required_env("FLASHINFER_CUBIN_DIR"),
        },
        "model": {
            "path": str(model_path),
            "model_type": baseline["inputs"]["model_type"],
            "architectures": baseline["inputs"]["architectures"],
            "language_model_only": selected["language_model_only"],
            "metadata_sha256": model_metadata,
        },
        "dataset": {
            "path": str(data_path),
            "rows": baseline["inputs"]["dataset_rows"],
            "instance_id": baseline["inputs"]["instance_id"],
            "sha256": data_sha,
            "canonical_sandbox_image": baseline["inputs"]["canonical_sandbox_image"],
        },
        "sandbox": {
            "provider": task["sandbox"]["provider"],
            "runtime_timeout": task["sandbox"]["runtime_timeout"],
            "eval_timeout": task["eval_timeout"],
            "agent_run_timeout": task["agent"]["run_timeout"],
            "agent_step_limit": task["agent"]["step_limit"],
            "tool_image": task["sandbox"]["sandbox_kwargs"]["mounts"][0]["image_url"],
            "tool_python": task["agent"]["tool_python"],
            "run_agent_script": task["agent"]["run_agent_script"],
            "resolved_digest": baseline["mini_swe_agent_runtime"]["image_digest"],
        },
        "credential_reference": {
            "path": str(credential_path),
            "mode": "0600",
            "owner_uid": credential_stat.st_uid,
            "values_read_or_logged": False,
        },
        "rollout": {
            "engine": os.environ["ENGINE"],
            "shape": selected,
            "ray_num_cpus": int(os.environ["RAY_NUM_CPUS"]) if os.environ.get("RAY_NUM_CPUS") else None,
            "ray_object_store_memory_bytes": int(os.environ["RAY_OBJECT_STORE_MEMORY_BYTES"])
            if os.environ.get("RAY_OBJECT_STORE_MEMORY_BYTES")
            else None,
            "ray_idle_worker_killing_memory_threshold_bytes": int(
                os.environ["RAY_idle_worker_killing_memory_threshold_bytes"]
            )
            if os.environ.get("RAY_idle_worker_killing_memory_threshold_bytes")
            else None,
            "host_mem_available_bytes_before_ray": mem_available_kib * 1024,
            "data_parallel_size": 1,
            "temperature": float(_required_env("TEMPERATURE")),
            "top_p": float(_required_env("TOP_P")),
            "top_k": int(_required_env("TOP_K")),
            "response_length": int(_required_env("RESPONSE_LENGTH")),
            "seed": 42,
            "enforce_eager": os.environ.get("ENFORCE_EAGER", "0").lower() in {"1", "true"},
            "gpu_memory_utilization": float(_required_env("GPU_MEMORY_UTILIZATION")),
            "tool_parser": _required_env("TOOL_PARSER"),
            "trajectory_selection": "all",
        },
        "gpu_baseline": gpu_rows,
        "paths": {
            "output_dir": str(output_dir),
            "ray_temp_dir": _required_env("RAY_TMPDIR"),
            "task_config": str(task_config_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary_path, manifest_path)
    print(f"pre-run manifest frozen: {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    _write_manifest()
