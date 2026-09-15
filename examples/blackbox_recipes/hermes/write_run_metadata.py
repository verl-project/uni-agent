#!/usr/bin/env python3
"""Write reproducibility metadata for one Hermes inference run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

VERL_UPSTREAM_COMMIT = "a9f2985159536a607211dcac730d3f5d55028950"


def _git(repo: Path, *args: str) -> str | None:
    if not repo.is_dir() or not (repo / ".git").exists():
        return None
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _gitlink(repo: Path) -> str | None:
    value = _git(repo, "ls-tree", "HEAD", "verl")
    return value.split()[2] if value and len(value.split()) >= 3 else None


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--sample-metadata", required=True)
    parser.add_argument("--tool-parser", required=True)
    parser.add_argument("--nnodes", required=True)
    parser.add_argument("--n-gpus-per-node", required=True)
    parser.add_argument("--tensor-parallel-size", required=True)
    parser.add_argument("--limit", required=True)
    parser.add_argument("--n", required=True)
    parser.add_argument("--concurrency", required=True)
    parser.add_argument("--gateway-count", required=True)
    parser.add_argument("--cuda-home", default="")
    args = parser.parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    repo = Path(args.repo_root).expanduser().resolve()
    metadata = json.loads(Path(args.sample_metadata).read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "recipe": "hermes",
        "status": "prepared",
        "instance_id": metadata.get("instance_id"),
        "source_index": metadata.get("source_index"),
        "source_data_path": metadata.get("source_path"),
        "model_path": str(Path(args.model_path).expanduser()),
        "data_path": str(Path(args.data_path).expanduser()),
        "task_config": args.task_config,
        "images": {
            "task_image_from_sample": metadata.get("sandbox_image"),
            "tool_image": os.getenv(
                "HERMES_TOOL_IMAGE",
                "swr.cn-east-3.myhuaweicloud.com/openyuanrong/hermes-agent-tool:20260915",
            ),
            "tool_image_digest": os.getenv("HERMES_TOOL_IMAGE_DIGEST") or None,
        },
        "uni_agent_commit": _git(repo, "rev-parse", "HEAD"),
        "uni_agent_branch": _git(repo, "branch", "--show-current"),
        "uni_agent_dirty": bool(_git(repo, "status", "--porcelain")),
        "verl_gitlink": _gitlink(repo),
        "verl_git_commit": _git(repo / "verl", "rev-parse", "HEAD"),
        "verl_upstream_commit": VERL_UPSTREAM_COMMIT,
        "verl_checkout_mode": "git" if _git(repo / "verl", "rev-parse", "HEAD") else "source_archive",
        "verl_compatibility": {
            "mode": "standalone_placement_only_worker",
            "inference_env": "VERL_STANDALONE_NO_WEIGHT_SYNC=1",
            "ray_version_observed": "2.56.1",
            "reason": "remote186 Ray 2.56 standalone CheckpointEngineWorker actor EOF; training path unchanged",
        },
        "openyuanrong_compatibility": {
            "provider_api": "legacy",
            "legacy_sdk": "akernel_sdk==0.9.22",
            "legacy_openyuanrong_sdk": "openyuanrong-sdk==0.7.61rc1",
            "helper_python": "/home/zxh/miniconda3/envs/uni-agent/bin/python",
            "libffi": "/usr/lib/x86_64-linux-gnu/libffi.so.7",
            "credentials_source": "protected env file; raw values intentionally not recorded",
        },
        "hermes_commit": "5eb99eb2844b22ebb723711b8e6a0bbb80bb5f04",
        "host_runtime": {
            "python": sys.version.split()[0],
            "ray": _dist_version("ray"),
            "vllm": _dist_version("vllm"),
            "torch": _dist_version("torch"),
            "swebench": _dist_version("swebench"),
            "openyuanrong_sandbox": _dist_version("openyuanrong-sandbox"),
        },
        "sidecar_runtime": {"python": "3.12.13", "python_build_standalone_release": "20260602", "uv": "0.12.14"},
        "cuda_home": args.cuda_home or None,
        "hardware": {
            "nnodes": int(args.nnodes),
            "n_gpus_per_node": int(args.n_gpus_per_node),
            "tensor_parallel_size": int(args.tensor_parallel_size),
        },
        "rollout": {
            "limit": int(args.limit),
            "n": int(args.n),
            "concurrency": int(args.concurrency),
            "gateway_count": int(args.gateway_count),
            "tool_parser": args.tool_parser,
            "language_model_only": True,
        },
        "paths": {
            "prepared_sample": str(root / "prepared_sample.parquet"),
            "logs": str(root / "logs"),
            "hermes": str(root / "hermes"),
            "result": str(root / "result.json"),
            "acceptance": str(root / "acceptance.json"),
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    resolved = [
        "recipe: hermes",
        f"instance_id: {metadata.get('instance_id')}",
        f"model_path: {Path(args.model_path).expanduser()}",
        f"data_path: {Path(args.data_path).expanduser()}",
        f"task_config: {args.task_config}",
        f"tool_parser: {args.tool_parser}",
        "language_model_only: true",
        f"nnodes: {args.nnodes}",
        f"n_gpus_per_node: {args.n_gpus_per_node}",
        f"tensor_parallel_size: {args.tensor_parallel_size}",
        f"limit: {args.limit}",
        f"n: {args.n}",
        f"concurrency: {args.concurrency}",
        f"gateway_count: {args.gateway_count}",
        f"cuda_home: {args.cuda_home or 'unset'}",
        "openyuanrong_provider_api: legacy",
        "openyuanrong_helper_python: /home/zxh/miniconda3/envs/uni-agent/bin/python",
        "openyuanrong_credentials: protected env file (raw values omitted)",
        "session_policy: one session, trajectory_selection=all",
        "hermes_toolsets: [terminal, file]",
        "hermes_compression: disabled",
        "hermes_memory: disabled",
        "hermes_delegation: disabled",
    ]
    (root / "resolved_config.yaml").write_text("\n".join(resolved) + "\n", encoding="utf-8")
    command = " ".join(
        [
            "MODEL_PATH=<model-path>",
            "DATA_PATH=<dataset.parquet>",
            f"OUTPUT_DIR={root}",
            f"INSTANCE_ID={metadata.get('instance_id')}",
            "OPENYUANRONG_SERVER_ADDRESS=<redacted>",
            "OPENYUANRONG_TOKEN=<redacted>",
            "bash examples/blackbox_recipes/hermes/run_infer_hermes.sh",
        ]
    )
    (root / "command.txt").write_text(command + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
