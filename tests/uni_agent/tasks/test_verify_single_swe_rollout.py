from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from examples.mini_swe_agent.verify_single_swe_rollout import verify_run


INSTANCE_ID = "astropy__astropy-12907"
SESSION_ID = "session-sample-0-rollout-0-0123456789abcdef"
RUN_ID = "test-run-sglang"


def _write_run(run_dir: Path, *, backend: str, score: float, resolved: bool) -> None:
    logs_dir = run_dir / "logs"
    session_dir = logs_dir / SESSION_ID
    session_dir.mkdir(parents=True)
    data_path = run_dir.parent.parent / "swe_bench.parquet"
    data_path.write_bytes(b"synthetic one-row SWE-bench fixture")
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    uid = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    # The finalized TQ key carries uid/sample/session indices, not the runtime
    # session UUID. The framework log is the authoritative runtime-session link.
    tq_key = f"{uid}_0_0"
    if backend == "sglang":
        backend_file = "/repo/.envs/sglang-v0.5.20/lib/python3.11/site-packages/sglang/__init__.py"
        backend_python = "/repo/.envs/sglang-v0.5.20/bin/python"
        actor_identity = {
            "python_executable": backend_python,
            "sglang_module_file": backend_file,
        }
    else:
        backend_file = f"/env/site-packages/{backend}/__init__.py"
        backend_python = "/env/bin/python"
        actor_identity = None
    manifest = {
        "run_id": RUN_ID,
        "ray_temp_dir": str(run_dir.parent.parent / ".tmp" / "r12345"),
        "engine": backend,
        "cuda_visible_devices": "0,1,2,3,4,5,6,7",
        "backend_runtime": {
            "module": backend,
            "version": "0.5.20" if backend == "sglang" else "0.24.0",
            "module_file": backend_file,
            "python_executable": backend_python,
        },
        "engine_actor_identities": [actor_identity] if actor_identity else [],
        "engine_server_addresses": ["127.0.0.1:30000"],
        "model": {"path": "/models/Qwen3.5-9B", "language_model_only": True},
        "input": {
            "data_path": str(data_path),
            "num_prompts": 1,
            "n": 1,
            "instance_ids": [INSTANCE_ID],
        },
        "rollout": {
            "nnodes": 1,
            "n_gpus_per_node": 8,
            "tensor_parallel_size": 8,
            "data_parallel_size": 1,
            "gateway_count": 1,
            "max_concurrent_sessions": 1,
            "trajectory_selection": "all",
            "calculate_log_probs": True,
            "enforce_eager": False,
        },
    }
    result = {
        "num_prompts": 1,
        "n": 1,
        "num_scored_sessions": 1,
        "scores": [score],
        "scores_by_uid": {uid: [score]},
        "final_tq_keys": [tq_key],
        "all_tq_trajectory_keys": [tq_key],
        "uid_status": {uid: "finished"},
    }
    trajectory = {
        "session_id": SESSION_ID,
        "num_trajectories": 1,
        "trajectories": [
            {
                "num_turns": 1,
                "finished": True,
                "reward_score": score,
                "has_logprobs": True,
                "model_token_count": 2,
            }
        ],
    }
    for path, value in (
        (
            run_dir / "manifest.json",
            {"dataset": {"path": str(data_path), "rows": 1, "instance_id": INSTANCE_ID, "sha256": data_sha}},
        ),
        (logs_dir / "runtime_manifest.json", manifest),
        (run_dir / "result.json", result),
        (session_dir / "trajectory.json", trajectory),
    ):
        path.write_text(json.dumps(value), encoding="utf-8")
    np.savez_compressed(
        session_dir / "trajectory.npz",
        traj0_prompt_ids=np.asarray([10, 11], dtype=np.int32),
        traj0_response_ids=np.asarray([20, 21, 22], dtype=np.int32),
        traj0_response_mask=np.asarray([1, 1, 0], dtype=np.int8),
        traj0_response_logprobs=np.asarray([-0.5, -0.7, -1.2], dtype=np.float32),
    )
    (session_dir / "framework.log").write_text(
        f"session {SESSION_ID} start\n"
        f"session {SESSION_ID}: scored via agent_runner\n",
        encoding="utf-8",
    )
    (session_dir / "task.log").write_text(
        "mini_swe_agent: done exit_status=Submitted submission=10 chars rc=0\n"
        "eval finished in 1.0s (exit_code=0)\n"
        f"reward for {INSTANCE_ID}: resolved={str(resolved)}\n",
        encoding="utf-8",
    )
    with (run_dir / "gpu-snapshot.csv").open("w", encoding="utf-8") as gpu_file:
        with (run_dir / "gpu-processes.csv").open("w", encoding="utf-8") as process_file:
            for index in range(8):
                uuid = f"GPU-{index:04d}"
                gpu_file.write(f'{index}, {uuid}, NVIDIA GeForce RTX 3090, 1024 MiB\n')
                process_file.write(f"{uuid}, {1000 + index}, python, 1024 MiB\n")


@pytest.mark.cpu
@pytest.mark.level0
def test_acceptance_verifier_accepts_one_complete_unsolved_control(tmp_path):
    run_dir = tmp_path / "artifacts" / RUN_ID
    _write_run(run_dir, backend="vllm", score=0.0, resolved=False)

    report = verify_run(
        run_dir,
        backend="vllm",
        expected_instance_id=INSTANCE_ID,
        protocol_only=True,
    )

    assert report["status"] == "PASS"


@pytest.mark.cpu
@pytest.mark.level0
def test_acceptance_verifier_requires_solved_reward_for_formal_run(tmp_path):
    run_dir = tmp_path / "artifacts" / RUN_ID
    _write_run(run_dir, backend="sglang", score=0.0, resolved=False)
    ray_logs = tmp_path / ".tmp" / "r12345" / "session_latest" / "logs"
    ray_logs.mkdir(parents=True)
    (ray_logs / "job-driver-1.log").write_text(
        "sglang runtime identity: pid=1001 python=/repo/.envs/sglang-v0.5.20/bin/python sglang_version=0.5.20 "
        "sglang_module=/repo/.envs/sglang-v0.5.20/lib/python3.11/site-packages/sglang/__init__.py "
        "torch_version=2.13.0+cu130 torch_module=/env/torch "
        "ray_version=2.56.1 ray_module=/env/ray replica_rank=0 node_rank=0 "
        "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\n"
        "sglang request dispatch: request_id=backend-request-1 prompt_tokens=10\n",
        encoding="utf-8",
    )

    report = verify_run(
        run_dir,
        backend="sglang",
        expected_instance_id=INSTANCE_ID,
    )

    assert report["status"] == "FAIL"
    assert next(check for check in report["checks"] if check["id"] == "solved")["status"] == "FAIL"


@pytest.mark.cpu
@pytest.mark.level0
def test_acceptance_verifier_rejects_filtered_or_multiple_trajectories(tmp_path):
    run_dir = tmp_path / "artifacts" / RUN_ID
    _write_run(run_dir, backend="vllm", score=1.0, resolved=True)
    trajectory_path = run_dir / "logs" / SESSION_ID / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    trajectory["num_trajectories"] = 2
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")

    report = verify_run(
        run_dir,
        backend="vllm",
        expected_instance_id=INSTANCE_ID,
    )

    assert report["status"] == "FAIL"
    assert next(check for check in report["checks"] if check["id"] == "one_raw_framework_trajectory")["status"] == "FAIL"


@pytest.mark.cpu
@pytest.mark.level0
def test_acceptance_verifier_requires_live_sglang_request_linkage(tmp_path):
    run_dir = tmp_path / "artifacts" / RUN_ID
    _write_run(run_dir, backend="sglang", score=1.0, resolved=True)
    ray_logs = tmp_path / ".tmp" / "r12345" / "session_latest" / "logs"
    ray_logs.mkdir(parents=True)
    (ray_logs / "job-driver-1.log").write_text(
        "sglang runtime identity: pid=1001 python=/repo/.envs/sglang-v0.5.20/bin/python sglang_version=0.5.20 "
        "sglang_module=/repo/.envs/sglang-v0.5.20/lib/python3.11/site-packages/sglang/__init__.py "
        "torch_version=2.13.0+cu130 torch_module=/env/torch "
        "ray_version=2.56.1 ray_module=/env/ray replica_rank=0 node_rank=0 "
        "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\n"
        "sglang request dispatch: request_id=backend-request-2 prompt_tokens=10\n",
        encoding="utf-8",
    )

    report = verify_run(
        run_dir,
        backend="sglang",
        expected_instance_id=INSTANCE_ID,
    )

    assert report["status"] == "PASS"
