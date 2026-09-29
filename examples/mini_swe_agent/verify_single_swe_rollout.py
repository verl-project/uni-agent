#!/usr/bin/env python3
"""Strictly verify one mini-swe-agent inference run and its raw trajectory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def _record(
    checks: list[dict[str, Any]],
    name: str,
    *,
    passed: bool,
    expected: Any,
    observed: Any,
    evidence_path: Path | str,
    reason: str,
) -> None:
    checks.append(
        {
            "id": name,
            "status": "PASS" if passed else ("UNVERIFIED" if observed is None else "FAIL"),
            "expected": expected,
            "observed": observed,
            "evidence_path": str(evidence_path),
            "reason": reason,
        }
    )


def _read_json(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return {}, "JSON root must be an object"
    return value, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_run(
    run_dir: Path,
    *,
    backend: str,
    expected_instance_id: str,
    protocol_only: bool = False,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    require_solved = not protocol_only
    log_dir = run_dir / "logs"
    manifest_path = log_dir / "runtime_manifest.json"
    result_path = run_dir / "result.json"
    pre_run_manifest_path = run_dir / "manifest.json"
    manifest, manifest_error = _read_json(manifest_path)
    result, result_error = _read_json(result_path)
    pre_run_manifest, pre_run_manifest_error = _read_json(pre_run_manifest_path)

    _record(
        checks,
        "manifest_valid",
        passed=manifest_error is None,
        expected="readable runtime_manifest.json",
        observed=None if manifest_error else "valid JSON object",
        evidence_path=manifest_path,
        reason=manifest_error or "effective runtime identity was recorded",
    )
    _record(
        checks,
        "result_valid",
        passed=result_error is None,
        expected="readable result.json",
        observed=None if result_error else "valid JSON object",
        evidence_path=result_path,
        reason=result_error or "framework result summary was recorded",
    )
    pre_dataset = pre_run_manifest.get("dataset") or {}
    pre_dataset_path = Path(str(pre_dataset.get("path", ""))) if pre_dataset.get("path") else None
    pre_dataset_sha = pre_dataset.get("sha256")
    try:
        pre_dataset_sha_matches = (
            pre_dataset_path is not None
            and pre_dataset_path.is_file()
            and isinstance(pre_dataset_sha, str)
            and _sha256(pre_dataset_path) == pre_dataset_sha
        )
    except OSError:
        pre_dataset_sha_matches = False
    _record(
        checks,
        "pre_run_manifest_and_dataset_hash",
        passed=(
            pre_run_manifest_error is None
            and pre_dataset.get("rows") == 1
            and pre_dataset.get("instance_id") == expected_instance_id
            and pre_dataset_sha_matches
        ),
        expected="pre-run manifest freezes one expected row and its unchanged SHA256",
        observed={
            "manifest_error": pre_run_manifest_error,
            "rows": pre_dataset.get("rows"),
            "instance_id": pre_dataset.get("instance_id"),
            "sha256_matches": pre_dataset_sha_matches,
        },
        evidence_path=pre_run_manifest_path,
        reason="the actual input file is re-hashed against the pre-run frozen digest",
    )

    runtime = manifest.get("backend_runtime") or {}
    _record(
        checks,
        "selected_backend",
        passed=manifest.get("engine") == backend and runtime.get("module") == backend,
        expected=backend,
        observed={"engine": manifest.get("engine"), "module": runtime.get("module")},
        evidence_path=manifest_path,
        reason="the engine selection and imported backend module agree",
    )
    _record(
        checks,
        "backend_runtime_identity",
        passed=bool(runtime.get("version") and runtime.get("module_file")),
        expected="backend version and imported module path",
        observed={"version": runtime.get("version"), "module_file": runtime.get("module_file")},
        evidence_path=manifest_path,
        reason="the active runtime package is recorded rather than inferred from configuration",
    )
    if backend == "sglang":
        actor_identities = manifest.get("engine_actor_identities") or []
        actor_identity = actor_identities[0] if len(actor_identities) == 1 else {}
        _record(
            checks,
            "sglang_actor_uses_isolated_python",
            passed=(
                len(actor_identities) == 1
                and str(actor_identity.get("python_executable", "")).endswith("/.envs/sglang-v0.5.20/bin/python")
                and "/.envs/sglang-v0.5.20/lib/python3.11/site-packages/sglang/" in str(actor_identity.get("sglang_module_file", ""))
            ),
            expected="one server actor running SGLang from the isolated repository-local venv",
            observed=actor_identity or None,
            evidence_path=manifest_path,
            reason="the driver environment stays unchanged and only the SGLang HTTP actor uses its isolated Python",
        )

    input_config = manifest.get("input") or {}
    rollout = manifest.get("rollout") or {}
    model = manifest.get("model") or {}
    runtime_instance_ids = input_config.get("instance_ids")
    runtime_data_path = input_config.get("data_path")
    runtime_id_matches = runtime_instance_ids == [expected_instance_id]
    frozen_single_row_matches = (
        pre_dataset.get("instance_id") == expected_instance_id
        and pre_dataset.get("rows") == 1
        and runtime_data_path == pre_dataset.get("path")
        and pre_dataset_sha_matches
    )
    _record(
        checks,
        "single_fixed_input",
        passed=(
            input_config.get("num_prompts") == 1
            and input_config.get("n") == 1
            and (runtime_id_matches or frozen_single_row_matches)
        ),
        expected={"num_prompts": 1, "n": 1, "instance_ids": [expected_instance_id]},
        observed={
            "num_prompts": input_config.get("num_prompts"),
            "n": input_config.get("n"),
            "runtime_instance_ids": runtime_instance_ids,
            "frozen_pre_run_instance_id": pre_dataset.get("instance_id"),
            "runtime_data_path_matches_frozen": runtime_data_path == pre_dataset.get("path"),
            "pre_run_sha256_matches": pre_dataset_sha_matches,
        },
        evidence_path=manifest_path,
        reason="the run used exactly the frozen SWE-bench instance and one sample",
    )
    _record(
        checks,
        "qwen35_text_only_model",
        passed=Path(str(model.get("path", ""))).name == "Qwen3.5-9B" and model.get("language_model_only") is True,
        expected={"model": "Qwen3.5-9B", "language_model_only": True},
        observed={"path": model.get("path"), "language_model_only": model.get("language_model_only")},
        evidence_path=manifest_path,
        reason="the run used the selected Qwen3.5-9B language-only mode",
    )
    _record(
        checks,
        "single_node_eight_gpu_shape",
        passed=(
            rollout.get("nnodes") == 1
            and rollout.get("n_gpus_per_node") == 8
            and rollout.get("tensor_parallel_size") == 8
            and rollout.get("data_parallel_size") == 1
            and manifest.get("cuda_visible_devices") == "0,1,2,3,4,5,6,7"
        ),
        expected={"nnodes": 1, "gpus": 8, "tp": 8, "dp": 1, "cuda_visible_devices": "0,1,2,3,4,5,6,7"},
        observed={
            "nnodes": rollout.get("nnodes"),
            "gpus": rollout.get("n_gpus_per_node"),
            "tp": rollout.get("tensor_parallel_size"),
            "dp": rollout.get("data_parallel_size"),
            "cuda_visible_devices": manifest.get("cuda_visible_devices"),
        },
        evidence_path=manifest_path,
        reason="the runtime manifest records the requested single-node TP=8 resource shape",
    )
    _record(
        checks,
        "single_gateway_session_policy",
        passed=(
            rollout.get("gateway_count") == 1
            and rollout.get("max_concurrent_sessions") == 1
            and rollout.get("trajectory_selection") == "all"
        ),
        expected={"gateway_count": 1, "max_concurrent_sessions": 1, "trajectory_selection": "all"},
        observed={
            "gateway_count": rollout.get("gateway_count"),
            "max_concurrent_sessions": rollout.get("max_concurrent_sessions"),
            "trajectory_selection": rollout.get("trajectory_selection"),
        },
        evidence_path=manifest_path,
        reason="one session is allowed and materialized trajectories are not filtered",
    )
    _record(
        checks,
        "logprob_capture_enabled",
        passed=rollout.get("calculate_log_probs") is True,
        expected=True,
        observed=rollout.get("calculate_log_probs"),
        evidence_path=manifest_path,
        reason="the framework trajectory is expected to retain model logprobs for integrity checks",
    )
    _record(
        checks,
        "cuda_graph_policy_recorded",
        passed=type(rollout.get("enforce_eager")) is bool,
        expected="an explicit boolean backend CUDA graph policy",
        observed=rollout.get("enforce_eager"),
        evidence_path=manifest_path,
        reason="the backend memory/workload policy must be frozen per attempt",
    )

    server_addresses = manifest.get("engine_server_addresses")
    _record(
        checks,
        "one_live_engine_endpoint",
        passed=isinstance(server_addresses, list) and len(server_addresses) == 1,
        expected="one server address",
        observed=server_addresses,
        evidence_path=manifest_path,
        reason="LLMServerManager reported one endpoint for the single TP=8 replica",
    )

    try:
        num_prompts = int(result.get("num_prompts", -1))
        n = int(result.get("n", -1))
        num_scored_sessions = int(result.get("num_scored_sessions", -1))
        scores = result.get("scores")
        final_keys = result.get("final_tq_keys")
        all_keys = result.get("all_tq_trajectory_keys")
        uid_status = result.get("uid_status")
        result_shape_ok = (
            num_prompts == 1
            and n == 1
            and num_scored_sessions == 1
            and isinstance(scores, list)
            and len(scores) == 1
            and isinstance(final_keys, list)
            and len(final_keys) == 1
            and isinstance(all_keys, list)
            and len(all_keys) == 1
            and isinstance(uid_status, dict)
            and len(uid_status) == 1
            and set(uid_status.values()) == {"finished"}
        )
    except (TypeError, ValueError):
        result_shape_ok = False
        scores, final_keys, all_keys, uid_status = None, None, None, None
    _record(
        checks,
        "one_scored_session_and_one_tq_trajectory",
        passed=result_shape_ok,
        expected="1 prompt, n=1, one scored session, one final key, one trajectory key, status=finished",
        observed={
            "num_prompts": result.get("num_prompts"),
            "n": result.get("n"),
            "num_scored_sessions": result.get("num_scored_sessions"),
            "scores": scores,
            "final_tq_keys": final_keys,
            "all_tq_trajectory_keys": all_keys,
            "uid_status": uid_status,
        },
        evidence_path=result_path,
        reason="the unfiltered TransferQueue result contains exactly one finished session and trajectory",
    )

    trajectory_paths = sorted(log_dir.rglob("trajectory.json")) if log_dir.is_dir() else []
    trajectory_path = trajectory_paths[0] if len(trajectory_paths) == 1 else log_dir / "<single trajectory.json>"
    trajectory_file_ok = len(trajectory_paths) == 1
    trajectory, trajectory_error = _read_json(trajectory_path) if trajectory_file_ok else ({}, None)
    trajectories = trajectory.get("trajectories") or []
    one_raw_trajectory = trajectory_file_ok and trajectory_error is None and trajectory.get("num_trajectories") == 1 and len(trajectories) == 1
    _record(
        checks,
        "one_raw_framework_trajectory",
        passed=one_raw_trajectory,
        expected="exactly one trajectory.json with num_trajectories=1",
        observed={
            "trajectory_json_count": len(trajectory_paths),
            "num_trajectories": trajectory.get("num_trajectories"),
            "trajectory_entries": len(trajectories),
            "parse_error": trajectory_error,
        },
        evidence_path=trajectory_path,
        reason="the raw session materialization is recorded before any optional filtering",
    )

    session_id = trajectory.get("session_id")
    trajectory_meta = trajectories[0] if len(trajectories) == 1 else {}
    final_key_uids = []
    final_key_shape_ok = isinstance(final_keys, list)
    if final_key_shape_ok:
        for key in final_keys:
            parts = str(key).rsplit("_", 2)
            if len(parts) == 3:
                final_key_uids.append(parts[0])
            else:
                final_key_shape_ok = False
    scores_by_uid = result.get("scores_by_uid")
    scored_uids = set(scores_by_uid) if isinstance(scores_by_uid, dict) else set(uid_status or {})
    framework_log_path = trajectory_path.parent / "framework.log"
    try:
        framework_log_text = framework_log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        framework_log_text = ""
    framework_session_linkage = bool(session_id) and session_id in framework_log_text and "scored via agent_runner" in framework_log_text
    result_session_linkage = (
        bool(session_id)
        and final_key_shape_ok
        and len(final_key_uids) == 1
        and final_key_uids[0] in scored_uids
        and final_key_uids[0] in set(uid_status or {})
        and framework_session_linkage
    )
    _record(
        checks,
        "session_identity_linkage",
        passed=result_session_linkage,
        expected="one scored TransferQueue key links to the trajectory UID and the same session framework log",
        observed={
            "trajectory_session_id": session_id,
            "scored_tq_uids": final_key_uids,
            "result_uids": sorted(scored_uids),
            "framework_session_linkage": framework_session_linkage,
        },
        evidence_path=f"{trajectory_path}; {framework_log_path}",
        reason="the result format stores a numeric sample/session suffix, so the runtime session is linked through the same session directory's framework log",
    )

    npz_path = trajectory_path.with_name("trajectory.npz")
    arrays: dict[str, np.ndarray] = {}
    npz_error = None
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        npz_error = f"{type(exc).__name__}: {exc}"
    prompt_ids = arrays.get("traj0_prompt_ids")
    response_ids = arrays.get("traj0_response_ids")
    response_mask = arrays.get("traj0_response_mask")
    response_logprobs = arrays.get("traj0_response_logprobs")
    token_arrays_ok = (
        npz_error is None
        and prompt_ids is not None
        and prompt_ids.ndim == 1
        and len(prompt_ids) > 0
        and response_ids is not None
        and response_ids.ndim == 1
        and len(response_ids) > 0
        and response_mask is not None
        and response_mask.ndim == 1
        and len(response_mask) == len(response_ids)
        and set(np.unique(response_mask).tolist()).issubset({0, 1})
        and response_logprobs is not None
        and response_logprobs.ndim == 1
        and len(response_logprobs) == len(response_ids)
        and bool(np.isfinite(response_logprobs).all())
        and trajectory_meta.get("has_logprobs") is True
        and trajectory_meta.get("model_token_count") == int(response_mask.sum())
    )
    _record(
        checks,
        "trajectory_token_and_logprob_integrity",
        passed=token_arrays_ok,
        expected="non-empty 1-D prompt/response ids; aligned binary response mask and finite logprobs",
        observed=None
        if npz_error
        else {
            "prompt_tokens": None if prompt_ids is None else len(prompt_ids),
            "response_tokens": None if response_ids is None else len(response_ids),
            "mask_tokens": None if response_mask is None else len(response_mask),
            "logprob_tokens": None if response_logprobs is None else len(response_logprobs),
            "num_turns": trajectory_meta.get("num_turns"),
            "model_token_count": trajectory_meta.get("model_token_count"),
        },
        evidence_path=npz_path,
        reason=npz_error or "token IDs, response mask, and requested logprobs are aligned and valid",
    )

    session_dir = trajectory_path.parent
    framework_log_path = session_dir / "framework.log"
    task_log_path = session_dir / "task.log"
    framework_text = framework_log_path.read_text(encoding="utf-8", errors="replace") if framework_log_path.is_file() else ""
    task_text = task_log_path.read_text(encoding="utf-8", errors="replace") if task_log_path.is_file() else ""
    _record(
        checks,
        "session_logs_present",
        passed=framework_log_path.is_file() and task_log_path.is_file(),
        expected="framework.log and task.log for the same session directory",
        observed={"framework_log": framework_log_path.is_file(), "task_log": task_log_path.is_file()},
        evidence_path=session_dir,
        reason="the framework lifecycle and sandbox task logs are both retained",
    )
    _record(
        checks,
        "agent_submission_and_verifier_completion",
        passed=(
            "exit_status=Submitted" in task_text
            and expected_instance_id in task_text
            and re.search(r"eval finished.*exit_code=0", task_text, flags=re.IGNORECASE) is not None
            and re.search(
                r"resolved\s*[=:]\s*true" if require_solved else r"resolved\s*[=:]\s*(true|false)",
                task_text,
                flags=re.IGNORECASE,
            )
            is not None
        ),
        expected="Submitted plus verifier exit_code=0 and resolution status in task.log",
        observed={
            "submitted": "exit_status=Submitted" in task_text,
            "expected_instance_id_logged": expected_instance_id in task_text,
            "verifier_exit_zero": re.search(r"eval finished.*exit_code=0", task_text, flags=re.IGNORECASE) is not None,
            "resolution_recorded": re.search(
                r"resolved\s*[=:]\s*true" if require_solved else r"resolved\s*[=:]\s*(true|false)",
                task_text,
                flags=re.IGNORECASE,
            )
            is not None,
        },
        evidence_path=task_log_path,
        reason="mini-swe-agent submitted a patch and the SWE-bench verifier completed successfully",
    )

    score = scores[0] if isinstance(scores, list) and len(scores) == 1 else None
    trajectory_reward = trajectory_meta.get("reward_score")
    finished = trajectory_meta.get("finished")
    if require_solved:
        reward_ok = score == 1.0 and trajectory_reward == 1.0 and finished is True
        resolved_ok = re.search(r"resolved\s*[=:]\s*true", task_text, flags=re.IGNORECASE) is not None
    else:
        reward_ok = finished is True and isinstance(score, (int, float))
        resolved_ok = re.search(r"resolved\s*[=:]\s*(true|false)", task_text, flags=re.IGNORECASE) is not None
    _record(
        checks,
        "task_resolution_and_reward",
        passed=reward_ok and resolved_ok,
        expected="finished=true and verifier/reward agree with requested acceptance mode",
        observed={"finished": finished, "trajectory_reward": trajectory_reward, "score": score, "protocol_only": protocol_only},
        evidence_path=task_log_path,
        reason="finished and verifier resolution are checked separately from the scalar reward",
    )

    if backend == "sglang":
        paths = manifest.get("paths") or {}
        ray_temp_dir = manifest.get("ray_temp_dir") or paths.get("ray_temp_dir")
        ray_log_root = Path(ray_temp_dir) if isinstance(ray_temp_dir, str) and ray_temp_dir else run_dir.parent.parent / ".tmp" / "<missing-ray-temp>"
        ray_logs = []
        if ray_log_root.is_dir():
            for pattern in ("worker-*.out", "worker-*.err", "job-driver-*.log"):
                ray_logs.extend(ray_log_root.rglob(pattern))
        matching_log_files: dict[Path, list[str]] = {}
        for path in ray_logs:
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            request_ids_in_file = {
                match.group(1)
                for line in lines
                if "sglang request dispatch:" in line
                if (match := re.search(r"request_id=([A-Za-z0-9-]+)", line))
            }
            if request_ids_in_file:
                matching_log_files[path] = lines
        actor_lines = [
            (path, line)
            for path, lines in matching_log_files.items()
            for line in lines
            if "sglang runtime identity:" in line
        ]
        request_lines = [
            (path, line)
            for path, lines in matching_log_files.items()
            for line in lines
            if "sglang request dispatch:" in line
        ]
        expected_devices = "0,1,2,3,4,5,6,7"
        actor_line_values = sorted(
            {
                re.sub(r"\x1b\[[0-9;]*m", "", line).split("sglang runtime identity:", 1)[1].strip()
                for _, line in actor_lines
                if "sglang runtime identity:" in line
            }
        )
        actor_identity_text = f"sglang runtime identity: {actor_line_values[0]}" if actor_line_values else ""
        actor_ok = (
            len(actor_line_values) == 1
            and "sglang_version=0.5.20" in actor_identity_text
            and "torch_version=2.13.0+cu130" in actor_identity_text
            and f"CUDA_VISIBLE_DEVICES={expected_devices}" in actor_identity_text
        )
        _record(
            checks,
            "sglang_actor_and_gpu_identity",
            passed=actor_ok,
            expected="one SGLang 0.5.20 actor with torch 2.13.0+cu130 and CUDA devices 0-7",
            observed=[actor_identity_text] if actor_identity_text else None,
            evidence_path=ray_log_root,
            reason="identity must come from the live SGLang Ray actor, not only the driver manifest",
        )
        request_ids = set()
        for _, line in request_lines:
            match = re.search(r"request_id=([A-Za-z0-9-]+)", line)
            if match:
                request_ids.add(match.group(1))
        request_linked = bool(session_id) and bool(request_lines) and framework_session_linkage and bool(request_ids)
        _record(
            checks,
            "sglang_request_session_linkage",
            passed=request_linked,
            expected="non-empty SGLang request IDs are recorded in the same job log as the sole materialized framework session",
            observed={
                "request_count": len(request_lines),
                "request_ids": sorted(request_ids),
                "framework_session_linkage": framework_session_linkage,
            },
            evidence_path=f"{ray_log_root}; {framework_log_path}",
            reason="SGLang generates per-request IDs; the framework session directory and scored framework log provide the attempt-level linkage",
        )

    gpu_snapshot_path = run_dir / "gpu-snapshot.csv"
    gpu_processes_path = run_dir / "gpu-processes.csv"
    gpu_rows = []
    gpu_error = None
    try:
        with gpu_snapshot_path.open(encoding="utf-8", newline="") as file:
            gpu_rows = list(csv.reader(file))
        process_rows = []
        with gpu_processes_path.open(encoding="utf-8", newline="") as file:
            process_rows = list(csv.reader(file))
        uuid_to_index = {}
        gpu_ok = len(gpu_rows) == 8
        for row in gpu_rows:
            if len(row) < 4:
                gpu_ok = False
                continue
            index, uuid, name, memory = (cell.strip() for cell in row[:4])
            uuid_to_index[uuid] = index
            memory_mib = int(re.sub(r"[^0-9]", "", memory) or "0")
            gpu_ok &= "RTX 3090" in name and memory_mib > 0
        process_gpu_indices = {
            uuid_to_index[row[0].strip()]
            for row in process_rows
            if row and row[0].strip() in uuid_to_index
        }
        gpu_ok &= process_gpu_indices == {str(index) for index in range(8)}
    except (OSError, ValueError, IndexError) as exc:
        gpu_error = f"{type(exc).__name__}: {exc}"
        gpu_ok = False
        process_gpu_indices = set()
    _record(
        checks,
        "eight_physical_gpu_processes",
        passed=gpu_ok,
        expected="8 RTX 3090 rows with memory use and compute processes mapped to every GPU UUID",
        observed=None
        if gpu_error
        else {"gpu_rows": len(gpu_rows), "process_gpu_indices": sorted(process_gpu_indices)},
        evidence_path=f"{gpu_snapshot_path}; {gpu_processes_path}",
        reason=gpu_error or "nvidia-smi captured active compute use for devices 0 through 7",
    )

    if require_solved:
        _record(
            checks,
            "solved",
            passed=score == 1.0 and trajectory_reward == 1.0 and finished is True,
            expected="reward=1.0, finished=true, resolved=true",
            observed={"score": score, "trajectory_reward": trajectory_reward, "finished": finished},
            evidence_path=result_path,
            reason="SWE-bench resolution, rather than process exit alone, is the success criterion",
        )

    return {
        "status": "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL",
        "run_dir": str(run_dir),
        "backend": backend,
        "expected_instance_id": expected_instance_id,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--expected-instance-id", required=True)
    parser.add_argument(
        "--protocol-only",
        action="store_true",
        help="Validate a vLLM control without requiring the instance to be solved.",
    )
    args = parser.parse_args()

    report = verify_run(
        args.run_dir.resolve(),
        backend=args.backend,
        expected_instance_id=args.expected_instance_id,
        protocol_only=args.protocol_only,
    )
    report_path = args.run_dir.resolve() / "acceptance.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{report['status']}: {report_path}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
