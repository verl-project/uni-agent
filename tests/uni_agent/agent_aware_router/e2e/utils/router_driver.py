# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared launcher for the agent-aware router e2e tests.

Runs ``examples/agent_aware_router/run_infer.sh`` once with the simulated
(container-free) sandbox runner and returns the captured log. Backend-agnostic:
NVIDIA runs use ``CUDA_VISIBLE_DEVICES``; Ascend runs use
``ASCEND_RT_VISIBLE_DEVICES`` and ``--device ascend``.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
RUN_INFER = PROJECT_ROOT / "examples/agent_aware_router/run_infer.sh"
TASK_CONFIG = PROJECT_ROOT / "examples/agent_aware_router/task_config_mini_swe_agent.yaml"
SIMULATED_RUNNER_FQN = "tests.uni_agent.agent_aware_router.e2e.utils.simulated_sandbox.simulated_runner"

# 4-row fabricated fixture (no open-dataset content): the simulated sandbox only
# reads prompt text; real runs override SWEBENCH_DATASET with a preprocessed parquet.
DATASET = os.environ.get(
    "SWEBENCH_DATASET",
    str(PROJECT_ROOT / "tests/uni_agent/agent_aware_router/e2e/testdata/smoke_swe_bench.parquet"),
)
# Gate default: the Qwen3-0.6B snapshot provisioned on the npu runner host;
# local runs override VLLM_MODEL with a local checkpoint dir.
MODEL = os.environ.get(
    "VLLM_MODEL",
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
)
LOG_DIR = Path(os.environ.get("ROUTER_E2E_LOG_DIR", "/tmp/e2e_router_logs"))


def _resolve_model_path() -> str:
    """MODEL must be a flat checkpoint dir readable in this environment."""
    path = Path(MODEL)
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"VLLM_MODEL={MODEL} is not readable here: no config.json at its root.")
    return str(path)


def _require_assets() -> None:
    """Fail fast with a clear message when an asset path does not resolve."""
    if not Path(DATASET).exists():
        raise FileNotFoundError(
            f"SWEBENCH_DATASET resolves to '{DATASET}' which does not exist. Set the env var to a local asset path."
        )


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill the whole process group: run_infer.sh wraps python without exec, so
    killing only the bash wrapper would orphan the vLLM/Ray workers holding
    device memory. SIGTERM lets them clean up; SIGKILL is the backstop."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run_infer(
    *,
    device: str,
    visible_devices: str,
    tensor_parallel_size: int,
    log_name: str,
    timeout: int = 900,
) -> str:
    """Run one inference pass and return its log; assert the process exit code."""
    if device not in ("gpu", "ascend"):
        raise ValueError(f"device must be 'gpu' or 'ascend', got {device!r}")

    visible = [v.strip() for v in visible_devices.split(",") if v.strip()]
    if not visible:
        raise ValueError("visible_devices must list at least one device")

    _require_assets()
    model_path = _resolve_model_path()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / log_name

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    # Keep per-worker logs un-deduplicated so a worker crash stack is not
    # folded into "[repeated Nx across cluster]".
    env["RAY_DEDUP_LOGS"] = "0"
    if device == "ascend":
        env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(visible)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible)

    cmd = [
        "bash",
        str(RUN_INFER),
        "--model-path",
        model_path,
        "--data-path",
        DATASET,
        "--task-config",
        str(TASK_CONFIG),
        "--simulated-runner-fqn",
        SIMULATED_RUNNER_FQN,
        "--device",
        device,
        "--n-gpus-per-node",
        str(len(visible)),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-samples",
        "4",
        "--n",
        "2",
        "--max-model-len",
        "8192",
        "--response-length",
        "4096",
        "--prompt-length",
        "3072",
        "--load-threshold",
        "0.8",
    ]

    try:
        with log_file.open("w") as stream:
            proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        log = log_file.read_text(errors="replace")
        raise AssertionError(f"run_infer.sh timed out after {timeout}s:\n{log[-12000:]}") from None

    log = log_file.read_text(errors="replace")
    assert proc.returncode == 0, f"run_infer.sh failed with {proc.returncode}:\n{log[-12000:]}"
    return log


def assert_router_signals(log: str, *, replicas: int | None = None) -> None:
    """Assert the router telemetry expected from a completed inference pass."""
    assert "route(): replicas=" in log, "No routing decisions in log"
    assert "CAPACITY_TOKEN_AWARE" in log, "No CAPACITY_TOKEN_AWARE scoring — strategy may have failed"
    assert "mean rm_score" in log, "run_infer.sh did not complete (no rm_score)"
    assert "inference summary" in log, "run_infer.sh did not finish (no inference summary)"
    assert "load_threshold=0.8" in log, "load-threshold != 0.8, the config not work."
    if replicas is not None:
        assert f"route(): replicas={replicas}" in log, f"Expected {replicas} replicas in routing log"
