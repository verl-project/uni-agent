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
RUN_INFER = PROJECT_ROOT / "examples" / "agent_aware_router" / "run_infer.sh"
TASK_CONFIG = PROJECT_ROOT / "examples" / "agent_aware_router" / "task_config_mini_swe_agent.yaml"
# Simulated sandbox runner: drives the gateway session for real but answers
# tool calls with canned observations (no container). See utils/simulated_sandbox.py.
SIMULATED_RUNNER_FQN = "tests.uni_agent.agent_aware_router.e2e.utils.simulated_sandbox.simulated_runner"

# The default dataset is a 4-row fabricated fixture (invented problem statements,
# no open-dataset content) checked into the repo: the simulated sandbox runner
# only reads the prompt text, so synthetic rows drive the full routing pipeline.
# Real runs point SWEBENCH_DATASET at a preprocessed parquet
# (python -m uni_agent.tasks.swe_bench.preprocess).
DATASET = os.environ.get(
    "SWEBENCH_DATASET",
    str(PROJECT_ROOT / "tests/uni_agent/agent_aware_router/e2e/testdata/smoke_swe_bench.parquet"),
)
# No fabricated substitute for a model checkpoint: VLLM_MODEL must point at a
# real local model (gate runners get it provisioned); local runs override it.
MODEL = os.environ.get("VLLM_MODEL", "/path/to/Qwen/Qwen3-4B-Instruct-2507")
LOG_DIR = Path(os.environ.get("ROUTER_E2E_LOG_DIR", "/tmp/e2e_router_logs"))


def _require_assets() -> None:
    """Fail fast with a clear message when an asset path does not resolve."""
    for env_var, path in (("VLLM_MODEL", MODEL), ("SWEBENCH_DATASET", DATASET)):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{env_var} resolves to '{path}' which does not exist. "
                "Set the env var to a local asset path."
            )


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill the launcher's whole process group and reap the leader.

    ``run_infer.sh`` invokes ``python run_infer.py`` without ``exec``, so killing
    only the bash wrapper would orphan the vLLM/Ray workers and their device
    memory. SIGTERM first lets them clean up; SIGKILL is the backstop.
    """
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

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / log_name

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    if device == "ascend":
        env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(visible)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible)

    cmd = [
        "bash",
        str(RUN_INFER),
        "--model-path",
        MODEL,
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

    # Own session so a timeout can kill the whole process group, not just the
    # bash wrapper (see _kill_process_group).
    try:
        with log_file.open("w") as stream:
            proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        log = log_file.read_text(errors="replace")
        raise AssertionError(f"run_infer.sh timed out after {timeout}s:\n{log[-4000:]}") from None

    log = log_file.read_text(errors="replace")
    assert proc.returncode == 0, f"run_infer.sh failed with {proc.returncode}:\n{log[-4000:]}"
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
