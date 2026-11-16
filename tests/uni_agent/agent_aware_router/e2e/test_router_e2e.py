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

"""GPU end-to-end coverage for agent-aware routing."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.level1, pytest.mark.gpu]

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", ".."))
_RUN_INFER = os.path.join(_PROJECT_ROOT, "examples", "agent_aware_router", "run_infer.sh")
# Simulated sandbox runner: drives the gateway session for real but answers
# tool calls with canned observations (no container). See utils/simulated_sandbox.py.
_SIMULATED_RUNNER_FQN = "tests.uni_agent.agent_aware_router.e2e.utils.simulated_sandbox.simulated_runner"
_MODEL = os.environ.get("VLLM_MODEL", "/path/to/Qwen/Qwen3-4B-Instruct-2507")
_DATASET = os.environ.get("SWEBENCH_DATASET", "/path/to/swe_bench_verified_modal.parquet")
_TASK_CONFIG = os.path.join(_PROJECT_ROOT, "examples", "agent_aware_router", "task_config_mini_swe_agent.yaml")
_LOG_DIR = Path("/tmp/e2e_router_logs")


def _run_infer(timeout: int = 600) -> str:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _LOG_DIR / "router_e2e.log"

    # GPU config: CUDA_VISIBLE_DEVICES controls which GPUs Ray/vLLM see;
    # --n-gpus-per-node must match the count.
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    num_gpus = len(cuda_vis.split(","))
    cmd = [
        "bash",
        _RUN_INFER,
        "--model-path",
        _MODEL,
        "--data-path",
        _DATASET,
        "--task-config",
        _TASK_CONFIG,
        "--simulated-runner-fqn",
        _SIMULATED_RUNNER_FQN,
        "--n-gpus-per-node",
        str(num_gpus),
        "--tensor-parallel-size",
        "2",
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
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = cuda_vis

    with log_file.open("w") as stream:
        result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    log = log_file.read_text()
    assert result.returncode == 0, f"run_infer.sh failed with {result.returncode}:\n{log[-4000:]}"
    return log


class TestKVCAwareRouterE2E:
    def test_kvc_aware_router_full_e2e(self):
        log = _run_infer()

        assert "route(): replicas=" in log, "No routing decisions in log"
        assert "CAPACITY_TOKEN_AWARE" in log, "No CAPACITY_TOKEN_AWARE scoring — strategy may have failed"
        assert "mean rm_score" in log, "run_infer.sh did not complete (no rm_score)"
        assert "inference summary" in log, "run_infer.sh did not finish (no inference summary)"
        assert "load_threshold=0.8" in log, "load-threshold != 0.8, the config not work."
