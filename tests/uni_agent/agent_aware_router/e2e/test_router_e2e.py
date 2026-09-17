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

"""E2E test: KVCAware router via run_infer.sh — full agent loop with routing.

Launches ``run_infer.sh`` (drives ``parallel_infer_verl_kvc.py`` with the
simulated-sandbox runner swapped in), waits for completion, then checks:
  1. Routing decisions produced ("routed to server")
  2. COMBINED scoring (not falling back to random)
  3. mean rm_score printed (end-to-end completion)
  4. Inference summary printed (agent loop actually ran)

This is a GPU test (needs real vLLM + GPU + model + dataset).
"""

from __future__ import annotations

import os
import subprocess

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
_LOG_DIR = "/tmp/e2e_router_logs"


def _run_infer(timeout: int = 600) -> str:
    """Run run_infer.sh with KVCAware router. Returns log content."""
    os.makedirs(_LOG_DIR, exist_ok=True)
    log_file = os.path.join(_LOG_DIR, "router_e2e.log")

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

    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    return open(log_file).read()


class TestKVCAwareRouterE2E:
    """E2E: run_infer.sh with KVCAware router — full agent loop."""

    def test_kvc_aware_router_full_e2e(self):
        """
        Feature: KVCAware router end-to-end via run_infer.sh
        Description: run full agent loop with --router-config-path, verify:
          - routing decisions ("routed to server" >= 1)
          - COMBINED scoring (not random fallback)
          - mean rm_score printed (end-to-end completion)
          - inference summary printed (agent loop actually ran)
        """
        log = _run_infer()

        # 1. Routing decisions
        assert "routed to server" in log, "No routing decisions in log"
        routing_count = log.count("routed to server")
        assert routing_count >= 1, f"Expected >=1 routing decision, got {routing_count}"

        # 2. CAPACITY_TOKEN_AWARE scoring (not random fallback)
        assert "CAPACITY_TOKEN_AWARE" in log, "No CAPACITY_TOKEN_AWARE scoring — strategy may have failed"

        # 3. End-to-end completion
        assert "mean rm_score" in log, "run_infer.sh did not complete (no rm_score)"

        # 4. Agent loop actually ran (inference summary block at the end)
        assert "inference summary" in log, "run_infer.sh did not finish (no inference summary)"

        # 5. load-threshold == 0.8
        assert "load_threshold=0.8" in log, "load-threshold != 0.8, the config not work."
