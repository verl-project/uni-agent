#!/usr/bin/env bash
# Inference for the Codex black-box recipe.
#
# Usage:
#   bash examples/codex/run_infer_codex.sh
#
# The script uses the same verl rollout path as the training launcher.  verl
# starts the engine, AgentFrameworkRolloutAdapter owns the gateway session, and
# the Codex task is dispatched through task_runner/run_task.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

export DEPLOYMENT="${DEPLOYMENT:-openyuanrong}"
export OPENYUANRONG_SERVER_ADDRESS="${OPENYUANRONG_SERVER_ADDRESS:?OPENYUANRONG_SERVER_ADDRESS must be set}"
export OPENYUANRONG_TOKEN="${OPENYUANRONG_TOKEN:?OPENYUANRONG_TOKEN must be set}"
export OPENYUANRONG_TUNNEL_SSL_VERIFY="${OPENYUANRONG_TUNNEL_SSL_VERIFY:-0}"
export TUNNEL_SSL_VERIFY="${TUNNEL_SSL_VERIFY:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

DATA_PATH="${DATA_PATH:-/swe_bench_verified.parquet}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen/Qwen3.5-9B}"
TASK_CONFIG="${TASK_CONFIG:-examples/codex/task_config_codex.yaml}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
LIMIT="${LIMIT:-1}"
N="${N:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
LOG_DIR="${LOG_DIR:-/tmp/uni_agent_codex_infer_logs}"
RESULT_PATH="${RESULT_PATH:-/tmp/uni_agent_codex_infer_result.json}"
PROMPT_LENGTH="${PROMPT_LENGTH:-8192}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-122880}"

# `ray job submit` does not forward this shell's environment to the job
# driver, so pass the provider settings explicitly in the runtime environment.
ray job submit \
    --runtime-env-json "$(cat <<JSON
{
  "env_vars": {
    "PYTHONPATH": "verl",
    "DEPLOYMENT": "${DEPLOYMENT}",
    "OPENYUANRONG_SERVER_ADDRESS": "${OPENYUANRONG_SERVER_ADDRESS}",
    "OPENYUANRONG_TOKEN": "${OPENYUANRONG_TOKEN}",
    "OPENYUANRONG_TUNNEL_SSL_VERIFY": "${OPENYUANRONG_TUNNEL_SSL_VERIFY}",
    "TUNNEL_SSL_VERIFY": "${TUNNEL_SSL_VERIFY}",
    "CUDA_VISIBLE_DEVICES": "${CUDA_VISIBLE_DEVICES}"
  },
  "excludes": ["/.git/"]
}
JSON
)" \
    --working-dir . \
    -- python3 examples/inference/parallel_infer_verl.py \
    --data-path "${DATA_PATH}" \
    --model-path "${MODEL_PATH}" \
    --task-config "${TASK_CONFIG}" \
    --language-model-only \
    --disable-thinking \
    --prompt-length "${PROMPT_LENGTH}" \
    --response-length "${RESPONSE_LENGTH}" \
    --tool-parser "${TOOL_PARSER}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --nnodes "${NNODES}" \
    --n-gpus-per-node "${N_GPUS_PER_NODE}" \
    --gateway-count "${GATEWAY_COUNT}" \
    --limit "${LIMIT}" \
    --n "${N}" \
    --concurrency "${CONCURRENCY}" \
    --log-dir "${LOG_DIR}" \
    --result-path "${RESULT_PATH}"

# Keep the single-sample acceptance check close to the launch command.  A
# successful process alone is insufficient: exactly one session must be scored
# and the SWE-bench verifier must return reward 1.
python3 - "${RESULT_PATH}" "${LIMIT}" "${N}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
expected_prompts = int(sys.argv[2])
expected_sessions = expected_prompts * int(sys.argv[3])
if expected_prompts != 1 or expected_sessions != 1:
    raise SystemExit("Codex acceptance requires LIMIT=1 and N=1")
if not result_path.is_file():
    raise SystemExit(f"inference result was not written: {result_path}")

result = json.loads(result_path.read_text(encoding="utf-8"))
if result.get("num_prompts") != 1:
    raise SystemExit(f"expected one prompt, got {result.get('num_prompts')!r}")
if result.get("num_scored_sessions") != 1:
    raise SystemExit(f"expected one scored session, got {result.get('num_scored_sessions')!r}")
scores = result.get("scores")
if scores != [1.0]:
    raise SystemExit(f"expected one solved SWE-bench trajectory with score 1.0, got {scores!r}")
print(f"Codex inference acceptance passed: one prompt, one solved trajectory, score={scores[0]}")
PY
