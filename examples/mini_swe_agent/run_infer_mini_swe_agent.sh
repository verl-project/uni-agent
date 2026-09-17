#!/usr/bin/env bash
# Inference for the mini-swe-agent blackbox recipe.
#
# Usage:
#   bash examples/mini_swe_agent/run_infer_mini_swe_agent.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

export DEPLOYMENT="${DEPLOYMENT:-openyuanrong}"
export OPENYUANRONG_SERVER_ADDRESS="${OPENYUANRONG_SERVER_ADDRESS:?OPENYUANRONG_SERVER_ADDRESS must be set}"
export OPENYUANRONG_TOKEN="${OPENYUANRONG_TOKEN:?OPENYUANRONG_TOKEN must be set}"
export OPENYUANRONG_TUNNEL_SSL_VERIFY="${OPENYUANRONG_TUNNEL_SSL_VERIFY:-0}"
export TUNNEL_SSL_VERIFY="${TUNNEL_SSL_VERIFY:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

DATA_PATH="${DATA_PATH:-/swe_bench_verified.parquet}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen/Qwen3.5-9B}"
TASK_CONFIG="${TASK_CONFIG:-examples/mini_swe_agent/task_config_mini_swe_agent.yaml}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
LIMIT="${LIMIT:-1}"
N="${N:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
LOG_DIR="${LOG_DIR:-/tmp/uni_agent_logs}"
RESULT_PATH="${RESULT_PATH:-/tmp/mini_swe_agent_infer_result.json}"

# `ray job submit` does not forward this shell's env into the job driver, so
# the job's runtime env is spelled out here and passed via --runtime-env-json.
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
    --tool-parser "${TOOL_PARSER}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --nnodes "${NNODES}" \
    --n-gpus-per-node "${N_GPUS_PER_NODE}" \
    --limit "${LIMIT}" \
    --n "${N}" \
    --concurrency "${CONCURRENCY}" \
    --log-dir "${LOG_DIR}" \
    --result-path "${RESULT_PATH}"
