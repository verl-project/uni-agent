#!/usr/bin/env bash
# Standalone inference for the blackbox claude-code recipe.
# Runs rollout + reward only (no Megatron trainer) and reports resolve rate.
#
# Usage:
#   bash examples/blackbox_recipes/claude_code/run_infer.sh
#
# All configurable via environment variables (see defaults below).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${REPO_ROOT}"

# ── Model & data ─────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-${HOME}/models/Qwen3.5-9B}"
DATA_PATH="${DATA_PATH:-${HOME}/data/swe_agent/swe_bench_verified.parquet}"

# ── Inference parameters ─────────────────────────────────────────────────
MAX_SAMPLES="${MAX_SAMPLES:--1}"
PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-131072}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
N="${N:-1}"
ENGINE="${ENGINE:-vllm}"
TP="${TP:-4}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-8}"

# ── Agent parameters ─────────────────────────────────────────────────────
AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-100}"
CLAUDE_CODE_TOOL_IMAGE="${CLAUDE_CODE_TOOL_IMAGE:-swr.cn-east-3.myhuaweicloud.com/openyuanrong/claude-code-tool:latest}"
SWE_AGENT_RUN_TIMEOUT="${SWE_AGENT_RUN_TIMEOUT:-7200}"

# ── AKernel (remote sandbox) ─────────────────────────────────────────────
export AKERNEL_SERVER_ADDRESS="${AKERNEL_SERVER_ADDRESS:-}"
export AKERNEL_TOKEN="${AKERNEL_TOKEN:-}"
export AKERNEL_TUNNEL_SSL_VERIFY="${AKERNEL_TUNNEL_SSL_VERIFY:-0}"

# ── Logging & env ────────────────────────────────────────────────────────
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.7}"
export AGENT_MAX_TURNS
export SWE_AGENT_EVAL_TIMEOUT="${SWE_AGENT_EVAL_TIMEOUT:-600}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"

echo "=== Claude Code Blackbox Inference ==="
echo "Model:       ${MODEL_PATH}"
echo "Data:        ${DATA_PATH}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Engine:      ${ENGINE} (TP=${TP})"
echo "Tool image:  ${CLAUDE_CODE_TOOL_IMAGE}"
echo "Batch:       n=${N}, gateway=${GATEWAY_COUNT}, max_sessions=${MAX_CONCURRENT_SESSIONS}"
echo "====================================="

python examples/blackbox_recipes/claude_code/parallel_infer.py \
    --model-path "${MODEL_PATH}" \
    --data-path "${DATA_PATH}" \
    --max-samples "${MAX_SAMPLES}" \
    --prompt-length "${PROMPT_LENGTH}" \
    --response-length "${RESPONSE_LENGTH}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --n "${N}" \
    --engine "${ENGINE}" \
    --tensor-parallel-size "${TP}" \
    --nnodes "${NNODES}" \
    --n-gpus-per-node "${N_GPUS_PER_NODE}" \
    --gateway-count "${GATEWAY_COUNT}" \
    --max-concurrent-sessions "${MAX_CONCURRENT_SESSIONS}" \
    --tool-image "${CLAUDE_CODE_TOOL_IMAGE}" \
    --run-timeout "${SWE_AGENT_RUN_TIMEOUT}" \
    --max-turns "${AGENT_MAX_TURNS}"
