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

ENGINE="${ENGINE:-vllm}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
GATEWAY_COUNT="${GATEWAY_COUNT:-4}"
case "${ENGINE}" in
    vllm|sglang) ;;
    *) echo "ENGINE must be vllm or sglang" >&2; exit 2 ;;
esac
EXTRA_ARGS=(--engine "${ENGINE}")
for option in LANGUAGE_MODEL_ONLY ENFORCE_EAGER; do
    case "${!option,,}" in
        1|true)
            if [[ "${option}" == "LANGUAGE_MODEL_ONLY" ]]; then
                EXTRA_ARGS+=(--language-model-only)
            else
                EXTRA_ARGS+=(--enforce-eager)
            fi
            ;;
        0|false) ;;
        *) echo "${option} must be 0, 1, false, or true" >&2; exit 2 ;;
    esac
done
RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

names = (
    "DEPLOYMENT", "OPENYUANRONG_SERVER_ADDRESS", "OPENYUANRONG_TOKEN",
    "OPENYUANRONG_TUNNEL_SSL_VERIFY", "TUNNEL_SSL_VERIFY", "CUDA_VISIBLE_DEVICES",
)
env_vars = {name: os.environ[name] for name in names}
env_vars["PYTHONPATH"] = "verl"
print(json.dumps({"env_vars": env_vars, "excludes": ["/.git/"]}))
PY
)"

ray job submit \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
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
    --gateway-count "${GATEWAY_COUNT}" \
    --log-dir "${LOG_DIR}" \
    --result-path "${RESULT_PATH}" \
    "${EXTRA_ARGS[@]}"
