#!/usr/bin/env bash
# Run Codex tasks through the standard verl-managed inference path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

export DEPLOYMENT="${DEPLOYMENT:-openyuanrong}"
export OPENYUANRONG_SERVER_ADDRESS="${OPENYUANRONG_SERVER_ADDRESS:?Set OPENYUANRONG_SERVER_ADDRESS}"
export OPENYUANRONG_TOKEN="${OPENYUANRONG_TOKEN:?Set OPENYUANRONG_TOKEN}"
export OPENYUANRONG_TUNNEL_SSL_VERIFY="${OPENYUANRONG_TUNNEL_SSL_VERIFY:-0}"
export TUNNEL_SSL_VERIFY="${TUNNEL_SSL_VERIFY:-0}"

DATA_PATH="${DATA_PATH:?Set DATA_PATH to a prepared Parquet dataset}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the model checkpoint}"
TASK_CONFIG="${TASK_CONFIG:-examples/codex/task_config_codex.yaml}"
ENGINE="${ENGINE:-vllm}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
LIMIT="${LIMIT:-1}"
N="${N:-1}"

INFER_ARGS=(
    --data-path "${DATA_PATH}"
    --model-path "${MODEL_PATH}"
    --task-config "${TASK_CONFIG}"
    --engine "${ENGINE}"
    --tool-parser "${TOOL_PARSER}"
)

if [[ "${LANGUAGE_MODEL_ONLY:-true}" == "true" ]]; then
    INFER_ARGS+=(--language-model-only)
fi
if [[ "${DISABLE_THINKING:-true}" == "true" ]]; then
    INFER_ARGS+=(--disable-thinking)
fi

append_optional_arg() {
    local variable_name="$1"
    local option="$2"
    local value="${!variable_name:-}"
    if [[ -n "${value}" ]]; then
        INFER_ARGS+=("${option}" "${value}")
    fi
}

append_optional_arg PROMPT_LENGTH --prompt-length
append_optional_arg RESPONSE_LENGTH --response-length
append_optional_arg TEMPERATURE --temperature
append_optional_arg TOP_P --top-p
append_optional_arg TENSOR_PARALLEL_SIZE --tensor-parallel-size
append_optional_arg NNODES --nnodes
append_optional_arg N_GPUS_PER_NODE --n-gpus-per-node
append_optional_arg GPU_MEMORY_UTILIZATION --gpu-memory-utilization
append_optional_arg GATEWAY_COUNT --gateway-count
append_optional_arg LIMIT --limit
append_optional_arg N --n
append_optional_arg CONCURRENCY --concurrency
append_optional_arg RESULT_PATH --result-path
if [[ "${LOG_DIR+x}" == x ]]; then
    INFER_ARGS+=(--log-dir "${LOG_DIR}")
fi

# Pass credentials and provider settings to the Ray job without relying on
# shell interpolation inside JSON.
RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

keys = (
    "DEPLOYMENT",
    "OPENYUANRONG_SERVER_ADDRESS",
    "OPENYUANRONG_TOKEN",
    "OPENYUANRONG_TUNNEL_SSL_VERIFY",
    "TUNNEL_SSL_VERIFY",
)
env_vars = {"PYTHONPATH": "verl", **{key: os.environ[key] for key in keys}}
print(json.dumps({"env_vars": env_vars, "excludes": ["/.git/"]}))
PY
)"

ray job submit \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
    --working-dir . \
    -- python3 examples/inference/parallel_infer_verl.py "${INFER_ARGS[@]}"
