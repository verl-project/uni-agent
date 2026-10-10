#!/usr/bin/env bash
# One-sample Harbor Terminus-2 rollout through the verl-managed Uni-Agent Gateway.
#
# Optional overrides:
#   DATA_PATH=... MODEL_PATH=... LIMIT=1 CONCURRENCY=1 bash "$0"
#   SERVED_MODEL_NAME=openai/Qwen3.6-35B-A3B bash "$0"
#   TASK_CONFIG=... MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... bash "$0"
#
# The default Task Config uses Harbor Docker for a same-network smoke test.
# A Modal Task Config additionally requires Modal credentials and a network
# route from the Modal sandbox to the session-scoped Gateway URL.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${REPO_ROOT}"

DATA_PATH="${DATA_PATH:-${HOME}/data/uni_agent/harbor_terminal-bench_terminal-bench-2-1.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.6-35B-A3B}"
MODEL_BASENAME="${MODEL_PATH%/}"
MODEL_BASENAME="${MODEL_BASENAME##*/}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-hosted_vllm/${MODEL_BASENAME}}"
TASK_CONFIG="${TASK_CONFIG:-examples/quickstart/harbor/task_config_terminus2_rollout.yaml}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
ENGINE="${ENGINE:-vllm}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${TENSOR_PARALLEL_SIZE}}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
LIMIT="${LIMIT:-1}"
N="${N:-1}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

LOG_DIR="${LOG_DIR:-/tmp/uni_agent_logs/harbor_terminus2}"
RESULT_PATH="${RESULT_PATH:-/tmp/harbor_terminus2_rollout.json}"
export HARBOR_PACKAGE="${HARBOR_PACKAGE:-harbor[modal]==0.16.1}"

if [[ ! -f "${REPO_ROOT}/verl/verl/__init__.py" ]]; then
    echo "The verl submodule is not initialized. Run: git submodule update --init --recursive" >&2
    exit 1
fi
if ! command -v ray >/dev/null 2>&1; then
    echo "ray is not installed or not available on PATH." >&2
    exit 1
fi
if [[ "${DATA_PATH}" != *"://"* && ! -f "${DATA_PATH}" ]]; then
    echo "Dataset not found: ${DATA_PATH}" >&2
    echo "Run uni_agent.tasks.harbor.preprocess first, or set DATA_PATH." >&2
    exit 1
fi

# Install the pinned Harbor client on every Ray worker. Forward Modal
# credentials only when the caller provides them for a Modal Task Config.
RUNTIME_ENV_JSON="$(
    python3 - <<'PY'
import json
import os

env_vars = {
    "PYTHONPATH": "verl",
    "PYTHONNOUSERSITE": "1",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "VLLM_DISABLE_COMPILE_CACHE": "1",
}
for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
    if value := os.environ.get(key):
        env_vars[key] = value

print(
    json.dumps(
        {
            "pip": {"packages": [os.environ["HARBOR_PACKAGE"]]},
            "env_vars": env_vars,
            "excludes": ["/.git/"],
        }
    )
)
PY
)"

echo "Agent:             Terminus-2"
echo "Dataset:           ${DATA_PATH}"
echo "Model checkpoint:  ${MODEL_PATH}"
echo "Agent model name:  ${SERVED_MODEL_NAME}"
echo "Hardware:          ${NNODES} node(s), ${N_GPUS_PER_NODE} GPU(s)/node, TP=${TENSOR_PARALLEL_SIZE}"
echo "Smoke size:        ${LIMIT} prompt(s) x n=${N}, concurrency=${CONCURRENCY}"

ray job submit \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
    --working-dir . \
    -- python3 examples/inference/parallel_infer_verl.py \
    --data-path "${DATA_PATH}" \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --task-config "${TASK_CONFIG}" \
    --engine "${ENGINE}" \
    --tool-parser "${TOOL_PARSER}" \
    --allowed-request-sampling-param-keys temperature top_p top_k \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --nnodes "${NNODES}" \
    --n-gpus-per-node "${N_GPUS_PER_NODE}" \
    --gateway-count "${GATEWAY_COUNT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --response-length "${RESPONSE_LENGTH}" \
    --limit "${LIMIT}" \
    --n "${N}" \
    --concurrency "${CONCURRENCY}" \
    --log-dir "${LOG_DIR}" \
    --result-path "${RESULT_PATH}"
