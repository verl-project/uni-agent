#!/usr/bin/env bash
# Run standard preprocessed data through verl, the framework and Hermes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."
: "${DATA_PATH:?Set DATA_PATH to standard preprocessed SWE-bench parquet}"
: "${MODEL_PATH:?Set MODEL_PATH to the policy model}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new absolute directory outside the checkout}"
: "${RAY_RUNTIME_ENV:?Set RAY_RUNTIME_ENV to your protected Ray runtime-env YAML file}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RAY_BIN="${RAY_BIN:-ray}"
[[ "${OUTPUT_DIR}" == /* ]] || { echo "OUTPUT_DIR must be absolute" >&2; exit 2; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Use a new OUTPUT_DIR for each run" >&2; exit 2; }
[[ -f "${DATA_PATH}" && -f "${RAY_RUNTIME_ENV}" ]] || { echo "Missing data or runtime-env file" >&2; exit 2; }
mkdir -p "${OUTPUT_DIR}"

ARGS=(
    --data-path "${DATA_PATH}"
    --model-path "${MODEL_PATH}"
    --task-config "${TASK_CONFIG:-examples/blackbox_recipes/hermes_swe_task/task_config_hermes.yaml}"
    --tool-parser "${TOOL_PARSER:-qwen3_coder}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
    --nnodes "${NNODES:-1}"
    --n-gpus-per-node "${N_GPUS_PER_NODE:-1}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
    --limit "${LIMIT:-1}" --n "${N:-1}"
    --concurrency "${CONCURRENCY:-1}" --gateway-count "${GATEWAY_COUNT:-1}"
    --log-dir "${OUTPUT_DIR}/logs" --result-path "${OUTPUT_DIR}/result.json"
)
if [[ "${LANGUAGE_MODEL_ONLY:-1}" == "1" ]]; then
    ARGS+=(--language-model-only)
fi

# Ray applies this runtime environment to the driver and its workers. The file
# belongs to the operator and stays outside the uploaded working directory.
# Submission waits and propagates failures; the launcher never manages Ray.
: "${RAY_API_SERVER_ADDRESS:?Set RAY_API_SERVER_ADDRESS to the Ray Jobs API address}"
"${RAY_BIN}" job submit \
    --address "${RAY_API_SERVER_ADDRESS}" \
    --runtime-env "${RAY_RUNTIME_ENV}" --working-dir . \
    -- "${PYTHON_BIN}" examples/inference/parallel_infer_verl.py "${ARGS[@]}"
