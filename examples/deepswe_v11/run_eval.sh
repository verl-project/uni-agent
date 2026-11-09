#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${E2B_API_KEY:?Set E2B_API_KEY before running DeepSWE evaluation}"
: "${E2B_API_URL:?Set E2B_API_URL before running DeepSWE evaluation}"
: "${E2B_SANDBOX_URL:?Set E2B_SANDBOX_URL before running DeepSWE evaluation}"

# Use the installed verl, or an explicitly selected DeepSeek-V4 checkout.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${VERL_ROOT:-}" ]]; then
    if [[ ! -f "${VERL_ROOT}/verl/__init__.py" ]]; then
        echo "VERL_ROOT must point to a verl checkout: ${VERL_ROOT}" >&2
        exit 2
    fi
    VERL_ROOT="$(cd "${VERL_ROOT}" && pwd)"
    export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH}"
fi
# Keep Ray workers in the prepared environment instead of replaying uv's lockfile.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

for command in python harbor ray; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "Missing ${command}; follow examples/deepswe_v11/README.md to prepare the environment." >&2
        exit 2
    fi
done

DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/deepswe_v11.parquet}"
: "${MODEL_PATH:?Set MODEL_PATH to the DeepSeek-V4-Flash-0731 model directory}"
TASK_CONFIG="${TASK_CONFIG:-${SCRIPT_DIR}/task_config.yaml}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/deepswe_v11}"

for input_file in "${DATA_PATH}" "${TASK_CONFIG}"; do
    if [[ ! -f "${input_file}" ]]; then
        echo "Input file not found: ${input_file}" >&2
        exit 2
    fi
done

if ray status >/dev/null 2>&1; then
    echo "An existing Ray cluster is running; wait for it to finish, then run: ray stop -f" >&2
    exit 2
fi

python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

expected = {"TransferQueue": "0.1.11", "transformers": "5.16.0", "tokenizers": "0.23.1"}
for package, required in expected.items():
    try:
        actual = version(package)
    except PackageNotFoundError:
        actual = "not installed"
    if actual != required:
        raise SystemExit(
            f"Expected {package}=={required}, found {actual}. "
            "Follow examples/deepswe_v11/README.md to prepare this Python environment."
        )
PY

mkdir -p "${OUTPUT_DIR}/${RUN_ID}/logs"
echo "Run directory: ${OUTPUT_DIR}/${RUN_ID}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
fi

python -u examples/inference/parallel_infer_verl.py \
  --data-path "${DATA_PATH}" \
  --model-path "${MODEL_PATH}" \
  --served-model-name openai/DeepSeek-V4-Flash-0731 \
  --task-config "${TASK_CONFIG}" \
  --engine vllm \
  --tool-parser deepseek_v4 \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-4}" \
  --nnodes 1 --n-gpus-per-node "${N_GPUS_PER_NODE:-8}" --gateway-count "${GATEWAY_COUNT:-16}" \
  --concurrency "${CONCURRENCY:-64}" \
  --temperature 1.0 --top-p 0.95 --top-k -1 \
  --prompt-length 4096 --response-length 520192 --n "${N:-4}" \
  --gpu-memory-utilization 0.9 --kv-cache-dtype fp8 \
  --apply-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"max","drop_thinking":false}' \
  --log-dir "${OUTPUT_DIR}/${RUN_ID}/logs" \
  --result-path "${OUTPUT_DIR}/${RUN_ID}/results.json" \
  "${LIMIT_ARGS[@]}"
