#!/usr/bin/env bash
# Run MemAgent inference against an existing OpenAI-compatible model endpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

DATA_FILE="${DATA_FILE:-${HOME}/data/uni_agent/hotpotqa_dev.parquet}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL="${MODEL:-Qwen3-8B}"
TASK_CONFIG="${TASK_CONFIG:-examples/mem_agent/task_config.yaml}"
CONCURRENCY="${CONCURRENCY:-8}"
ROLLOUT_N="${ROLLOUT_N:-1}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/mem_agent/inference}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"

if [[ ! -f "${DATA_FILE}" ]]; then
    echo "HotpotQA data file not found: ${DATA_FILE}" >&2
    echo "Run uni_agent.tasks.hotpotqa.preprocess first or set DATA_FILE." >&2
    exit 1
fi

inference_args=(
    --data-path "${DATA_FILE}"
    --task-config "${TASK_CONFIG}"
    --base-url "${BASE_URL}"
    --model "${MODEL}"
    --concurrency "${CONCURRENCY}"
    --n "${ROLLOUT_N}"
    --log-dir "${LOG_DIR}"
)

if [[ -n "${API_KEY:-}" ]]; then
    inference_args+=(--api-key "${API_KEY}")
fi
if [[ -n "${LIMIT:-}" ]]; then
    inference_args+=(--limit "${LIMIT}")
fi

python3 examples/inference/parallel_infer_api.py "${inference_args[@]}" "$@"
