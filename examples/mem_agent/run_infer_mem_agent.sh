#!/usr/bin/env bash
# Run the raw 8k-1M HotpotQA benchmark without thinking/COT on physical GPUs 4-7.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

: "${MODEL_PATH:=./models/Qwen3-4B}"
: "${CONDA_ENV_DIR:=/root/.miniconda3/envs/xxx}"
: "${PYTHON_BIN:=${CONDA_ENV_DIR}/bin/python3}"
: "${VLLM_BIN:=${CONDA_ENV_DIR}/bin/vllm}"
: "${GPU_IDS:=0,1,2,3,4,5,6,7}"
: "${HOST:=127.0.0.1}"
: "${PORT:=8000}"
: "${BASE_URL:=http://${HOST}:${PORT}/v1}"
: "${START_SERVER:=1}"
: "${SERVER_START_TIMEOUT:=600}"
: "${GPU_MEMORY_UTILIZATION:=0.80}"
: "${MAX_MODEL_LEN:=10240}"
: "${MAX_NUM_BATCHED_TOKENS:=${MAX_MODEL_LEN}}"
: "${ENFORCE_EAGER:=0}"
: "${CONCURRENCY:=32}"
: "${CONTEXT_CHUNK_SIZE:=5000}"

TASK_CONFIG="${TASK_CONFIG:-examples/mem_agent/task_config.yaml}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"
DATA_DIR="${DATA_DIR:-./datas/hotpotqa}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/mem_agent/inference_no_thinking_$(basename "${MODEL_PATH}")}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/mem_agent/inference_no_thinking_$(basename "${MODEL_PATH}")}"
LENGTHS_VALUE="${LENGTHS:-8k,16k,32k,64k,128k,256k,512k,1M}"

export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"

for executable in "${PYTHON_BIN}" "${VLLM_BIN}"; do
    if [[ ! -x "${executable}" ]]; then
        echo "Required executable is missing: ${executable}" >&2
        exit 1
    fi
done
for required_path in "${MODEL_PATH}" "${TASK_CONFIG}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 1
    fi
done

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "HotpotQA data directory not found: ${DATA_DIR}" >&2
    exit 1
fi

IFS=',' read -r -a LENGTH_ARRAY <<< "${LENGTHS_VALUE// /,}"
if ((${#LENGTH_ARRAY[@]} == 0)); then
    echo "LENGTHS did not select any benchmark lengths" >&2
    exit 1
fi
for len in "${LENGTH_ARRAY[@]}"; do
    if [[ ! -f "${DATA_DIR}/eval_hotpotqa_${len}.json" ]]; then
        echo "HotpotQA input file not found: ${DATA_DIR}/eval_hotpotqa_${len}.json" >&2
        exit 1
    fi
done

IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
TP_SIZE="${TP_SIZE:-${#GPU_ID_ARRAY[@]}}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Stopping vLLM server (pid=${SERVER_PID})..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

SERVER_ROOT="${BASE_URL%/v1}"
HEALTH_URL="${SERVER_ROOT}/health"
SERVER_LOG="${LOG_DIR}/vllm_server.log"
if [[ "${START_SERVER}" == "1" ]]; then
    if curl --silent --fail --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
        echo "Endpoint is already serving at ${SERVER_ROOT}; use another PORT or set START_SERVER=0." >&2
        exit 1
    fi
    echo "Starting vLLM on physical GPUs ${GPU_IDS} (tensor parallel=${TP_SIZE})..."
    vllm_args=(
        --host "${HOST}" \
        --port "${PORT}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --dtype bfloat16 \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --enable-chunked-prefill \
        --trust-remote-code \
        --default-chat-template-kwargs '{"enable_thinking": false}'
    )
    if [[ "${ENFORCE_EAGER}" == "1" ]]; then
        vllm_args+=(--enforce-eager)
    fi
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${VLLM_BIN}" serve "${MODEL_PATH}" "${vllm_args[@]}" \
        >"${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!

    deadline=$((SECONDS + SERVER_START_TIMEOUT))
    until curl --silent --fail --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "vLLM exited during startup. Last log lines:" >&2
            tail -80 "${SERVER_LOG}" >&2 || true
            exit 1
        fi
        if ((SECONDS >= deadline)); then
            echo "Timed out after ${SERVER_START_TIMEOUT}s waiting for ${HEALTH_URL}" >&2
            tail -80 "${SERVER_LOG}" >&2 || true
            exit 1
        fi
        sleep 2
    done
    echo "vLLM is ready: ${BASE_URL} (model=${SERVED_MODEL_NAME})"
else
    if ! curl --silent --fail --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
        echo "START_SERVER=${START_SERVER}, but no endpoint is healthy at ${HEALTH_URL}" >&2
        exit 1
    fi
    echo "Using existing endpoint: ${BASE_URL} (model=${SERVED_MODEL_NAME})"
fi

for len in "${LENGTH_ARRAY[@]}"; do
    DATA_FILE="${DATA_DIR}/eval_hotpotqa_${len}.json"
    RESULT_FILE="${OUTPUT_DIR}/hotpotqa_${len}.jsonl"
    SUMMARY_FILE="${OUTPUT_DIR}/hotpotqa_${len}_summary.json"

    echo "Running inference on ${DATA_FILE} -> ${RESULT_FILE}"

    inference_args=(
        --data-path "${DATA_FILE}"
        --output-path "${RESULT_FILE}"
        --summary-path "${SUMMARY_FILE}"
        --task-config "${TASK_CONFIG}"
        --tokenizer-path "${MODEL_PATH}"
        --base-url "${BASE_URL}"
        --model "${SERVED_MODEL_NAME}"
        --concurrency "${CONCURRENCY}"
        --chunk-size "${CONTEXT_CHUNK_SIZE}"
    )

    if [[ -n "${API_KEY:-}" ]]; then
        inference_args+=(--api-key "${API_KEY}")
    fi
    if [[ -n "${LIMIT:-}" ]]; then
        inference_args+=(--limit "${LIMIT}")
    fi

    "${PYTHON_BIN}" examples/mem_agent/infer.py "${inference_args[@]}" "$@"
done

echo "All selected HotpotQA runs finished. Results: ${OUTPUT_DIR}"
