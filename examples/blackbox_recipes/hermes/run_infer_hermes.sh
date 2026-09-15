#!/usr/bin/env bash
# One-sample Hermes SWE-bench Verified rollout through Uni-Agent's framework.
#
# This script waits for the ray job, writes all run artifacts outside the source
# checkout, and returns the acceptance checker status.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${REPO_ROOT}"

export DEPLOYMENT="${DEPLOYMENT:-openyuanrong}"
if [[ "${DEPLOYMENT}" == "openyuanrong" ]]; then
    export OPENYUANRONG_SERVER_ADDRESS="${OPENYUANRONG_SERVER_ADDRESS:?OPENYUANRONG_SERVER_ADDRESS must be set}"
    export OPENYUANRONG_TOKEN="${OPENYUANRONG_TOKEN:?OPENYUANRONG_TOKEN must be set}"
fi
export OPENYUANRONG_TUNNEL_SSL_VERIFY="${OPENYUANRONG_TUNNEL_SSL_VERIFY:-0}"
export TUNNEL_SSL_VERIFY="${TUNNEL_SSL_VERIFY:-0}"
export OPENYUANRONG_ENV_FILE="${OPENYUANRONG_ENV_FILE:-/home/zxh/.config/uni-agent/openyuanrong.env}"

# Use the project runtime even when the caller has not activated Conda.  The
# system Python on remote186 does not carry datasets/Ray and would fail only
# after creating a half-populated output directory.
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    PYTHON_CANDIDATES=()
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        PYTHON_CANDIDATES+=("${CONDA_PREFIX}/bin/python")
    fi
    PYTHON_CANDIDATES+=(
        "/home/zxh/miniconda3/envs/openclaw-train312-20260907/bin/python"
        "$(command -v python3 || true)"
        "$(command -v python || true)"
    )
    for candidate in "${PYTHON_CANDIDATES[@]}"; do
        if [[ -x "${candidate}" ]] && "${candidate}" -c 'import datasets, ray, yaml' >/dev/null 2>&1; then
            PYTHON_BIN="${candidate}"
            break
        fi
    done
fi
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "No project Python with datasets, Ray and PyYAML was found; set PYTHON_BIN" >&2
    exit 2
fi
RAY_BIN="${RAY_BIN:-$(dirname "${PYTHON_BIN}")/ray}"
if [[ ! -x "${RAY_BIN}" ]]; then
    RAY_BIN="$(command -v ray || true)"
fi
if [[ -z "${RAY_BIN}" || ! -x "${RAY_BIN}" ]]; then
    echo "Ray CLI was not found for ${PYTHON_BIN}; set RAY_BIN" >&2
    exit 2
fi
if [[ "${DEPLOYMENT}" == "openyuanrong" ]] && ! "${PYTHON_BIN}" -c 'import yr_sandbox' >/dev/null 2>&1; then
    echo "${PYTHON_BIN} is missing yr_sandbox; install openyuanrong-sandbox in the project environment" >&2
    exit 2
fi
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"

# FlashInfer's JIT compiler needs the CUDA toolkit headers.  Prefer an explicit
# operator setting; otherwise resolve the installed host toolkit without baking
# this remote host's path into the shared Uni-Agent code.
if [[ -z "${CUDA_HOME:-}" ]]; then
    for candidate in /usr/local/cuda-12.4 /usr/local/cuda; do
        if [[ -f "${candidate}/include/cuda_runtime.h" ]]; then
            export CUDA_HOME="${candidate}"
            break
        fi
    done
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export CUDA_PATH="${CUDA_HOME}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi

DATA_PATH="${DATA_PATH:-/home/zxh/data/swe_agent/swe_bench_verified.parquet}"
MODEL_PATH="${MODEL_PATH:-/home/zxh/models/Qwen3.5-9B}"
TASK_CONFIG="${TASK_CONFIG:-examples/blackbox_recipes/hermes/task_config_hermes.yaml}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
LIMIT="${LIMIT:-1}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
N="${N:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
INSTANCE_ID="${INSTANCE_ID:-}"

if [[ ! -f "${DATA_PATH}" ]]; then
    echo "DATA_PATH does not exist: ${DATA_PATH}" >&2
    exit 2
fi
if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 2
fi

RUN_ID="${RUN_ID:-hermes-$(date +%Y%m%d-%H%M%S)-$$}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zxh/hermesrecipe/artifacts/${RUN_ID}}"
if [[ -e "${OUTPUT_DIR}/result.json" || -e "${OUTPUT_DIR}/acceptance.json" ]]; then
    echo "OUTPUT_DIR already contains a completed run: ${OUTPUT_DIR}" >&2
    exit 2
fi
mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/hermes"
PREPARED_DATA="${OUTPUT_DIR}/prepared_sample.parquet"
SAMPLE_METADATA="${OUTPUT_DIR}/sample_metadata.json"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
RESULT_PATH="${RESULT_PATH:-${OUTPUT_DIR}/result.json}"

PREPARE_ARGS=(
    --input "${DATA_PATH}"
    --output "${PREPARED_DATA}"
    --index "${SAMPLE_INDEX}"
    --metadata-output "${SAMPLE_METADATA}"
)
if [[ -n "${INSTANCE_ID}" ]]; then
    PREPARE_ARGS+=(--instance-id "${INSTANCE_ID}")
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_sample.py" "${PREPARE_ARGS[@]}"

INSTANCE_ID="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["instance_id"])' "${SAMPLE_METADATA}")"
"${PYTHON_BIN}" "${SCRIPT_DIR}/write_run_metadata.py" \
    --output-dir "${OUTPUT_DIR}" \
    --repo-root "${REPO_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --data-path "${DATA_PATH}" \
    --task-config "${TASK_CONFIG}" \
    --sample-metadata "${SAMPLE_METADATA}" \
    --tool-parser "${TOOL_PARSER}" \
    --nnodes "${NNODES}" \
    --n-gpus-per-node "${N_GPUS_PER_NODE}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --limit "${LIMIT}" \
    --n "${N}" \
    --concurrency "${CONCURRENCY}" \
    --gateway-count "${GATEWAY_COUNT}" \
    --cuda-home "${CUDA_HOME:-}"

# The one-shot launcher owns a single-node Ray head only when no external Ray
# Jobs endpoint was supplied. This mirrors the existing recipe launchers while
# avoiding interference with a user-managed cluster.
if [[ -n "${RAY_API_SERVER_ADDRESS:-}" ]]; then
    RAY_JOB_ADDRESS="${RAY_API_SERVER_ADDRESS}"
elif [[ "${RAY_ADDRESS:-}" == http://* || "${RAY_ADDRESS:-}" == https://* ]]; then
    RAY_JOB_ADDRESS="${RAY_ADDRESS}"
else
    RAY_JOB_ADDRESS="http://127.0.0.1:8265"
fi
RAY_STARTED_BY_LAUNCHER=0
if ! curl -fsS --max-time 3 "${RAY_JOB_ADDRESS}/api/version" >/dev/null 2>&1; then
    if [[ "${NNODES}" != "1" ]]; then
        echo "No Ray Jobs endpoint at ${RAY_JOB_ADDRESS}; set RAY_API_SERVER_ADDRESS for NNODES=${NNODES}" >&2
        exit 2
    fi
    echo "Starting local Ray head for the one-node rollout"
    "${RAY_BIN}" start --head --dashboard-host=127.0.0.1 --num-gpus="${N_GPUS_PER_NODE}" --disable-usage-stats
    RAY_STARTED_BY_LAUNCHER=1
    for _ in {1..30}; do
        if curl -fsS --max-time 2 "${RAY_JOB_ADDRESS}/api/version" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    if ! curl -fsS --max-time 3 "${RAY_JOB_ADDRESS}/api/version" >/dev/null 2>&1; then
        echo "Local Ray Jobs endpoint did not become ready: ${RAY_JOB_ADDRESS}" >&2
        exit 2
    fi
fi
RUNTIME_ENV_FILE=""
cleanup_launcher() {
    if [[ -n "${RUNTIME_ENV_FILE}" && -f "${RUNTIME_ENV_FILE}" ]]; then
        rm -f -- "${RUNTIME_ENV_FILE}"
    fi
    if [[ "${RAY_STARTED_BY_LAUNCHER}" == "1" ]]; then
        "${RAY_BIN}" stop >/dev/null 2>&1 || true
    fi
}
trap cleanup_launcher EXIT

# The job driver is created by Ray's Job Agent and does not inherit this shell's
# environment. Credentials are passed only to that job runtime; command.txt
# deliberately contains redacted placeholders.
RUNTIME_ENV_FILE="$(mktemp /tmp/hermes-runtime-env.XXXXXX.json)"
chmod 600 "${RUNTIME_ENV_FILE}"
"${PYTHON_BIN}" -c 'import json,os; keys=("DEPLOYMENT","OPENYUANRONG_TUNNEL_SSL_VERIFY","TUNNEL_SSL_VERIFY","CUDA_HOME","CUDA_PATH","PYTHONPATH","OPENYUANRONG_ENV_FILE"); print(json.dumps({"env_vars": {k: os.environ[k] for k in keys if os.environ.get(k)}}))' > "${RUNTIME_ENV_FILE}"

JOB_COMMAND=(
    "${PYTHON_BIN}" examples/inference/parallel_infer_verl.py
    --data-path "${PREPARED_DATA}"
    --model-path "${MODEL_PATH}"
    --task-config "${TASK_CONFIG}"
    --tool-parser "${TOOL_PARSER}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --nnodes "${NNODES}"
    --n-gpus-per-node "${N_GPUS_PER_NODE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --limit "${LIMIT}"
    --n "${N}"
    --concurrency "${CONCURRENCY}"
    --gateway-count "${GATEWAY_COUNT}"
    --language-model-only
    --artifact-dir "${OUTPUT_DIR}/hermes"
    --log-dir "${LOG_DIR}"
    --result-path "${RESULT_PATH}"
)
if [[ "${DEPLOYMENT}" == "openyuanrong" ]]; then
    JOB_COMMAND=(bash examples/blackbox_recipes/hermes/run_with_openyuanrong_env.sh "${JOB_COMMAND[@]}")
fi

INFER_EXIT=0
"${RAY_BIN}" job submit \
    --address "${RAY_JOB_ADDRESS}" \
    --runtime-env "${RUNTIME_ENV_FILE}" \
    --working-dir . \
    -- "${JOB_COMMAND[@]}" || INFER_EXIT=$?
rm -f -- "${RUNTIME_ENV_FILE}"
RUNTIME_ENV_FILE=""

CHECK_EXIT=0
"${PYTHON_BIN}" "${SCRIPT_DIR}/check_acceptance.py" \
    --output-dir "${OUTPUT_DIR}" \
    --instance-id "${INSTANCE_ID}" || CHECK_EXIT=$?

if [[ "${INFER_EXIT}" -ne 0 ]]; then
    exit "${INFER_EXIT}"
fi
exit "${CHECK_EXIT}"
