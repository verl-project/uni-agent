#!/usr/bin/env bash
# Submit one or more OpenClaw SWE-bench rollouts through Ray Jobs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${REPO_ROOT}"

: "${DATA_PATH:?Set DATA_PATH to a prepared SWE-bench parquet path}"
: "${MODEL_PATH:?Set MODEL_PATH to the policy checkpoint}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new output directory outside the checkout}"
: "${RAY_RUNTIME_ENV:?Set RAY_RUNTIME_ENV to a protected Ray runtime-env YAML file}"
: "${RAY_API_SERVER_ADDRESS:?Set RAY_API_SERVER_ADDRESS to the Ray Jobs API address}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RAY_BIN="${RAY_BIN:-ray}"
TASK_CONFIG="${TASK_CONFIG:-examples/blackbox_recipes/openclaw_swe_task/task_config_openclaw.yaml}"
ENGINE="${ENGINE:-vllm}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
LIMIT="${LIMIT:-1}"
N="${N:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-65536}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
RESULT_PATH="${RESULT_PATH:-${OUTPUT_DIR}/result.json}"

[[ "${OUTPUT_DIR}" == /* ]] || { echo "OUTPUT_DIR must be absolute" >&2; exit 2; }
[[ -f "${RAY_RUNTIME_ENV}" ]] || { echo "RAY_RUNTIME_ENV is not a readable file: ${RAY_RUNTIME_ENV}" >&2; exit 2; }
[[ -f "${TASK_CONFIG}" ]] || { echo "TASK_CONFIG is not a readable file: ${TASK_CONFIG}" >&2; exit 2; }
command -v "${PYTHON_BIN}" >/dev/null || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 2; }
command -v "${RAY_BIN}" >/dev/null || { echo "Ray not found: ${RAY_BIN}" >&2; exit 2; }

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
if [[ -e "${RESULT_PATH}" ]] || find "${LOG_DIR}" -type f -print -quit | grep -q .; then
    echo "Output already contains inference artifacts; choose a fresh OUTPUT_DIR or LOG_DIR." >&2
    exit 2
fi

is_enabled() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        0|false|no|off) return 1 ;;
        *) echo "Expected a boolean value, got: $1" >&2; exit 2 ;;
    esac
}

ARGS=(
    --data-path "${DATA_PATH}"
    --model-path "${MODEL_PATH}"
    --task-config "${TASK_CONFIG}"
    --engine "${ENGINE}"
    --tool-parser "${TOOL_PARSER}"
    --response-length "${RESPONSE_LENGTH}"
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
    --nnodes "${NNODES}"
    --n-gpus-per-node "${N_GPUS_PER_NODE}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --limit "${LIMIT}"
    --n "${N}"
    --concurrency "${CONCURRENCY}"
    --gateway-count "${GATEWAY_COUNT}"
    --log-dir "${LOG_DIR}"
    --result-path "${RESULT_PATH}"
)
if is_enabled "${LANGUAGE_MODEL_ONLY}"; then ARGS+=(--language-model-only); fi
if is_enabled "${DISABLE_THINKING}"; then ARGS+=(--disable-thinking); fi

echo "Submitting OpenClaw inference: limit=${LIMIT}, n=${N}, concurrency=${CONCURRENCY}, gateway_count=${GATEWAY_COUNT}"
echo "Output: ${OUTPUT_DIR}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${RAY_BIN}" job submit --address "${RAY_API_SERVER_ADDRESS}" --runtime-env "${RAY_RUNTIME_ENV}" --working-dir . -- "${PYTHON_BIN}" examples/inference/parallel_infer_verl.py "${ARGS[@]}"
    printf '\n'
    exit 0
fi

"${RAY_BIN}" job submit \
    --address "${RAY_API_SERVER_ADDRESS}" \
    --runtime-env "${RAY_RUNTIME_ENV}" \
    --working-dir . \
    -- "${PYTHON_BIN}" examples/inference/parallel_infer_verl.py "${ARGS[@]}"

"${PYTHON_BIN}" - "${RESULT_PATH}" "${LOG_DIR}" "${LIMIT}" "${N}" <<'PY'
import json
import sys
from pathlib import Path

result_path, log_dir = Path(sys.argv[1]), Path(sys.argv[2])
limit, n = int(sys.argv[3]), int(sys.argv[4])
payload = json.loads(result_path.read_text(encoding="utf-8"))
scores = payload.get("scores")
count = int(payload.get("num_scored_sessions", -1))
if not isinstance(scores, list) or count != len(scores) or count < 1:
    raise SystemExit(f"inference result is incomplete: sessions={count}, scores={scores!r}")
if limit > 0 and count != limit * max(1, n):
    raise SystemExit(f"unexpected session count: expected {limit * max(1, n)}, got {count}")
trajectory_paths = sorted(log_dir.rglob("trajectory.json"))
if len(trajectory_paths) != count:
    raise SystemExit(f"expected {count} trajectory files, found {len(trajectory_paths)} under {log_dir}")
for path in trajectory_paths:
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    items = trajectory.get("trajectories", [])
    if trajectory.get("num_trajectories") != 1 or len(items) != 1 or items[0].get("finished") is not True:
        raise SystemExit(f"expected one finished trajectory entry in {path}")
print(f"validated {count} scored session(s) and one finished trajectory per session")
PY
