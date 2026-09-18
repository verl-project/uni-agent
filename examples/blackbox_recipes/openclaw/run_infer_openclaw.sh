#!/usr/bin/env bash
# Run OpenClaw through the framework's standard verl inference path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${REPO_ROOT}"

: "${DATA_PATH:?Set a preprocessed SWE-bench parquet path}"
: "${MODEL_PATH:?Set a local model checkpoint path}"
: "${OUTPUT_DIR:?Set a fresh output directory outside the repository}"
: "${OPENYUANRONG_SERVER_ADDRESS:?Set the OpenYuanRong server address}"
: "${OPENYUANRONG_TOKEN:?Set the OpenYuanRong token}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "${PYTHON_BIN}" >/dev/null || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 2; }
TASK_CONFIG="${TASK_CONFIG:-examples/blackbox_recipes/openclaw/config/openclaw_swe_bench.yaml}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
LIMIT="${LIMIT:-1}"
N="${N:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
RESULT_PATH="${RESULT_PATH:-${OUTPUT_DIR}/result.json}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
if [[ -e "${RESULT_PATH}" ]] || find "${LOG_DIR}" -type f \( -name task.log -o -name trajectory.json \) -print -quit | grep -q .; then
    echo "Output already contains inference results; choose a fresh OUTPUT_DIR or LOG_DIR." >&2
    exit 2
fi

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

INFER_CMD=(
    "${PYTHON_BIN}" examples/inference/parallel_infer_verl.py
    --data-path "${DATA_PATH}"
    --model-path "${MODEL_PATH}"
    --task-config "${TASK_CONFIG}"
    --tool-parser "${TOOL_PARSER}"
    --engine vllm
    --language-model-only
    --limit "${LIMIT}"
    --n "${N}"
    --gateway-count "${GATEWAY_COUNT}"
    --concurrency "${CONCURRENCY}"
    --log-dir "${LOG_DIR}"
    --result-path "${RESULT_PATH}"
)
if [[ -n "${NNODES:-}" ]]; then INFER_CMD+=(--nnodes "${NNODES}"); fi
if [[ -n "${N_GPUS_PER_NODE:-}" ]]; then INFER_CMD+=(--n-gpus-per-node "${N_GPUS_PER_NODE}"); fi
if [[ -n "${TENSOR_PARALLEL_SIZE:-}" ]]; then INFER_CMD+=(--tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"); fi
if [[ -n "${GPU_MEMORY_UTILIZATION:-}" ]]; then INFER_CMD+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"); fi

echo "=== OpenClaw inference ==="
echo "Model:       ${MODEL_PATH}"
echo "Data:        ${DATA_PATH}"
echo "Task config: ${TASK_CONFIG}"
echo "Output:      ${OUTPUT_DIR}"
echo "Shape:       limit=${LIMIT}, n=${N}, concurrency=${CONCURRENCY}, gateway_count=${GATEWAY_COUNT}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${INFER_CMD[@]}"
    printf '\n'
    exit 0
fi

"${INFER_CMD[@]}"

"${PYTHON_BIN}" - "${RESULT_PATH}" "${LOG_DIR}" <<'PY'
import json
import sys
from pathlib import Path

result_path, log_dir = Path(sys.argv[1]), Path(sys.argv[2])
result = json.loads(result_path.read_text(encoding="utf-8"))
task_logs = sorted(log_dir.glob("**/task.log"))
trajectories = sorted(log_dir.glob("**/trajectory.json"))
scores = [float(score) for score in result.get("scores", [])]
num_sessions = int(result.get("num_scored_sessions", len(scores)))
if num_sessions < 1 or len(scores) != num_sessions:
    raise SystemExit(f"no complete inference sessions: num_scored_sessions={num_sessions}, scores={len(scores)}")
if len(task_logs) != num_sessions or len(trajectories) != num_sessions:
    raise SystemExit(
        f"expected {num_sessions} task logs and framework trajectories, "
        f"found {len(task_logs)} and {len(trajectories)}"
    )

for path in trajectories:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("trajectories", [])
    if data.get("num_trajectories") != 1 or len(items) != 1:
        raise SystemExit(f"expected one framework trajectory in {path}")
    item = items[0]
    if item.get("finished") is not True:
        raise SystemExit(f"OpenClaw did not finish a valid audited episode: {path}")
print(f"validated {num_sessions} session(s), task logs and finished trajectories")
PY

echo "Inference complete: ${RESULT_PATH}"
