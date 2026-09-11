#!/usr/bin/env bash
# Run the OpenClaw recipe through verl's inference driver.
#
# This is the standard PR-level entrypoint for OpenClaw inference.  It starts
# the internal vLLM engine through ``parallel_infer_verl.py``; callers must not
# start a separate ``vllm serve`` process for the same run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${REPO_ROOT}"

: "${CONDA_PREFIX:?请先激活专用 Conda 环境 openclaw-train312-20260907}"
if [[ "$(basename "${CONDA_PREFIX}")" != "openclaw-train312-20260907" ]]; then
    echo "必须使用专用环境 openclaw-train312-20260907，当前为 ${CONDA_PREFIX}" >&2
    exit 2
fi
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "专用环境缺少 Python: ${PYTHON_BIN}" >&2
    exit 2
fi
: "${DATA_PATH:?请设置预处理后的 SWE-bench parquet 路径}"
: "${OUTPUT_DIR:?请设置仓库外的独立输出目录}"

MODEL_PATH="${MODEL_PATH:-/home/zxh/models/Qwen3.5-9B}"
TASK_CONFIG="${TASK_CONFIG:-examples/blackbox_recipes/openclaw/config/openclaw_swe_bench.yaml}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
LIMIT="${LIMIT:-1}"
N="${N:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
RESULT_PATH="${RESULT_PATH:-${OUTPUT_DIR}/result.json}"
TRAJECTORY_DIR="${TRAJECTORY_DIR:-${OUTPUT_DIR}/trajectories}"
RUNTIME_TASK_CONFIG="${OUTPUT_DIR}/task_config.yaml"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${TRAJECTORY_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"
# Ray isolation is opt-in after the host's existing cluster has been inspected.
# For example, an operator may set RAY_ADDRESS=local and RAY_TMPDIR to a path
# under OUTPUT_DIR when the default control plane is confirmed unrelated.
if [[ -n "${RAY_ADDRESS:-}" ]]; then export RAY_ADDRESS; fi
if [[ -n "${RAY_TMPDIR:-}" ]]; then export RAY_TMPDIR; fi
export VERL_DISABLE_PIN_MEMORY="${VERL_DISABLE_PIN_MEMORY:-1}"
export OPENCLAW_DEBUG_ENGINE="${OPENCLAW_DEBUG_ENGINE:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export MALLOC_CONF="${MALLOC_CONF:-background_thread:false}"
export ARROW_DEFAULT_MEMORY_POOL="${ARROW_DEFAULT_MEMORY_POOL:-system}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

echo "=== OpenClaw verl inference ==="
echo "Model:       ${MODEL_PATH}"
echo "Data:        ${DATA_PATH}"
echo "Task config: ${TASK_CONFIG}"
echo "Output:      ${OUTPUT_DIR}"
echo "GPU:         ${CUDA_VISIBLE_DEVICES}"
echo "Shape:       limit=${LIMIT}, n=${N}, concurrency=${CONCURRENCY}, gateway_count=${GATEWAY_COUNT}, TP=8"
echo "================================"

# Keep the checked-in YAML portable.  The artifact path is run-specific, so
# write a derived config next to the run outputs instead of hard-coding it in
# the repository.  The task prompt itself still comes from each data row.
"${PYTHON_BIN}" - "${TASK_CONFIG}" "${RUNTIME_TASK_CONFIG}" "${TRAJECTORY_DIR}" <<'PY'
import sys
from pathlib import Path

import yaml

source, target, artifact_dir = map(Path, sys.argv[1:])
raw = yaml.safe_load(source.read_text(encoding="utf-8"))
entries = raw if isinstance(raw, list) else [raw]
for entry in entries:
    if isinstance(entry, dict) and entry.get("name") == "swe_bench":
        agent = entry.setdefault("agent", {})
        if isinstance(agent, dict):
            agent["artifact_dir"] = str(Path(artifact_dir).resolve())
target.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${PYTHON_BIN}" examples/inference/parallel_infer_verl.py \
        --data-path "${DATA_PATH}" --model-path "${MODEL_PATH}" --task-config "${RUNTIME_TASK_CONFIG}" \
        --tool-parser "${TOOL_PARSER}" --engine vllm --nnodes 1 --n-gpus-per-node 8 \
        --tensor-parallel-size 8 --gpu-memory-utilization 0.2 --max-model-len 8192 \
        --max-num-seqs 1 --max-num-batched-tokens 8192 --enforce-eager --enable-chunked-prefill --language-model-only \
        --free-cache-engine --checkpoint-engine-backend naive --cudagraph-mode NONE --mamba-cache-mode align \
        --enable-cpu-binding --async-scheduling --multi-turn --max-assistant-turns 100 \
        --max-parallel-calls 1 --limit "${LIMIT}" --n "${N}" \
        --gateway-count "${GATEWAY_COUNT}" --concurrency "${CONCURRENCY}" \
        --log-dir "${LOG_DIR}" --result-path "${RESULT_PATH}" --artifact-dir "${TRAJECTORY_DIR}"
    printf '\n'
    exit 0
fi

"${PYTHON_BIN}" examples/inference/parallel_infer_verl.py \
    --data-path "${DATA_PATH}" \
    --model-path "${MODEL_PATH}" \
    --task-config "${RUNTIME_TASK_CONFIG}" \
    --tool-parser "${TOOL_PARSER}" \
    --engine vllm \
    --nnodes 1 \
    --n-gpus-per-node 8 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.2 \
    --max-model-len 8192 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 8192 \
    --enforce-eager \
    --enable-chunked-prefill \
    --language-model-only \
    --free-cache-engine \
    --checkpoint-engine-backend naive \
    --cudagraph-mode NONE \
    --mamba-cache-mode align \
    --enable-cpu-binding \
    --async-scheduling \
    --multi-turn \
    --max-assistant-turns 100 \
    --max-parallel-calls 1 \
    --limit "${LIMIT}" \
    --n "${N}" \
    --gateway-count "${GATEWAY_COUNT}" \
    --concurrency "${CONCURRENCY}" \
    --log-dir "${LOG_DIR}" \
    --result-path "${RESULT_PATH}" \
    --artifact-dir "${TRAJECTORY_DIR}"

"${PYTHON_BIN}" - "${RESULT_PATH}" "${LOG_DIR}" "${TRAJECTORY_DIR}" > "${OUTPUT_DIR}/acceptance.json" <<'PY'
import json
import sys
from pathlib import Path

result_path, log_dir, trajectory_dir = map(Path, sys.argv[1:])
result = json.loads(result_path.read_text(encoding="utf-8"))
task_logs = sorted(log_dir.glob("**/task.log"))
trajectories = sorted(trajectory_dir.glob("*/trajectory.json"))
if len(task_logs) != 1:
    raise SystemExit(f"expected one task.log, found {len(task_logs)}")
if len(trajectories) != 1:
    raise SystemExit(f"expected one trajectory.json, found {len(trajectories)}")
task_text = task_logs[0].read_text(encoding="utf-8", errors="replace")
required = (
    "reward=1.0 acc=1.0 finished=True",
    "resolved=True",
)
missing = [marker for marker in required if marker not in task_text]
if missing:
    raise SystemExit(f"task did not meet acceptance markers: {missing}")
if float(result.get("mean_rm_score", 0.0)) != 1.0:
    raise SystemExit(f"mean_rm_score is not 1.0: {result.get('mean_rm_score')!r}")
summary = {
    "task_logs": [str(path) for path in task_logs],
    "trajectory_files": [str(path) for path in trajectories],
    "num_task_logs": len(task_logs),
    "num_trajectories": len(trajectories),
    "mean_rm_score": result["mean_rm_score"],
    "scores": result.get("scores", []),
    "acceptance": "single_task_single_trajectory_resolved",
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "验收通过：${OUTPUT_DIR}/acceptance.json"
