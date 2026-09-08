#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
: "${CONDA_PREFIX:?请先激活本实验专用 Conda 环境}"
: "${DATA_PATH:?请设置预处理后的任务数据路径}"
: "${BASE_URL:?请设置 sandbox 可访问的模型 endpoint}"
: "${OUTPUT_DIR:?请设置仓库外的独立输出目录}"
mkdir -p "$OUTPUT_DIR"
exec python examples/inference/parallel_infer_api.py \
  --data-path "$DATA_PATH" \
  --task-config "${TASK_CONFIG:-examples/blackbox_recipes/openclaw/config/openclaw_terminal_bench.yaml}" \
  --base-url "$BASE_URL" --model "${MODEL:-Qwen3.5-9B}" \
  --limit "${LIMIT:-1}" --n 1 --concurrency "${CONCURRENCY:-1}" \
  --log-dir "$OUTPUT_DIR/logs" --result-path "$OUTPUT_DIR/results.json" "$@"
