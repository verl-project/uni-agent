#!/usr/bin/env bash
# Foreground text-only Qwen service; do not launch until this experiment env is ready.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
: "${CONDA_PREFIX:?请激活本实验专用Conda环境}"
: "${OUTPUT_DIR:?请指定仓库外的新运行目录}"
MODEL_PATH="${MODEL_PATH:-/home/zxh/models/Qwen3.5-9B}"
# Native sampler avoids FlashInfer JIT requiring a system CUDA development toolkit.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$OUTPUT_DIR/vllm-cache}"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
case "$OUTPUT_DIR/" in "$ROOT/"*) echo "输出目录必须在仓库外" >&2; exit 2;; esac
mkdir "$OUTPUT_DIR"
python -c 'import importlib.metadata as m; print("uni-agent",m.version("uni-agent")); print("vllm",m.version("vllm"))' > "$OUTPUT_DIR/packages.txt"
printf "VLLM_USE_FLASHINFER_SAMPLER=%s\n" "$VLLM_USE_FLASHINFER_SAMPLER" > "$OUTPUT_DIR/runtime-env.txt"
git rev-parse HEAD > "$OUTPUT_DIR/commit.txt"
git branch --show-current > "$OUTPUT_DIR/branch.txt"
printf "checkout=%s\nconda=%s\nmodel_source=ModelScope:Qwen/Qwen3.5-9B\nmodel_path=%s\nCUDA_VISIBLE_DEVICES=%s\n" "$ROOT" "$CONDA_PREFIX" "$MODEL_PATH" "$CUDA_VISIBLE_DEVICES" > "$OUTPUT_DIR/run.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv > "$OUTPUT_DIR/gpus.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$OUTPUT_DIR/gpu-processes.csv"
CMD=(python -m vllm.entrypoints.openai.api_server --model "$MODEL_PATH" \
  --served-model-name Qwen3.5-9B --host "${HOST:-127.0.0.1}" --port "${PORT:-18090}" \
  --tensor-parallel-size "${TP:-2}" --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --max-num-seqs "${MAX_NUM_SEQS:-1}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --language-model-only --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 --enforce-eager)
printf "%q " "${CMD[@]}" > "$OUTPUT_DIR/command.sh"
printf "\n" >> "$OUTPUT_DIR/command.sh"
exec "${CMD[@]}" > "$OUTPUT_DIR/server.log" 2>&1
