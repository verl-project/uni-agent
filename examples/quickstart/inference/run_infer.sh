#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a checkpoint accessible from the Ray cluster}"

DATA_PATH="${DATA_PATH:-$HOME/data/swe_agent/swe_bench_verified.parquet}"
TASK_CONFIG="${TASK_CONFIG:-examples/quickstart/inference/task_config.yaml}"
RUNTIME_ENV="${RUNTIME_ENV:-examples/quickstart/inference/runtime_env.yaml}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
CONCURRENCY="${CONCURRENCY:-64}"
LIMIT="${LIMIT:-8}"

ray job submit --no-wait \
    --runtime-env "$RUNTIME_ENV" \
    --working-dir . \
    -- python3 examples/inference/parallel_infer_verl.py \
    --data-path "$DATA_PATH" \
    --model-path "$MODEL_PATH" \
    --task-config "$TASK_CONFIG" \
    --tool-parser "$TOOL_PARSER" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --nnodes "$NNODES" \
    --n-gpus-per-node "$N_GPUS_PER_NODE" \
    --concurrency "$CONCURRENCY" \
    --limit "$LIMIT"
