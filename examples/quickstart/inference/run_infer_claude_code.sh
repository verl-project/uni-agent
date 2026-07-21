#!/usr/bin/env bash
set -euo pipefail

ray job submit --no-wait \
    --runtime-env examples/quickstart/inference/runtime_env.yaml \
    --working-dir . \
    -- python3 examples/inference/parallel_infer_verl.py \
    --data-path ~/data/swe_agent/swe_bench_verified.parquet \
    --model-path Qwen/Qwen3.6-35B-A3B \
    --task-config examples/quickstart/inference/task_config_claude_code.yaml \
    --tool-parser qwen3_coder \
    --tensor-parallel-size 4 \
    --nnodes 8 \
    --n-gpus-per-node 8 \
    --concurrency 128
