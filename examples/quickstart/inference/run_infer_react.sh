#!/usr/bin/env bash
set -euo pipefail

# external api mode using doubao-seed-2-1

python3 examples/inference/parallel_infer_api.py \
    --data-path ~/data/swe_agent/swe_bench_verified.parquet \
    --task-config examples/quickstart/inference/task_config_react.yaml \
    --base-url https://ark.cn-beijing.volces.com/api/v3 \
    --model doubao-seed-2-1-pro-260628 \
    --api-key xxxxxxxxxx \
    --log-dir /mnt/shared/uni_agent_logs \
    --concurrency 64 \
    --limit 4

# verl rollout mode

ray job submit --no-wait \
    --runtime-env examples/quickstart/inference/runtime_env.yaml \
    --working-dir . \
    -- python3 examples/inference/parallel_infer_verl.py \
    --data-path ~/data/swe_agent/swe_bench_verified.parquet \
    --model-path Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --task-config examples/quickstart/inference/task_config_react.yaml \
    --tool-parser qwen3_coder \
    --tensor-parallel-size 4 \
    --nnodes 8 \
    --n-gpus-per-node 8 \
    --log-dir /mnt/shared/uni_agent_logs \
    --concurrency 512
