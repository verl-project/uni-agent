#!/usr/bin/env bash
set -euo pipefail

ray job submit --no-wait \
    --runtime-env examples/quickstart/harbor/runtime_env.yaml \
    --working-dir . \
    -- python3 examples/inference/parallel_infer_verl.py \
    --data-path ~/data/swe_agent/swe_bench_verified.parquet \
    --model-path Qwen/Qwen3.6-35B-A3B \
    --served-model-name hosted_vllm/Qwen3.6-35B-A3B \
    --task-config examples/quickstart/harbor/task_config_terminus2_rollout.yaml \
    --allowed-request-sampling-param-keys temperature top_p top_k \
    --tool-parser qwen3_coder \
    --tensor-parallel-size 4 \
    --nnodes 4 \
    --n-gpus-per-node 8 \
    --gateway-count 8 \
    --log-dir /mnt/shared/harbor_terminus2 \
    --concurrency 512
