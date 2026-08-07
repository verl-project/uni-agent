ray job submit --no-wait \
    --runtime-env examples/inference/runtime_env.yaml \
    --working-dir . \
    -- python3 examples/inference/parallel_infer_verl.py \
    --data-path /home/tiger/data/swe_agent/swe_bench_verified.parquet \
    --model-path $RAY_DATA_HOME/models/Qwen3.6-35B-A3B --tool-parser qwen3_coder \
    --task-config examples/inference/task_config_claude_code.yaml \
    --tensor-parallel-size 4 --nnodes 8 --n-gpus-per-node 8 --concurrency 512 --n 1
