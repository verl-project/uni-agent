ray job submit --no-wait \
    --runtime-env $RAY_DATA_HOME/data/swe_agent/runtime_env.yaml \
    --working-dir . \
    -- python3 examples/agent_interaction/parallel_infer.py \
    --data-path $RAY_DATA_HOME/data/swe_agent/swe_bench_verified.parquet \
    --model-path $RAY_DATA_HOME/models/Qwen3-Coder-30B-A3B-Instruct \
    --agent-config-path examples/agent_interaction/agent_config.yaml \
    --nnodes 1 \
