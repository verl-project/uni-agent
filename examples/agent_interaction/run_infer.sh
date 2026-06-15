# swe-bench verified
ray job submit --no-wait \
    --runtime-env $RAY_DATA_HOME/data/swe_agent/runtime_env.yaml \
    --working-dir . \
    -- python3 examples/agent_interaction/parallel_infer.py \
    --data-path $RAY_DATA_HOME/data/swe_agent/swe_bench_verified_modal.parquet \
    --model-path $RAY_DATA_HOME/models/Qwen3-Coder-30B-A3B-Instruct \
    --agent-config-path examples/agent_interaction/agent_config_modal.yaml \
    --nnodes 8 --response-length 131072 --temperature 1.0 --top-p 0.95 --n 16 \


# terminal bench v2
ray job submit --no-wait \
    --runtime-env $RAY_DATA_HOME/data/swe_agent/runtime_env.yaml \
    --working-dir . \
    -- python3 examples/agent_interaction/parallel_infer.py \
    --data-path $RAY_DATA_HOME/data/swe_agent/terminal_bench_v2_modal.parquet \
    --agent-config-path examples/agent_interaction/agent_config_terminal_bench.yaml \
    --model-path $RAY_DATA_HOME/models/Qwen3.6-35B-A3B --tp 8 \
    --prompt-length 8192 \
    --response-length 204800 \
    --temperature 1.0 \
    --top-p 0.95 \
    --n 1 \
    --num-workers 8 \
    --nnodes 1
