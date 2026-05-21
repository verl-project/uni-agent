# ray job submit --no-wait \
#     --runtime-env $RAY_DATA_HOME/data/swe_agent/runtime_env.yaml \
#     --working-dir . \
#     -- python3 examples/agent_interaction/parallel_infer.py \
#     --data-path $RAY_DATA_HOME/data/swe_agent/swe_bench_verified.parquet \
#     --model-path $RAY_DATA_HOME/models/Qwen3-Coder-30B-A3B-Instruct \
#     --agent-config-path examples/agent_interaction/agent_config.yaml \
#     --nnodes 1 \


DEBUG_MODE=1 DEPLOYMENT=modal python examples/agent_interaction/parallel_infer.py \
    --data-path ~/data/terminal_bench/fix_git_only.parquet \
    --agent-config-path examples/agent_interaction/agent_config_terminal_bench.yaml \
    --model-path /mnt/hdfs/yyding/models/Qwen3.6-35B-A3B --tp 8 \
    --prompt-length 4096 \
    --response-length 131072 \
    --temperature 0.6 \
    --top-p 0.95 \
    --n 1 \
    --max-samples 1 \
    --num-workers 1
