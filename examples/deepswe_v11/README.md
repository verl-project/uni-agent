# DeepSWE v1.1 Evaluation

## Setup

```bash
git clone https://github.com/verl-project/uni-agent.git
cd uni-agent
git submodule update --init --recursive
pip install --no-deps -e ./verl

export E2B_API_KEY="replace-with-your-e2b-api-key"

git clone --depth 1 https://github.com/datacurve-ai/deep-swe.git /path/to/deep-swe
python examples/deepswe_v11/prepare_data.py \
  --task-root /path/to/deep-swe/tasks \
  --output /path/to/data/deepswe_v11.parquet
```

## Oracle verification

```bash
python examples/inference/parallel_infer_api.py \
  --data-path /path/to/data/deepswe_v11.parquet \
  --task-config examples/deepswe_v11/task_config_oracle.yaml \
  --base-url http://unused.invalid/v1 \
  --limit 1 \
  --concurrency 1
```

## Evaluation

```bash
DATA_PATH=/path/to/data/deepswe_v11.parquet
MODEL_PATH=/path/to/models/DeepSeek-V4-Flash-0731
TASK_CONFIG=examples/deepswe_v11/task_config.yaml
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR=/path/to/logs/deepswe_v11_dsv4f0731

python examples/inference/parallel_infer_verl.py \
  --data-path "$DATA_PATH" \
  --model-path "$MODEL_PATH" \
  --served-model-name DeepSeek-V4-Flash-0731 \
  --task-config "$TASK_CONFIG" \
  --tool-parser deepseek_v4 \
  --tensor-parallel-size 8 \
  --nnodes 8 --n-gpus-per-node 8 --gateway-count 16 \
  --concurrency 128 \
  --temperature 1.0 --top-p 0.95 --response-length 520192 --n 4 \
  --kv-cache-dtype fp8 \
  --apply-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"max","drop_thinking":false}' \
  --log-dir "$OUTPUT_DIR/$RUN_ID" \
  --result-path "$OUTPUT_DIR/$RUN_ID.json"
```
