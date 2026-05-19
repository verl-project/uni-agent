# Train a Search Agent

This page walks through the end-to-end search agent example under `examples/search_agent`. The agent is trained on the [ASearcher](https://github.com/inclusionAI/ASearcher) dataset and learns to answer open-domain questions by calling a self-hosted LocalWiki retrieval service.

The example uses two tools:

- `search`: query Wikipedia passages through LocalWiki, or use its `crawl` command to fetch full passages for selected Wikipedia URLs.
- `finish`: submit the final answer for reward computation.

Training uses the same fully asynchronous `verl` stack described in the agent training guide. The difference is the task: instead of editing code in a sandbox, the agent repeatedly searches, reads, reasons, and submits an answer.

---

## Workflow

The full workflow has three steps:

1. Preprocess ASearcher data into Uni-Agent Parquet format.
2. Start the LocalWiki retrieval service.
3. Submit the fully async training job.

The wrapper script `examples/search_agent/run_localwiki_and_train.sh` handles steps 2 and 3 together.

---

## Step 1: Prepare the Dataset

Use `examples/data_preprocess/asearcher.py` to convert raw ASearcher JSON or JSONL data into Parquet files:

```bash
python examples/data_preprocess/asearcher.py \
    --input_json /path/to/asearcher.jsonl \
    --local_save_dir ~/uni_agent_data/data/asearcher_uni_processed \
    --train_rows 8192 \
    --test_rows 100
```

This writes:

- `~/uni_agent_data/data/asearcher_uni_processed/train.parquet`
- `~/uni_agent_data/data/asearcher_uni_processed/test.parquet`

Each row contains:

- `prompt`: the system and user messages.
- `agent_name`: set to `search_agent`, matching `examples/search_agent/agent_config.yaml`.
- `extra_info.tools_kwargs.reward`: the ground-truth answer used by the search reward.

---

## Step 2: Prepare LocalWiki

The `search` and `crawl` tools call a LocalWiki HTTP service backed by a FAISS index and BGE-M3 embeddings. The service provides:

- `/retrieve`: semantic search over Wikipedia passages.
- `/crawl`: full-passage lookup by Wikipedia URL.

Follow `uni_agent/tools/search/localwiki/README.md` to prepare the retrieval artifacts. The recommended path is to download the prebuilt FAISS index and corpus:

```bash
export DATA_ROOT=${HOME}/uni_agent_data

hf download begunner/wikipedia-2024-06-bge-m3-faiss-ivf \
    --repo-type dataset \
    --local-dir "$DATA_ROOT/wiki24"

cd "$DATA_ROOT/wiki24"
cat wiki24_faiss.index.part?? > wiki24_faiss.index
mv "$DATA_ROOT/wiki24/preprocessed" "$DATA_ROOT/wiki24/wiki24_preprocessed"
```

You also need the retrieval model:

```bash
hf download BAAI/bge-m3 --local-dir "$DATA_ROOT/model/bge-m3"
```

The wrapper script starts LocalWiki for you, so you do not need to start the server manually for training.

---

## Step 3: Run Training

Start from the repository root with a running Ray cluster. Set `DATA_ROOT` to the directory that contains the processed ASearcher data, model checkpoint, and LocalWiki artifacts:

```bash
DATA_ROOT=~/uni_agent_data \
bash examples/search_agent/run_localwiki_and_train.sh
```

The expected layout is:

```text
${DATA_ROOT}/
├── data/asearcher_uni_processed/
│   ├── train.parquet
│   └── test.parquet
├── model/
│   ├── Qwen3-30B-A3B-Thinking-2507/
│   └── bge-m3/
└── wiki24/
    ├── wiki24_faiss.index
    ├── wiki24_data.jsonl
    └── wiki24_preprocessed/
```

The wrapper does the following:

1. Starts `uni_agent/tools/search/localwiki/run_localwiki.sh`.
2. Waits for `http://127.0.0.1:8001/docs` to become reachable.
3. Patches `examples/search_agent/agent_config.yaml` with the Ray head IP.
4. Submits `examples/search_agent/train_fully_async_128K.sh` with `ray job submit`.
5. Keeps the LocalWiki process alive for the training job.

LocalWiki logs are written under `${LOG_DIR:-logs}/localwiki_<timestamp>.log`.

---

## Key Files

- `examples/search_agent/agent_config.yaml`: agent loop config. It uses host deployment, the `hermes` tool parser, `search` and `finish` tools, and the `search` reward.
- `examples/search_agent/runtime_env.yaml`: Ray runtime env for packaging Uni-Agent, `verl`, Python dependencies, and environment variables.
- `examples/search_agent/train_fully_async_128K.sh`: fully async GRPO training script with a 128K response budget.
- `examples/search_agent/run_localwiki_and_train.sh`: wrapper that starts LocalWiki and submits the training job.

The training script automatically resolves the Ray head IP and writes a temporary agent config where:

```yaml
RETRIEVAL_SERVICE_URL: "http://${RAY_HEAD_IP}:8001/retrieve"
CRAWL_SERVICE_URL: "http://${RAY_HEAD_IP}:8001/crawl"
```

---

## Useful Overrides

Common environment variables:

- `DATA_ROOT`: root directory for data, model checkpoints, and LocalWiki artifacts.
- `LOCALWIKI_PORT`: LocalWiki service port, default `8001`.
- `LOCALWIKI_READY_TIMEOUT`: how long the wrapper waits for LocalWiki startup, default `300` seconds.
- `LOG_DIR`: where LocalWiki logs are written.
- `NNODES_ROLLOUT`, `NNODES_TRAIN`, `NGPUS_PER_NODE`: Ray cluster shape for fully async training.

Common training settings in `train_fully_async_128K.sh`:

- `rollout_n`: number of rollouts per prompt.
- `max_prompt_length`, `max_response_length`: context budget.
- `actor_rollout_ref.rollout.agent.num_workers`: number of agent rollout workers.
- `staleness_threshold`, `trigger_parameter_sync_step`, `require_batches`, `partial_rollout`: fully async scheduling behavior.

---

## Inference Only

If you only want to run rollouts without training, start LocalWiki first:

```bash
DATA_ROOT=~/uni_agent_data \
bash uni_agent/tools/search/localwiki/run_localwiki.sh
```

Then run parallel inference with the search agent config:

```bash
python examples/agent_interaction/parallel_infer.py \
    --data-path ~/uni_agent_data/data/asearcher_uni_processed/test.parquet \
    --model-path ~/uni_agent_data/model/Qwen3-30B-A3B-Thinking-2507 \
    --agent-config-path examples/search_agent/agent_config.yaml \
    --engine vllm \
    --tensor-parallel-size 4 \
    --num-workers 8 \
    --max-turns 64 \
    --max-samples 4
```

Make sure `RETRIEVAL_SERVICE_URL` and `CRAWL_SERVICE_URL` point to the LocalWiki server reachable by the rollout workers.

---

## Output

During training, metrics such as reward, response length, tool-call counts, and validation generations are logged under the `search_agent` project. The reward is computed by `uni_agent/reward/search.py`, which extracts the submitted `finish` answer and compares it against the ASearcher ground truth.

