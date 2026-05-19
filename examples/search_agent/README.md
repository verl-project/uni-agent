# Search Agent Training Example

This directory contains an end-to-end example of training a **search agent** with the **Uni-Agent** framework on the [ASearcher](https://github.com/inclusionAI/ASearcher) dataset, backed by a self-hosted LocalWiki retrieval service.

The agent learns to answer open-domain questions by iteratively calling two tools:
- `search`: query Wikipedia for relevant passages
- `crawl`: fetch full passages from specific Wikipedia URLs

and finally returns its answer via `finish`.

## Workflow Overview

The training process consists of three main steps:
1. **Dataset Preparation**: Convert the ASearcher JSON/JSONL data into the standardized Parquet format consumed by Uni-Agent.
2. **Retrieval Service Setup**: Build / download the LocalWiki FAISS index and bring up the retrieval HTTP server that the `search` tool calls during rollout.
3. **Agent Training**: Submit the fully-async RL training job, which spins up agent workers, generates rollouts against the retrieval service, computes rewards, and updates the policy.

---

## Step 1: Prepare the Dataset

Use `examples/data_preprocess/asearcher.py` to preprocess the raw ASearcher data. This script wraps each question into a chat prompt (with a system prompt instructing the model to use `search` / `crawl` / `finish`) and places the ground-truth answer into `tools_kwargs["reward"]` for the search reward to consume.

```bash
# From repo root
python examples/data_preprocess/asearcher.py \
    --input_json /path/to/asearcher.jsonl \
    --local_save_dir ~/data/asearcher_uni_processed \
    --train_rows 8192 \
    --test_rows 100
```

This produces `train.parquet` and `test.parquet` under `--local_save_dir`.

---

## Step 2: Set Up the LocalWiki Retrieval Service

The `search` / `crawl` tools call a LocalWiki HTTP service (FAISS IVF index + BGE-M3 embeddings) that simulates a search engine over Wikipedia.

Follow `uni_agent/tools/search/localwiki/README.md` to:
1. Install dependencies (`requirements_lwiki.txt`).
2. Either download the prebuilt index from ModelScope (`int040728/wiki24`) or build it from scratch with `scripts/ivf.py`.
3. Preprocess the corpus into `corpus.pkl` / `url_to_ids.pkl`.
4. Download the embedding model (`BAAI/bge-m3`).

You don't need to start the server manually here — Step 3's wrapper script handles that.

---

## Step 3: Run Training

Use `run_localwiki_and_train.sh` to launch the retrieval service on the Ray head node and then submit the training job. The retrieval process is kept alive for the entire training run.

```bash
# From repo root (Ray cluster must already be running)
DATA_ROOT=/path/to/data_root \
    bash examples/search_agent/run_localwiki_and_train.sh
```

`DATA_ROOT` is expected to contain:
- `data/asearcher_uni_processed/{train,test}.parquet` from Step 1
- `model/<model_name>` (default: `Qwen3-30B-A3B-Thinking-2507`)
- the LocalWiki index / corpus / encoder paths referenced by `run_localwiki.sh`

The wrapper:
1. Starts `uni_agent/tools/search/localwiki/run_localwiki.sh` in the background and waits for `http://127.0.0.1:${LOCALWIKI_PORT}/docs` to become reachable (default port `8001`, default timeout 300s).
2. Submits `train_fully_async_128K.sh` to the Ray cluster via `ray job submit`.
3. Keeps the LocalWiki process alive until you `Ctrl+C`.

Logs are written under `${LOG_DIR:-logs}/localwiki_<timestamp>.log`.

### Key Configuration

- `examples/search_agent/agent_config.yaml`: agent loop config — host deployment, `hermes` tool parser, `search` + `finish` tools, `search` reward.
- `examples/search_agent/runtime_env.yaml`: Ray runtime env (working dir, py modules, pip deps, env vars).
- `examples/search_agent/train_fully_async_128K.sh`: fully-async GRPO training script (4 rollout nodes + 4 train nodes by default, 128K context). Override via `NNODES_ROLLOUT`, `NNODES_TRAIN`, `NGPUS_PER_NODE`, etc.

The training script automatically resolves the Ray head IP and patches the agent config so the `search` tool talks to the LocalWiki service via `http://${RAY_HEAD_IP}:8001`.

---

## Inference Only

If you just want to run rollouts without training, point `parallel_infer.py` at the same dataset / agent config:

```bash
python examples/agent_interaction/parallel_infer.py \
    --data-path ~/data/asearcher_uni_processed/test.parquet \
    --model-path /path/to/your/model \
    --agent-config-path examples/search_agent/agent_config.yaml \
    --engine vllm \
    --tensor-parallel-size 4 \
    --num-workers 8 \
    --max-turns 64 \
    --max-samples 4
```

Make sure the LocalWiki service is up and `RAY_HEAD_IP` (or the env vars in `agent_config.yaml`) point to it.

---

## Output

During training, metrics (reward, response length, tool-call stats, etc.) are logged to wandb under project `search_agent`. The reward is the answer-correctness score computed by `uni_agent/reward/search.py` against the ASearcher ground truth.
