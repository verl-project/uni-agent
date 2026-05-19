# SWE-bench Verified Inference Example

This directory contains an example of how to run inference (rollouts) using the **Uni-Agent** agent loop on the [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) dataset.

## Workflow Overview

The inference process consists of two main steps:
1. **Data Preparation**: Fetch the SWE-bench Verified dataset from HuggingFace and convert it into a standardized Parquet format that the Uni-Agent framework can consume.
2. **Agent Inference**: Use the asynchronous agent loop manager to load the model, launch agent environment workers, generate responses, and collect evaluation scores.

---

## Step 1: Prepare the Dataset

Use `examples/data_preprocess/swe_bench_verified.py` to download and process the dataset. This script formats the inputs so that each instance contains the necessary `tools_kwargs` (like `instance_id`) needed to boot up the SWE-agent environments.

```bash
# From repo root. By default, it saves the data to ~/data/swe_agent/swe_bench_verified.parquet
python examples/data_preprocess/swe_bench_verified.py --local-save-dir ~/data/swe_agent
```

---

## Step 2: Run Inference

Use the `parallel_infer.py` script to start the inference process. It can run **locally** (starts a Ray cluster in-process) or be **submitted to an existing Ray cluster** with `ray job submit`.

### 2a. Local run (single machine)

```bash
# From repo root
python examples/agent_interaction/parallel_infer.py \
    --data-path ~/data/swe_agent/swe_bench_verified.parquet \
    --model-path /path/to/your/local/model \
    --engine vllm \
    --tensor-parallel-size 4 \
    --num-workers 8 \
    --max-turns 100 \
    --max-samples 4
```

### 2b. Submit to existing Ray cluster (`ray job submit`)

Use Ray’s `--working-dir` so the job runs from the repo root (config paths are relative). No need to pass `--working-dir` to the Python script.

```bash
export WORKING_DIR=/path/to/agent-rl   # project root
export RUNTIME_ENV=/path/to/runtime_env.yaml   # optional

ray job submit --no-wait --runtime-env "$RUNTIME_ENV" \
  --working-dir "${WORKING_DIR}" \
  -- python3 examples/agent_interaction/parallel_infer.py \
  --data-path /path/to/swe_bench_verified.parquet \
  --model-path /path/to/model \
  --engine vllm \
  --tensor-parallel-size 4 \
  --num-workers 8 \
  --max-turns 100 \
  --max-samples 0
```

- Omit `--runtime-env` if you don’t need it (or pass a JSON string). `--max-samples 0` = no limit.

### Key Parameters

- `--data-path`: Path to the prepared Parquet dataset.
- `--model-path`: Path to the local LLM weights (e.g., Qwen3-Coder).
- `--agent-config-path`: Path to the YAML file configuring the agent loop rules and environment tools.
- `--engine`: The inference backend to use (`vllm` or `sglang`).
- `--num-workers`: Number of parallel agent environments to spin up.
- `--tp, --tensor-parallel-size`: The number of GPUs to partition the model across for inference.
- `--max-turns`: The maximum number of interaction turns an agent is allowed to perform in a single episode.
- `--max-samples`: Cap number of samples (default 4). Use `0` for no limit (full dataset).
- `--runtime-env`: (Optional) Path to a Ray runtime environment YAML file to inject specific environment variables (like AWS keys or Docker configs).

For a complete list of parameters, run:
```bash
python examples/agent_interaction/parallel_infer.py --help
```

## Output
Once the process finishes, the script will output the **Mean RM Score** (Reward Model Score), which typically indicates the pass rate or success rate of the agent solving the GitHub issues in the dataset.
