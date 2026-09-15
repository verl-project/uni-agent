# MemAgent Recipe

This recipe trains MemAgent on the 32K HotpotQA training split and evaluates
the base and trained models on eight context lengths from 8K to 1M tokens.
MemAgent reads a long context in token-bounded chunks, starts a fresh model
context for every chunk, carries a compact memory between chunks, and answers
the question from the final memory.

## Result

The reported metric is the macro-average of the eight per-length main scores
(`mean_boxed_answer_token_lcs`):

| Model | 8K-1M macro score |
| --- | ---: |
| Qwen3-4B before training | 53.5 |
| Qwen3-4B after MemAgent training | 58.0 |
| Absolute improvement | +4.5 |

The eight evaluation lengths are `8k`, `16k`, `32k`, `64k`, `128k`,
`256k`, `512k`, and `1M`.

## Requirements

- A working uni-agent/verl environment with PyTorch, Ray, vLLM,
  Transformers, and Hugging Face Hub installed.
- Eight visible GPUs for the default recipe: four trainer GPUs and four
  rollout GPUs. The GPU split can be changed through the environment variables
  documented in `train_mem_agent.sh`.
- Enough local storage for the 32K training parquet and the 8K-1M evaluation
  JSON files.

Run every command below from the repository root:

```bash
cd /path/to/uni-agent

export CONDA_ENV_DIR=/path/to/conda/env
export PYTHON_BIN="${CONDA_ENV_DIR}/bin/python3"
export BASE_MODEL="${PWD}/models/Qwen3-4B"
export DATA_DIR="${PWD}/hotpotqa"
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PWD}/verl:${PYTHONPATH:-}"
```

## 1. Download the model and data

Download Qwen3-4B if it is not already available locally:

```bash
hf download Qwen/Qwen3-4B \
  --local-dir "${BASE_MODEL}"
```

Download the HotpotQA data from
[`BytedTsinghua-SIA/hotpotqa`](https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa):

```bash
hf download BytedTsinghua-SIA/hotpotqa \
  --repo-type dataset \
  --local-dir "${DATA_DIR}"
```

This recipe uses the following files directly; no intermediate preprocessing
step is required:

```text
hotpotqa/hotpotqa_train_32k.parquet
hotpotqa/hotpotqa_dev.parquet
hotpotqa/eval_hotpotqa_8k.json
hotpotqa/eval_hotpotqa_16k.json
hotpotqa/eval_hotpotqa_32k.json
hotpotqa/eval_hotpotqa_64k.json
hotpotqa/eval_hotpotqa_128k.json
hotpotqa/eval_hotpotqa_256k.json
hotpotqa/eval_hotpotqa_512k.json
hotpotqa/eval_hotpotqa_1M.json
```

Verify the required files before starting a long run:

```bash
test -f "${DATA_DIR}/hotpotqa_train_32k.parquet"
test -f "${DATA_DIR}/hotpotqa_dev.parquet"
for length in 8k 16k 32k 64k 128k 256k 512k 1M; do
  test -f "${DATA_DIR}/eval_hotpotqa_${length}.json"
done
```

`train_mem_agent.sh` expects the downloaded directory at `./hotpotqa` by
default. If the data is stored elsewhere, update `TRAIN_FILE` and `VAL_FILE`
at the top of the script before launching the Ray job.

## 2. Train MemAgent

The canonical training entry point is:

```bash
export EXPERIMENT_NAME=mem_agent_qwen3_4b_hotpotqa_32k

CONDA_ENV_DIR="${CONDA_ENV_DIR}" \
MODEL_PATH="${BASE_MODEL}" \
GPU_IDS=0,1,2,3,4,5,6,7 \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
bash ./examples/mem_agent/train_mem_agent.sh
```

The script launches verl v1 GRPO with `separate_async` resource pools:

- four GPUs for FSDP2 actor training;
- four GPUs for vLLM rollout with tensor parallelism 4;
- 5,000-token context chunks;
- Qwen thinking mode disabled with
  `data.apply_chat_template_kwargs.enable_thinking=false`.

The Ray job is submitted with `--no-wait`. Use the submission ID printed by
the launch command to follow it:

```bash
ray job list
ray job logs <raysubmit_id> --follow
```

Checkpoints are written under:

```text
checkpoints/mem_agent/<experiment_name>/global_step_<N>/actor/
```

## 3. Merge a training checkpoint

verl saves the actor as an FSDP checkpoint. Merge the step selected for
evaluation into a Hugging Face directory before starting vLLM:

```bash
export STEP=30
export CHECKPOINT_ROOT="${PWD}/checkpoints/mem_agent/${EXPERIMENT_NAME}"
export TRAINED_MODEL="${CHECKPOINT_ROOT}/global_step_${STEP}_hf"

"${PYTHON_BIN}" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${CHECKPOINT_ROOT}/global_step_${STEP}/actor" \
  --target_dir "${TRAINED_MODEL}"
```

Choose `STEP` based on the saved checkpoints and validation curve; step 30 is
shown only as an example.

## 4. Evaluate the untrained model

`run_infer_mem_agent.sh` starts vLLM, runs all eight HotpotQA lengths, writes
one JSONL record per sample, and writes a summary JSON for each length:

```bash
export BASE_RESULTS="${PWD}/outputs/mem_agent/qwen3_4b_base_8k_1m"

CONDA_ENV_DIR="${CONDA_ENV_DIR}" \
MODEL_PATH="${BASE_MODEL}" \
DATA_DIR="${DATA_DIR}" \
GPU_IDS=0,1,2,3,4,5,6,7 \
OUTPUT_DIR="${BASE_RESULTS}" \
LOG_DIR="${PWD}/logs/mem_agent/qwen3_4b_base_8k_1m" \
ENFORCE_EAGER=1 \
bash ./examples/mem_agent/run_infer_mem_agent.sh
```

The inference server also sets `enable_thinking=false`. Existing successful
JSONL rows are reused, so an interrupted long benchmark can be resumed by
running the same command again.

## 5. Evaluate the trained model

Run the identical benchmark with the merged checkpoint:

```bash
export TRAINED_RESULTS="${PWD}/outputs/mem_agent/qwen3_4b_memagent_step_${STEP}_8k_1m"

CONDA_ENV_DIR="${CONDA_ENV_DIR}" \
MODEL_PATH="${TRAINED_MODEL}" \
DATA_DIR="${DATA_DIR}" \
GPU_IDS=0,1,2,3,4,5,6,7 \
OUTPUT_DIR="${TRAINED_RESULTS}" \
LOG_DIR="${PWD}/logs/mem_agent/qwen3_4b_memagent_step_${STEP}_8k_1m" \
ENFORCE_EAGER=1 \
bash ./examples/mem_agent/run_infer_mem_agent.sh
```

Each result directory contains:

```text
hotpotqa_<length>.jsonl
hotpotqa_<length>_summary.json
```

The main score is the mean token-level LCS reward of the final boxed answers.
For each sample, the evaluator extracts the last `\boxed{...}` answer and
compares it with the accepted answers. `Answer Contained` is reported as a
separate diagnostic and is not used for the macro result above.


## Implementation map

- `uni_agent/agents/mem_agent/`: MemAgent policy and context management.
- `uni_agent/tasks/hotpotqa/`: HotpotQA preprocessing helpers, Task, and reward.
- `examples/mem_agent/dataset.py`: raw training-parquet adapter and runtime Task
  payload construction.
- `examples/mem_agent/infer.py`: raw 8K-1M JSON adapter, concurrent inference,
  resume handling, and summary generation.
- `examples/mem_agent/task_config.yaml`: task and agent defaults.
- `examples/mem_agent/train_mem_agent.sh`: verl v1 FSDP2/GRPO training entry
  point.
- `examples/mem_agent/run_infer_mem_agent.sh`: self-contained vLLM inference
  and 8K-1M evaluation entry point.
