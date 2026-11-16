# DeepEyes Recipe

This recipe trains a Qwen multimodal policy with GRPO to answer visual
questions. The policy can call `ImageZoomInTool` to crop an image before
returning its final answer, and an OpenAI-compatible Judge provides the
semantic accuracy reward.

## Result

The Qwen3.5-4B experiment uses the fixed validation split generated
by the preprocessing command below. Accuracy is the Judge's binary semantic
correctness score.

| Model | Deepeyes Validation accuracy (%) |
| --- | ---: |
| Qwen3.5-4B before training | 50.0 |
| Qwen3.5-4B after DeepEyes training | 79.2 |
| Absolute improvement | +29.2 |

These numbers are results on the local 48-sample split, not a score on the
full DeepEyes benchmark.

## Requirements

- A working Uni-Agent/verl environment with PyTorch, Ray, vLLM,
  `qwen-vl-utils`, Pillow, and the OpenAI Python client.
- A Qwen multimodal policy checkpoint.
- DeepEyes train and validation parquet files.
- An OpenAI-compatible Judge that can decide whether a prediction is
  semantically equivalent to the reference answer.

Run commands from the repository root. The reported 4B result used a 7+1
NPU setup (seven devices for policy training and rollout, plus one for the
Judge); that environment-specific launcher is not part of this repository.

## 1. Prepare the data

Download the official
[`ChenShawn/DeepEyes-Datasets-47k`](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k)
visual-toolbox parquet and create the train/validation split:

```bash
python -m uni_agent.tasks.deepeyes.preprocess \
  --local-save-dir /path/to/deepeyes-data
```

This writes `train.parquet`, `val.parquet`, and `manifest.json`. The default
validation size is 48. To use an existing source parquet without downloading
it again:

```bash
python -m uni_agent.tasks.deepeyes.preprocess \
  --local-save-dir /path/to/deepeyes-data \
  --source-file /path/to/data_0.1.2_visual_toolbox_v2.parquet
```

The manifest records the selected source rows and dataset indices so the split
can be audited.

## 2. Start the Judge

DeepEyes uses an OpenAI-compatible Judge to score semantic answer correctness.
For an Ascend NPU setup, start a Qwen3.5-4B Judge on a device reserved from
policy training:

```bash
JUDGE_MODEL=/path/to/Qwen3.5-4B
JUDGE_MODEL_NAME=Qwen3.5-4B
JUDGE_HOST=127.0.0.1
JUDGE_PORT=18901

ASCEND_RT_VISIBLE_DEVICES=7 \
vllm serve "${JUDGE_MODEL}" \
  --served-model-name "${JUDGE_MODEL_NAME}" \
  --host "${JUDGE_HOST}" \
  --port "${JUDGE_PORT}" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.75 \
  --enforce-eager \
  --trust-remote-code
```

Keep the Judge process running in a separate terminal. Before training, verify
that it is ready and exposes the configured model name:

```bash
curl --fail --silent http://127.0.0.1:18901/v1/models
```

For a remote Judge or a non-Ascend backend, start an equivalent
OpenAI-compatible service and substitute its endpoint and served model name in
the training command below.

## 3. Train

Use `train_deepeyes.sh` with the Judge endpoint:

```bash
MODEL_PATH=/path/to/qwen-multimodal-policy \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/val.parquet \
LLM_AS_A_JUDGE_BASE=http://127.0.0.1:18901/v1 \
LLM_AS_A_JUDGE_MODEL=Qwen3.5-4B \
bash examples/deepeyes/train_deepeyes.sh
```

For a seven-device policy/rollout setup, configure the generic launcher with
environment variables. The Judge command above reserves device 7, while the
training command uses devices 0-6:

```bash
MODEL_PATH=/path/to/Qwen3.5-4B \
TRAIN_FILE=/path/to/deepeyes-data/train.parquet \
VAL_FILE=/path/to/deepeyes-data/val.parquet \
LLM_AS_A_JUDGE_BASE=http://127.0.0.1:18901/v1 \
LLM_AS_A_JUDGE_MODEL=Qwen3.5-4B \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
NDEVICES_PER_NODE=7 \
GATEWAY_COUNT=7 \
TRAIN_BATCH_SIZE=7 \
PPO_MINI_BATCH_SIZE=7 \
ROLLOUT_N=4 \
bash examples/deepeyes/train_deepeyes.sh
```

## Reward

The reward is:

```text
reward = 0.8 * accuracy + 0.2 * format + 1.2 * tool
```

`accuracy` is the binary Judge result. `format` is `0` for valid output and
`-1` for invalid output. `tool` is `1` only when at least one crop succeeds and
the final answer is correct; failed or malformed calls receive no tool bonus.
Judge settings and optional agent token budgets are defined in the task config; training rollout limits are configured by the launcher's `MAX_RESPONSE_LENGTH`.

## Implementation map

- `dataset.py`: parquet adapter and per-sample task configuration.
- `task_config.yaml`: default task, Judge, agent, and crop-tool settings.
- `train_deepeyes.sh`: verl v1 colocate-async GRPO entry point.
- `uni_agent/agents/deepeyes/`: multimodal policy loop and crop tool.
- `uni_agent/tasks/deepeyes/`: preprocessing, task lifecycle, and reward.
