# Single-node training pipeline smoke test

`train_pipeline_smoke.sh` is the smallest end-to-end Uni-Agent training recipe.
It is intentionally separate from task-specific recipes such as MemAgent and
DeepEyes. The default run performs two complete
rollout -> task reward -> FSDP2 policy-update steps with Qwen3-0.6B.

The task uses the `local` provider only for lifecycle compatibility. The agent is
given exactly one control tool, `finish(answer=...)`; it has no shell, file-edit,
or system-command tool, so generated actions cannot execute commands on the host.

## Prerequisites

Use one Python environment containing editable installs of this repository and
`verl`, plus PyTorch, vLLM, Ray, and TransferQueue. Verify it before training:

```bash
python -c 'import ray, torch, transfer_queue, uni_agent, verl, vllm; print(torch.cuda.device_count())'
```

The first run downloads `Qwen/Qwen3-0.6B` if `MODEL_PATH` is not already a
local model directory. The tiny parquet dataset is regenerated locally on each
default run and does not download external data. Its 16 training and 4
validation prompts are disjoint.

The actor defaults to PyTorch SDPA attention, so FlashAttention is not required
for this smoke test. Set `ATTN_IMPLEMENTATION=flash_attention_2` only when the
optional `flash-attn` package is already installed and compatible with PyTorch.
The default prompt limit is 320 tokens and the model window is 512 tokens.
The prompt includes a tool-call format example to help the small model submit
its own answer through `finish`.

## Run

From the repository root:

```bash
PYTHON_BIN=/path/to/venv/bin/python \
NUM_GPUS=1 \
bash examples/quickstart/training/train_pipeline_smoke.sh
```

A successful run reaches `training/global_step=2`, prints task reward and
actor loss/optimizer metrics to the console, saves the console stream to
`logs/uni-agent-pipeline-smoke/<experiment>/run-XXXXXXXX/train.log`, and checks the required
training signals automatically. W&B and checkpoint saving are off by default.
Each invocation creates a fresh run directory, also when `AGENT_LOG_DIR` is set,
so previous tool-call evidence cannot satisfy the current run's checks.
Acceptance additionally requires at least one correct real `finish` observation
and its saved trajectory with reward 1.0. Plain-text-only runs fail this check.

The useful smoke-test signals are:

- `num_failed_sessions=0` and `num_unfinished_episodes=0` in the rollout summary;
- `critic/rewards/mean` (plus min/max) for task scoring;
- `actor/grad_norm`, `timing_s/update_actor`, and `timing_s/update_weights` for
  the policy update and rollout-weight synchronization.

Set `VERIFY_TRAINING_SIGNALS=0` only for configuration-only commands such as
Hydra `--cfg job`; normal smoke runs should keep verification enabled.

The task gives reward `1.0` for a correct answer submitted through the real
`finish` tool, `0.5` for a correct plain-text fallback, and `0.0` for an
incorrect answer. This keeps correctness as the requirement while providing a
small learning signal for tool-protocol compliance.

The defaults are `ADV_ESTIMATOR=rloo` and `ROLLOUT_N=1`. In verl's single-sample
case this is a zero-baseline REINFORCE update: the advantage is the outcome
reward on each response token. Equal positive rewards therefore remain positive
instead of being centered to zero. This choice targets a small pipeline check,
not a recommendation for full training. Using multiple samples per prompt
restores the leave-one-out baseline and can give zero advantage for equal rewards.
Acceptance still requires a non-zero actor gradient in at least one step.
Model correctness and tool use remain empirical checks; no prompt guarantees
that a sampled response will succeed.

The current defaults completed two training steps on both one and four RTX 3090
GPUs (24 GB each), with other jobs sharing the devices. Both runs passed the
automatic verifier and had non-zero gradients in both steps.

| GPUs | Reward mean, steps 1 / 2 | Actor grad norm, steps 1 / 2 |
| --- | --- | --- |
| 1 | 0.5 / 0.5 | 70.2565 / 12.5023 |
| 4 | 0.5625 / 0.625 | 22.0952 / 27.2137 |

The four-GPU run saved 24 trajectories, including async prefetch, with 10 correct
real `finish` calls. Both training batches included reward 1.0. The one-GPU run
saved six trajectories; its correct `finish` call was in prefetch rather than
the two consumed training batches. A tool JSON parsing error was logged during
the four-GPU run; training continued and rollout summaries reported no failed
sessions or unfinished episodes. This is pipeline validation, not a guarantee
of reliable tool use by the small model.

Two-GPU execution and Ascend NPU execution are not validated by these runs.
The shared task config is device-independent; this launcher targets CUDA.

## 1, 2, or 4 GPUs

The same script scales FSDP2 and colocated rollout workers from one to four GPUs:

```bash
NUM_GPUS=1 bash examples/quickstart/training/train_pipeline_smoke.sh
NUM_GPUS=2 bash examples/quickstart/training/train_pipeline_smoke.sh
NUM_GPUS=4 bash examples/quickstart/training/train_pipeline_smoke.sh
```

`TRAIN_BATCH_SIZE` defaults to `2 * NUM_GPUS`, so it remains divisible by the
FSDP world size. `ROLLOUT_TP=1` creates one small-model rollout replica per GPU.
Set `CUDA_VISIBLE_DEVICES` before the command when only selected GPUs should be
used.

## Memory controls

The defaults target 24 GB GPUs. If vLLM cannot reserve memory, reduce its share:

```bash
GPU_MEMORY_UTILIZATION=0.25 \
bash examples/quickstart/training/train_pipeline_smoke.sh
```

If actor memory is the limit, enable FSDP CPU offload by appending Hydra
overrides:

```bash
bash examples/quickstart/training/train_pipeline_smoke.sh \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
```

Every variable used above can also be set with a local model/data path; run the
script with additional Hydra overrides at the end for advanced settings.
