# SWE-bench Agent Training Example

End-to-end recipe for training a **SWE-bench coding agent** with the **Uni-Agent** framework, using **fully-async RL** (Megatron actors + vLLM rollout replicas on separate nodes) and **Modal swe-rex sandboxes** for safe, parallel code execution during rollout.

The reference configuration trains **Qwen3-235B-A22B-Instruct-2507** with GRPO, but the launch script is fully parameterized — point it at a smaller model and shrink the topology to reproduce on fewer GPUs.

The agent solves each SWE-bench task by iteratively calling three tools inside a per-task Modal sandbox:

- `str_replace_editor` — view / edit repository files
- `execute_bash` — run shell commands (build, run tests, inspect)
- `submit` — submit the final patch for evaluation

The reward is computed by running the task's test suite against the submitted patch (`uni_agent.reward.swe_rebench` for training, `uni_agent.reward.swe_bench` for SWE-bench Verified).

---

## Prerequisites

- A Ray cluster with GPU nodes (the reference uses 12 nodes × 4 GPU: 8 train + 4 rollout). A working verl + Megatron + vLLM install on every node.
- A [Modal](https://modal.com) account and API token. The rollout spins up one swe-rex sandbox per in-flight trajectory, so size your concurrency against your Modal workspace's sandbox quota (see `agent_config.yaml`).
- A Weights & Biases account (or change `trainer.logger` in the launch script).

## Step 1: Prepare the datasets

Build the train (SWE-reBench) and validation (SWE-bench Verified) parquet files with the existing preprocessing scripts:

```bash
python examples/data_preprocess/swe_rebench.py        --local-save-dir ~/data/swe_agent
python examples/data_preprocess/swe_bench_verified.py --local-save-dir ~/data/swe_agent
```

These write `swe_rebench_filtered_*.parquet` and `swe_bench_verified_*.parquet` into `--local-save-dir`. Make that directory reachable from every Ray node (shared filesystem or copied), then point `TRAIN_FILE` / `TEST_FILE` at the exact files produced (see Step 3).

## Step 2: Configure the runtime env

Copy `runtime_env.yaml` and fill in the placeholders:

- `working_dir` and `PYTHONPATH` → your uni-agent and verl checkouts.
- `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` → your Modal token (`modal token new`, or `modal token set --profile=<team>` for a team workspace). Alternatively leave them unset and rely on `~/.modal.toml` on every node.
- `WANDB_API_KEY` → your W&B key (or run `wandb login` on the nodes and remove it).

The file also documents two settings worth keeping:

- **Do not** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — it is incompatible with vLLM's sleep-mode `CuMemAllocator` (pytorch/pytorch#147851).
- Set `CUDA_HOME` — some MoE expert-parallel kernels JIT-compile at runtime and need it to locate `nvcc`.

## Step 3: Launch training

```bash
export RAY_ADDRESS=http://<ray-head>:8265
export DATA_ROOT=/path/to/data-root          # holds hf-models/ and data/swe_agent/
export TRAIN_FILE=$DATA_ROOT/data/swe_agent/swe_rebench_filtered_<impl>.parquet
export TEST_FILE=$DATA_ROOT/data/swe_agent/swe_bench_verified_<impl>.parquet

bash examples/swe_agent_235b/train_qwen3_235b_swebench.sh
```

Topology and parallelism are env-overridable, e.g.:

```bash
NNODES_TRAIN=8 NNODES_ROLLOUT=4 NGPUS_PER_NODE=4 \
ACTOR_TP=2 ACTOR_CP=2 ACTOR_PP=8 ACTOR_EP=4 ACTOR_ETP=1 \
INFER_TP=4 \
bash examples/swe_agent_235b/train_qwen3_235b_swebench.sh
```

Notable settings baked into the script (see its header for the full rationale):

- `max_response_length=128K` — SWE-bench trajectories are long (empirically mean ~70K tokens, ~90 turns); a 32K cap truncates roughly half of them.
- `tool_parser: hermes` (in `agent_config.yaml`) — Qwen3-235B-A22B uses the Hermes tool-call template; the wrong parser silently breaks tool calls.
- `moe_token_dispatcher_type=alltoall` — portable MoE dispatch (no extra expert-parallel comm library required).
- `VLLM_USE_DEEP_GEMM=0` — works around a vLLM 0.21 EP/CUTLASS init issue.
- `performance_mode=interactivity` — favoured throughput in our rollout concurrency sweep for this model.

## Step 4: Monitor

- W&B: reward curve under the configured `project_name` / `experiment_name`.
- Optional Prometheus rollout metrics: set `ENABLE_PROMETHEUS_MONITORING=true` and `PROMETHEUS_CONFIG_FILE=...`; verl rewrites the scrape targets to the live vLLM replicas automatically.
- Per-trajectory agent logs: `log_dir` in `agent_config.yaml` (default `/tmp/swe_agent_rollout_logs/<run_id>/run.log`).

## Tuning notes

- **Rollout concurrency** (`concurrency` in `agent_config.yaml`) is the main throughput/stability knob. Too high vs. the vLLM KV budget causes a preemption cascade; too high vs. your Modal quota causes sandbox-create failures. Start around `20 × (rollout replicas)` and ramp up once steady.
- **Checkpoint storage**: `save_freq=1` + `max_actor_ckpt_to_keep=2` keeps only the two most recent checkpoints; raise `save_freq` if I/O-bound.

## Files

| File | Purpose |
|---|---|
| `train_qwen3_235b_swebench.sh` | Ray job submit + full GRPO / Megatron / vLLM config |
| `agent_config.yaml` | UniAgentLoop config: tools, Modal deployment, concurrency, reward |
| `runtime_env.yaml` | Ray runtime env template (fill in tokens / paths) |
