# DeepSeek-V4 with Harbor

Prepare [DeepSWE v1.1](https://github.com/datacurve-ai/deep-swe) (113 tasks) and
[MiMo V2.6 Code + Terminal](https://huggingface.co/datasets/dyyyyyyyy/mimo-v26-rl-oss-code-terminal-harbor)
(2,762 tasks, used as training data), and run rollouts with mini-swe-agent 2.4.6
or Claude Code 2.1.148 through Harbor and E2B. Both datasets share the same inference launcher.
The inference example serves DeepSeek-V4-Flash-0731 on one 8-GPU node (TP=4).
The GRPO training example uses VeOmni and vLLM on 16 nodes with 8 GPUs each.

## Prepare the environment

Start with a Python 3.12 GPU environment with `uv` and DeepSeek-V4 support. For
inference, the tested stack used 8 B200 GPUs, PyTorch 2.11.0+cu130 and vLLM 0.24.0.
For training, prepare the same VeOmni environment on every Ray node. The reference
verl training script (`examples/grpo_trainer/run_deepseek_v4_veomni.sh`) specifies
VeOmni 0.1.12, Transformers 5.9.0 and vLLM 0.24.0. The training configuration uses
TileLang kernels for DSA and mHC. Install the shared project dependencies in the
activated environment.

```bash
git clone https://github.com/verl-project/uni-agent.git
cd uni-agent

# Use the DeepSeek-V4-compatible verl revision tested with this recipe.
git clone https://github.com/yyDing1/verl.git ../verl-dsv4
git -C ../verl-dsv4 checkout 3b58d4d408b90d36b0784f046bc062a270c689b7
export VERL_ROOT="$(cd ../verl-dsv4 && pwd)"

uv pip install --python "$(command -v python)" --no-deps -e "$VERL_ROOT" -e .
uv pip install --python "$(command -v python)" \
  "harbor[e2b] @ git+https://github.com/harbor-framework/harbor.git@dba6b9de4bd014d0a9d430b6eb85f6a4ed883b35"
uv pip install --python "$(command -v python)" --no-deps TransferQueue==0.1.11

export E2B_API_KEY="<your-e2b-api-key>"
export E2B_API_URL="http://<your-e2b-api-host>:<port>"
export E2B_SANDBOX_URL="http://<your-e2b-sandbox-host>:<port>"
export MODEL_PATH="/path/to/DeepSeek-V4-Flash-0731"
```

This recipe uses self-hosted E2B. Set both URLs to the full HTTP(S) endpoints
provided by your deployment. The sandbox must be able to reach the model host's Gateway ports.
Run the following commands in this same environment; `uv run --frozen` can
restore incompatible dependency versions from another project's lockfile.

## Prepare the data offline

Copy or mount an existing snapshot of DeepSWE or MiMo on the training machine,
including each complete Harbor task directory. The examples use `/data/dataset`
as the mount point. From the `uni-agent` root, run the command for the
dataset you want to index. The preprocessor scans the local `task.toml` files
and reads the prompts from `instruction.md`.

**DeepSWE v1.1** (snapshot `0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea`)

```bash
python examples/dsv4_rl/prepare_data.py \
  --task-name deepswe \
  --task-root /data/dataset/deep-swe-v1.1/tasks \
  --output data/deepswe_v11.parquet
```

**MiMo V2.6** (snapshot `5c2daf06daf877894547b49ad06203dfa0e0be64`)

```bash
python examples/dsv4_rl/prepare_data.py \
  --task-name mimo \
  --task-root /data/dataset/mimo-v26-rl-oss-code-terminal-harbor/tasks \
  --output data/harbor_mimo-v26-rl-oss-code-terminal-harbor.parquet
```

This MiMo revision removes the setup hosts blocklist, including its embedded
healthcheck payload. Agent execution keeps `network_mode = "no-network"`, with
the model host automatically allowlisted; setup and verification use `public`.

Both Parquet files store absolute task paths. Keep the local task directories
available at those paths in the environment that runs the tasks.

## Run inference

Install the inference-specific pins in the inference environment.

```bash
uv pip install --python "$(command -v python)" --no-deps transformers==5.16.0 tokenizers==0.23.1

# Choose deepswe_v11 (default) or mimo_v26.
export DATASET=mimo_v26

# Verify one task with one rollout first.
LIMIT=1 CONCURRENCY=1 N=1 bash examples/dsv4_rl/run_infer.sh

# Generate four rollouts per task on the selected dataset.
CONCURRENCY=64 N=4 bash examples/dsv4_rl/run_infer.sh
```

For DeepSWE with Claude Code 2.1.148, select its task configuration. Harbor installs
the pinned Claude Code version inside each sandbox; the model remains DeepSeek-V4.

```bash
DATASET=deepswe_v11 TASK_CONFIG=examples/dsv4_rl/task_config_claude_code.yaml \
  CONCURRENCY=64 N=4 bash examples/dsv4_rl/run_infer.sh
```

A full run generates 452 rollouts for DeepSWE or 11,048 for MiMo. Omit `LIMIT`
for the full dataset. Results are written to `outputs/<DATASET>/<RUN_ID>/results.json`, with
per-rollout logs under `logs/` in the same run directory. The score is the mean
reward across rollouts. Override `DATA_PATH`, `OUTPUT_DIR`, `RUN_ID`,
`TENSOR_PARALLEL_SIZE`, `N_GPUS_PER_NODE`, or `GATEWAY_COUNT` as needed.

Optional: verify a reference solution without loading the model:

```bash
python examples/inference/parallel_infer_api.py \
  --data-path data/deepswe_v11.parquet \
  --task-config examples/dsv4_rl/task_config_oracle.yaml \
  --base-url http://unused.invalid/v1 \
  --limit 1 --concurrency 1 \
  --log-dir outputs/deepswe_v11/oracle/logs \
  --result-path outputs/deepswe_v11/oracle/results.json
```

For MiMo, use `data/harbor_mimo-v26-rl-oss-code-terminal-harbor.parquet` and
change the oracle output paths accordingly.

## Run an RL trial

This trial trains DeepSeek-V4-Flash-0731 with VeOmni and uses vLLM for rollouts
(TP=8, EP=8, a 163,840-token context). The default allocation is 16 nodes with
8 GPUs each. Run the commands from the `uni-agent` repository root.

### Prepare the dual-harness training data

Select `mimo` or `deepswe` with `--task-name` and pass the local task directory
to `--task-root`. Use the same snapshot as in [Prepare the data offline](#prepare-the-data-offline).

The preprocessor reads every task under `--task-root` and writes one training
row per task and harness. Both harnesses on all 2,762 MiMo tasks produce 5,524 rows.

```bash
export TRAIN_FILE=/data/dataset/mimo_rl_trial.parquet
python examples/dsv4_rl/prepare_data.py \
  --task-name mimo \
  --task-root /data/dataset/mimo-v26-rl-oss-code-terminal-harbor/tasks \
  --harness '["mini-swe-agent", "claude-code"]' \
  --output "$TRAIN_FILE"
```

Both harnesses on all 113 DeepSWE v1.1 tasks produce 226 rows.

```bash
export TRAIN_FILE=/data/dataset/deepswe_v11_2harness.parquet
python examples/dsv4_rl/prepare_data.py \
  --task-name deepswe \
  --task-root /data/dataset/deep-swe-v1.1/tasks \
  --harness '["mini-swe-agent", "claude-code"]' \
  --output "$TRAIN_FILE"
```

Pass `--harness '["claude-code"]'` for one training row per task. Omitting
`--harness` writes the plain Harbor index used for inference.

### Launch training

Mount the model, Parquet files and the absolute Harbor task paths stored in the
Parquet at the same locations on every node. Use a shared output directory for
checkpoints and agent logs. Before starting Ray, activate the prepared training
environment and export `E2B_API_KEY`, `E2B_API_URL` and `E2B_SANDBOX_URL` on
each node so the Harbor workers inherit them. Set `RAY_HEAD_IP` to the head
node's cluster IP on each node. For a new cluster, start the head first.

```bash
ray start --head --node-ip-address="$RAY_HEAD_IP" --port=6379
```

Run the following on each of the other 15 nodes.

```bash
ray start --address="$RAY_HEAD_IP:6379"
```

On the head node, connect to the cluster and launch once. For an existing Ray
cluster, use its address directly. `ray status` should show all 128 GPUs.
For a different allocation, set `NNODES` and `N_GPUS_PER_NODE` to match it
and adjust the parallelism settings in the launcher. Training logs metrics to
the console and W&B by default. Authenticate with W&B before launching.

```bash
export RAY_ADDRESS="$RAY_HEAD_IP:6379"
ray status
export MODEL_PATH=/data/models/DeepSeek-V4-Flash-0731
export TRAIN_FILE=/data/dataset/mimo_rl_trial.parquet
export OUTPUT_DIR=/data/outputs
bash examples/dsv4_rl/run_train.sh
```

Each Parquet row selects its harness and E2B environment. Harbor passes the
verifier reward to GRPO and uses `openai/DeepSeek-V4-Flash-0731` as the default
agent model name. Set `SERVED_MODEL_NAME` to use a different provider/model alias.

`OUTPUT_DIR` is the shared storage root and defaults to `$PWD/outputs`.
The launcher writes agent logs to `$OUTPUT_DIR/$project_name/$exp_name/logs/`
and checkpoints to `$OUTPUT_DIR/$project_name/$exp_name/ckpts/`.

Validation is disabled by default. verl still constructs a validation loader,
so `VAL_FILE` defaults to `TRAIN_FILE`. To evaluate on held-out tasks every
10 steps, prepare a separate harness Parquet and pass it to the launcher.

```bash
VAL_FILE=/data/dataset/deepswe_v11_2harness.parquet \
  bash examples/dsv4_rl/run_train.sh trainer.test_freq=10
```
