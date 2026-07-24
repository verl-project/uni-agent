# Run Agent RL Training

Uni-Agent supports RL training for both white-box and black-box Agents. By integrating with the bundled `verl` module, the same Agent workflow can move seamlessly from inference to training.

This guide demonstrates:

1. Train `Qwen3-Coder-30B-A3B-Instruct` with the white-box `ReAct Agent`.
2. Train `Qwen3.5-4B` with the black-box `Claude Code` Agent.

## Prerequisites

We recommend completing the preceding Quickstart guides before starting training to ensure that the Task dependencies and Sandbox service are working correctly.

## Prepare the Data

Both examples train on SWE-reBench and validate on SWE-Bench Verified. The preprocessors convert each dataset row into the Task Config format consumed by Uni-Agent.

### Training Dataset

!!! note "Ready-to-use SWE-reBench dataset"
    You can directly use our processed `swe-rebench-filtered-1150` dataset, which contains 1,150 training samples. We preprocess and filter the original SWE-reBench examples to make them better suited for Agent RL training.

    **Dataset:** [https://huggingface.co/datasets/dyyyyyyyy/swe-rebench-filtered-1150](https://huggingface.co/datasets/dyyyyyyyy/swe-rebench-filtered-1150)

Prepare the filtered SWE-reBench split:

```bash
python3 -m uni_agent.tasks.swe_rebench.preprocess --local-save-dir ~/data/uni_agent
```

The command writes: `~/data/uni_agent/swe_rebench_filtered.parquet`

### Validation Dataset

Prepare SWE-Bench Verified:

```bash
python3 -m uni_agent.tasks.swe_bench.preprocess --local-save-dir ~/data/uni_agent
```

The command writes: `~/data/uni_agent/swe_bench_verified.parquet`

The processed rows remain independent of the runtime Sandbox provider. Each row contains the rendered prompt, task metadata, canonical image reference, and per-sample Task Config.

## Configuration

### Task Configuration

The Quickstart provides separate configs for the two Agent types:

=== "ReAct"

    ```yaml
    - name: swe_bench
      sandbox:
        provider: vefaas  # <-- Change to your Sandbox provider.
        runtime_timeout: 7200
      agent:
        name: react
        max_steps: 200
        tools:
          - name: stateful_shell
            command_timeout: 120
            env_vars:
              PAGER: "cat"
              GIT_PAGER: "cat"
              MANPAGER: "cat"
              TQDM_DISABLE: "1"
              PIP_PROGRESS_BAR: "off"
          - name: str_replace_editor
          - name: submit
        model:
          temperature: 1.0
          top_p: 1.0
          max_total_tokens: 131072

    - name: swe_rebench
      sandbox:
        provider: vefaas  # <-- Change to your Sandbox provider.
        runtime_timeout: 7200
      agent:
        name: react
        max_steps: 200
        tools:
          - name: stateful_shell
            command_timeout: 120
            env_vars:
              PAGER: "cat"
              GIT_PAGER: "cat"
              MANPAGER: "cat"
              TQDM_DISABLE: "1"
              PIP_PROGRESS_BAR: "off"
          - name: str_replace_editor
          - name: submit
        model:
          temperature: 1.0
          top_p: 1.0
          max_total_tokens: 131072
    ```

=== "Claude Code"

    ```yaml
    - name: swe_bench
      sandbox:
        provider: vefaas  # <-- Change to your Sandbox provider.
        runtime_timeout: 7200
      agent:
        name: claude_code
        max_turns: 100
        run_timeout: 4800
        model:
          temperature: 1.0
          top_p: 1.0
          max_total_tokens: 131072

    - name: swe_rebench
      sandbox:
        provider: vefaas  # <-- Change to your Sandbox provider.
        runtime_timeout: 7200
      agent:
        name: claude_code
        max_turns: 100
        run_timeout: 4800
        model:
          temperature: 1.0
          top_p: 1.0
          max_total_tokens: 131072
    ```

    !!! warning "Network connectivity"
        The Claude Code sandbox must be able to reach the GPU machine hosting its session-scoped Gateway endpoint.

### Ray Runtime Environment

Training runs as a Ray job. Use a Runtime Environment to distribute the repository, expose the bundled `verl` source, install lightweight Task and Sandbox dependencies, and pass credentials to every Agent runner.

=== "veFaaS"

    ```yaml
    working_dir: ./
    excludes: ["/.git/"]

    pip:
      packages:
        - "volcengine-python-sdk"
        - "swe-rex"
        - "swebench"

    env_vars:
      PYTHONPATH: "verl"
      PYTHONNOUSERSITE: "1"
      TORCH_NCCL_AVOID_RECORD_STREAMS: "1"
      CUDA_DEVICE_MAX_CONNECTIONS: "1"

      VEFAAS_FUNCTION_ID: "<vefaas-function-id>"
      VEFAAS_FUNCTION_ROUTE: "<vefaas-function-route>"
      VOLCE_ACCESS_KEY: "<volcengine-access-key>"
      VOLCE_SECRET_KEY: "<volcengine-secret-key>"
    ```

=== "Modal"

    ```yaml
    working_dir: ./
    excludes: ["/.git/"]

    pip:
      packages:
        - "modal"
        - "swebench"

    env_vars:
      PYTHONPATH: "verl"
      PYTHONNOUSERSITE: "1"
      TORCH_NCCL_AVOID_RECORD_STREAMS: "1"
      CUDA_DEVICE_MAX_CONNECTIONS: "1"

      MODAL_TOKEN_ID: "<modal-token-id>"
      MODAL_TOKEN_SECRET: "<modal-token-secret>"
    ```

## Case 1: ReAct Agent RL

### Launch Training

This recipe trains `Qwen3-Coder-30B-A3B-Instruct` with the ReAct Task Config. Set the shared data and runtime roots, then launch it from the repository root:

```bash
DATA_DIR=/path/to/data \
RUNTIME_DIR=/path/to/runtime \
NNODES=8 \
CONCURRENCY=1024 \
GEN_TP=4 \
TP=1 PP=2 CP=4 EP=8 ETP=1 \
TRAIN_PROMPT_BSZ=64 \
N_RESP_PER_PROMPT=8 \
PPO_MINI_BATCH_SIZE=16 \
TASK_CONFIG=examples/quickstart/training/task_config_react.yaml \
EXP_NAME=react_qwen3_coder_30b_gspo_r3 \
ADV_ESTIMATOR=rloo \
LOSS_MODE=gspo \
CLIP_RATIO_LOW=4e-4 \
CLIP_RATIO_HIGH=4e-4 \
CLIP_RATIO_C=10 \
LOSS_AGG_MODE=token-mean \
BYPASS_MODE=False \
ROLLOUT_IS=token \
ROLLOUT_IS_THRESHOLD=2.0 \
ROLLOUT_IS_BATCH_NORMALIZE=False \
ROLLOUT_RS=null \
ROUTER_REPLAY_MODE=R3 \
ENABLE_ROLLOUT_ROUTING_REPLAY=True \
LR_DECAY_STEPS=10000 \
TEST_FREQ=-1 \
bash examples/quickstart/training/train_qwen3_moe.sh
```

The default layout is:

```text
<DATA_DIR>/
├── models/Qwen3-Coder-30B-A3B-Instruct/
└── data/uni_agent/
    ├── swe_rebench_filtered_1150.parquet
    └── swe_bench_verified.parquet

<RUNTIME_DIR>/
├── data/uni_agent/runtime_env.yaml
├── ckpts/
└── logs/
```

Override `MODEL_PATH`, `TRAIN_FILE`, `TEST_FILE`, `RUNTIME_ENV`, or `TASK_CONFIG` when your layout differs.

### Monitor the Run

Checkpoints and per-session Agent logs are written under:

```text
<RUNTIME_DIR>/ckpts/Uni-Agent-Qwen3-Coder-30B-megatron/<EXP_NAME>/
<RUNTIME_DIR>/logs/Uni-Agent-Qwen3-Coder-30B-megatron/<EXP_NAME>/
```

### Results

_To be added._

## Case 2: Claude Code RL

### Launch Training

This recipe trains `Qwen3.5-4B` with the Claude Code Task Config:

```bash
DATA_DIR=/path/to/data \
RUNTIME_DIR=/path/to/runtime \
NNODES=4 \
CONCURRENCY=1024 \
TP=4 PP=2 CP=1 \
TASK_CONFIG=examples/quickstart/training/task_config_claude_code.yaml \
EXP_NAME=claude_code_qwen3_5_4b_dppo_tv \
ADV_ESTIMATOR=rloo \
LOSS_MODE=dppo_tv \
CLIP_RATIO_LOW=0.15 \
CLIP_RATIO_HIGH=0.15 \
CLIP_RATIO_C=10000 \
LOSS_AGG_MODE=seq-mean-token-sum-norm \
BYPASS_MODE=False \
ROLLOUT_IS=null \
ROLLOUT_RS=null \
bash examples/quickstart/training/train_qwen3p5_dense.sh
```

The Claude Code runner sets `trajectory_selection=longest`. If a Gateway session materializes multiple trajectories, the Framework keeps only the trajectory with the most model-generated tokens for RL training.

The script expects:

```text
<DATA_DIR>/
├── models/Qwen3.5-4B/
└── data/uni_agent/
    ├── swe_rebench_filtered_1150.parquet
    └── swe_bench_verified.parquet

<RUNTIME_DIR>/
├── data/uni_agent/runtime_env.yaml
├── ckpts/
└── logs/
```

The Claude Code sandbox must be able to reach the session-scoped Gateway running on the GPU cluster.

### Monitor the Run

Outputs are written under:

```text
<RUNTIME_DIR>/ckpts/Uni-Agent-Qwen3.5-4B-megatron/<EXP_NAME>/
<RUNTIME_DIR>/logs/Uni-Agent-Qwen3.5-4B-megatron/<EXP_NAME>/
```

### Results

_To be added._
