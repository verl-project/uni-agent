# Train an Agent with RL

This guide demonstrates Agentic RL training for both white-box and black-box agents:

1. Train `Qwen3-Coder-30B-A3B-Instruct` with the white-box `ReAct Agent`.
2. Train `Qwen3.5-9B` with the black-box `Claude Code` Agent.

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
        provider: vefaas
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
        provider: vefaas
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
        provider: vefaas
        runtime_timeout: 7200
      agent:
        name: claude_code
        max_turns: 100
        run_timeout: 7200
        model:
          temperature: 1.0
          top_p: 1.0
          max_total_tokens: 131072

    - name: swe_rebench
      sandbox:
        provider: vefaas
        runtime_timeout: 7200
      agent:
        name: claude_code
        max_turns: 100
        run_timeout: 7200
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

### Monitor the Run

### Results

## Case 2: Claude Code RL

### Launch Training

### Monitor the Run

### Results
