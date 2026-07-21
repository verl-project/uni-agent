# Run Agent Inference

Uni-Agent supports two inference modes:

1. **External API mode:** connect an agent to an existing model endpoint.
2. **verl rollout mode:** launch a rollout engine using `verl` and run inference.

This guide demonstrates both modes by running `Claude Code` and `ReAct` agents on SWE-Bench.

Specifically, this guide covers:

- **Claude Code** with `Doubao-Seed-2.1-Pro` through an external API, and `Qwen3.6-35B-A3B` through the verl rollout engine.
- **ReAct Agent** with `Doubao-Seed-2.1-Pro` through an external API, plus `Qwen3-Coder-30B-A3B-Instruct` and `Qwen3.6-35B-A3B` through the verl rollout engine.

## Prepare Data

This guide uses SWE-Bench Verified as the running example. Preprocess a small subset first:

```bash
python -m uni_agent.tasks.swe_bench.preprocess --local-save-dir ~/data/swe_agent
```

The command writes `~/data/swe_agent/swe_bench_verified.parquet`.

Each row contains the prompt and a provider-agnostic task definition under `extra_info.tools_kwargs.task`. The selected sandbox backend maps the task image at runtime.

## Task Configuration

For each dataset sample, Uni-Agent resolves the final Task Config in this order:

1. Select the YAML defaults whose `name` matches the sample's task name.
2. Deep-merge the sample Task Config over those defaults.
3. Inject the runtime model endpoint as the final override.

The Quickstart includes two ready-to-use configs:

=== "Claude Code"

    `examples/quickstart/inference/task_config_claude_code.yaml`

    ```yaml
    - name: swe_bench
      sandbox:
        provider: modal
        runtime_timeout: 7200
        sandbox_kwargs:
          memory_gb: 8
      agent:
        name: claude_code
        max_turns: 200
        run_timeout: 4800
        verbose: true
        model:
          temperature: 1.0
          top_p: 0.95
          max_total_tokens: 131072
    ```

=== "ReAct"

    `examples/quickstart/inference/task_config_react.yaml`

    ```yaml
    - name: swe_bench
      sandbox:
        provider: modal
      agent:
        name: react
        max_steps: 100
        tools:
          - name: stateful_shell
            command_timeout: 180
            env_vars:
              PIP_PROGRESS_BAR: "off"
              PAGER: "cat"
              TQDM_DISABLE: "1"
              GIT_PAGER: "cat"
          - name: str_replace_editor
          - name: submit
        model:
          temperature: 0.8
          top_p: 0.9
          max_total_tokens: 65536
    ```

Configure the sandbox provider and Agent limits in YAML. Do not hard-code the runtime endpoint there unless every run uses the same service: API mode injects it from `--base-url` and `--model`, while verl mode injects the session Gateway endpoint.

!!! note "Claude Code network access"
    Claude Code runs inside the sandbox and calls the Anthropic Messages endpoint from there. The endpoint must therefore be resolvable and reachable **from inside the sandbox**.

## External API Mode

The endpoint must support the protocol used by the selected Agent:

- ReAct calls OpenAI-compatible `POST /v1/chat/completions`.
- Claude Code calls Anthropic-compatible `POST /v1/messages`.

Modern vLLM can expose both routes. If you self-host the model, start the server before running Uni-Agent:

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --served-model-name Qwen3-Coder-30B-A3B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8000
```

Then point the API runner at the service:

```bash
BASE_URL=http://model-service:8000/v1 \
MODEL=Qwen3-Coder-30B-A3B-Instruct \
NUM_WORKERS=4 \
python examples/inference/parallel_infer_api.py \
    --data-path ~/data/swe_agent/swe_bench_verified.parquet \
    --task-config /path/to/task_config.yaml \
    --concurrency 8 \
    --log-dir /mnt/shared/uni_agent_logs \
    --limit 8
```

For an authenticated endpoint, also set `API_KEY` or pass `--api-key`. The script expands `--n` rollout attempts per sample, distributes them across Ray actors, and prints resolved, wrong-answer, and timeout/error counts.

Each rollout writes `<log_dir>/<log_id>/task.log`. Set `--log-dir` to a shared-storage path when Ray workers may run on different nodes.

Useful controls:

- `--concurrency`: maximum number of in-flight tasks across all workers; defaults to `GLOBAL_CONCURRENCY`.
- `NUM_WORKERS`: number of Ray inference actors.
- `--limit`: number of dataset rows to run.
- `--n`: rollout attempts per task.
- `--log-dir`: runtime root for per-rollout `task.log` files.

## verl Rollout Engine Mode

This mode requires the standard `verl` inference environment, GPUs, Ray, and TransferQueue. The execution path is:

```text
verl LLMServerManager (vLLM or SGLang)
    -> AgentFrameworkRolloutAdapter
    -> Uni-Agent Gateway sessions
    -> Task Runner and sandbox
    -> TransferQueue trajectories and rewards
```

Run a small example from the Ray head node:

```bash
python examples/inference/parallel_infer_verl.py \
    --data-path ~/data/swe_agent/swe_bench_verified.parquet \
    --model-path Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --task-config /path/to/task_config.yaml \
    --engine vllm \
    --tool-parser qwen3_coder \
    --tensor-parallel-size 4 \
    --n-gpus-per-node 4 \
    --n 1 \
    --log-dir /mnt/shared/uni_agent_logs \
    --limit 8
```

Important controls:

- `--engine`: rollout backend, either `vllm` or `sglang`.
- `--tool-parser`: parser matching the model's chat template.
- `--tensor-parallel-size`: tensor parallel size of the rollout engine.
- `--nnodes` and `--n-gpus-per-node`: hardware allocated to the engine.
- `--gateway-count`: number of Gateway actors.
- `--concurrency`: maximum number of in-flight Gateway sessions.
- `--n`: rollout sessions per task.
- `--result-path`: optional JSON output containing aggregate and per-session scores.
- `--log-dir`: runtime root for Framework logs, Task logs, and trajectory artifacts.

Task/Agent/Tool events go to `task.log`; Framework events go to `framework.log`.

For a Ray cluster, you can submit your job via a pre-defined Runtime Environment, for example:

```yaml
working_dir: ./
excludes:
  - "/.git/"

pip:
  packages:
    - "modal"

env_vars:
  PYTHONPATH: "verl"
  TORCH_NCCL_AVOID_RECORD_STREAMS: "1"
  CUDA_DEVICE_MAX_CONNECTIONS: "1"
  VLLM_DISABLE_COMPILE_CACHE: "1"

  # if you use veFaaS Sandbox
  VEFAAS_FUNCTION_ID: "<vefaas-function-id>"
  VEFAAS_FUNCTION_ROUTE: "<vefaas-function-route>"
  VOLCE_ACCESS_KEY: "<volcengine-access-key>"
  VOLCE_SECRET_KEY: "<volcengine-secret-key>"

  # if you use Modal Sandbox
  MODAL_TOKEN_ID: "<modal-token-id>"
  MODAL_TOKEN_SECRET: "<modal-token-secret>"
```

Then submit your job

```bash
ray job submit --no-wait \
    --runtime-env examples/quickstart/inference/runtime_env.yaml \
    --working-dir . \
    -- python3 examples/inference/parallel_infer_verl.py ...
```

## Recipes

### Claude Code

Claude Code is a black-box Agent Harness: the complete CLI runs inside the sandbox and owns its interaction loop and tools. The following examples run it with `doubao-seed-2.1-pro` through an external API and with `qwen3.6-35b-a3b` through the verl rollout engine.

=== "Doubao-Seed-2.1-Pro"

    Configure any Anthropic-compatible model service that is reachable from the sandbox. The example below uses Doubao through Volcengine Ark:

    ```bash
    export BASE_URL="https://ark.cn-beijing.volces.com/api/compatible"
    export API_KEY="replace-with-your-ark-api-key"
    export MODEL="doubao-seed-2-1-pro-260628"

    python3 examples/inference/parallel_infer_api.py \
        --data-path ~/data/swe_agent/swe_bench_verified.parquet \
        --task-config examples/quickstart/inference/task_config_claude_code.yaml \
        --base-url "${BASE_URL}" \
        --model "${MODEL}" \
        --api-key "${API_KEY}" \
        --concurrency 64 \
        --log-dir /mnt/shared/uni_agent_logs \
        --limit 4
    ```

    !!! note "Result"
        We ran only a small subset with Doubao. Treat this as an end-to-end smoke test, not a benchmark result.

=== "Qwen3.6-35B-A3B"

    Choose your sandbox backend and replace the credential placeholders in `runtime_env.yaml`, then submit the job via:

    ```bash
    ray job submit --no-wait \
        --runtime-env examples/quickstart/inference/runtime_env.yaml \
        --working-dir . \
        -- python3 examples/inference/parallel_infer_verl.py \
        --data-path ~/data/swe_agent/swe_bench_verified.parquet \
        --model-path Qwen/Qwen3.6-35B-A3B \
        --task-config examples/quickstart/inference/task_config_claude_code.yaml \
        --tool-parser qwen3_coder \
        --tensor-parallel-size 4 \
        --nnodes 8 \
        --n-gpus-per-node 8 \
        --log-dir /mnt/shared/uni_agent_logs \
        --concurrency 128
    ```

    !!! success "Result"
        Claude Code with Qwen3.6-35B-A3B achieved a **67.8% resolve rate** on SWE-Bench Verified, with `max-turns` = 200, `temperature` = 1.0, `top-p` = 0.95.

### ReAct Agent

ReAct is a white-box Agent: Uni-Agent owns the interaction loop and exposes `stateful_shell`, `str_replace_editor`, and `submit` from `task_config_react.yaml`. The examples below run the same Agent with an external Doubao service and two verl-managed Qwen checkpoints.

=== "Qwen3-Coder-30B-A3B-Instruct"

    ```bash
    ray job submit --no-wait \
        --runtime-env examples/quickstart/inference/runtime_env.yaml \
        --working-dir . \
        -- python3 examples/inference/parallel_infer_verl.py \
        --data-path ~/data/swe_agent/swe_bench_verified.parquet \
        --model-path Qwen/Qwen3-Coder-30B-A3B-Instruct \
        --task-config examples/quickstart/inference/task_config_react.yaml \
        --tool-parser qwen3_coder \
        --tensor-parallel-size 4 \
        --nnodes 8 \
        --n-gpus-per-node 8 \
        --log-dir /mnt/shared/uni_agent_logs \
        --result-path /mnt/shared/results/react_qwen3_coder_30b.json \
        --concurrency 512
    ```

    !!! success "Result"
        ReAct with Qwen3-Coder-30B-A3B-Instruct achieved a **48.8% resolve rate** on SWE-Bench Verified, with `max-turns` = 100, `temperature` = 0.8, `top-p` = 0.9.

=== "Qwen3.6-35B-A3B"

    ```bash
    ray job submit --no-wait \
        --runtime-env examples/quickstart/inference/runtime_env.yaml \
        --working-dir . \
        -- python3 examples/inference/parallel_infer_verl.py \
        --data-path ~/data/swe_agent/swe_bench_verified.parquet \
        --model-path Qwen/Qwen3.6-35B-A3B \
        --task-config examples/quickstart/inference/task_config_react.yaml \
        --tool-parser qwen3_coder \
        --tensor-parallel-size 4 \
        --nnodes 8 \
        --n-gpus-per-node 8 \
        --log-dir /mnt/shared/uni_agent_logs \
        --result-path /mnt/shared/results/react_qwen3_6_35b.json \
        --concurrency 512
    ```

    !!! success "Result"
        ReAct with Qwen3.6-35B-A3B achieved a **72.6% resolve rate** on SWE-Bench Verified, with `max-turns` = 200, `temperature` = 1.0, `top-p` = 0.95.

=== "Doubao-Seed-2.1-Pro"

    ReAct uses the OpenAI-compatible Chat Completions protocol. Configure an endpoint that is reachable from the Ray inference workers; the example below uses Volcengine Ark:

    ```bash
    export BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
    export API_KEY="replace-with-your-ark-api-key"
    export MODEL="doubao-seed-2-1-pro-260628"

    python3 examples/inference/parallel_infer_api.py \
        --data-path ~/data/swe_agent/swe_bench_verified.parquet \
        --task-config examples/quickstart/inference/task_config_react.yaml \
        --base-url "${BASE_URL}" \
        --model "${MODEL}" \
        --api-key "${API_KEY}" \
        --concurrency 64 \
        --log-dir /mnt/shared/uni_agent_logs \
        --limit 4
    ```

    !!! note "Result"
        _To be added._

Next, you can [train an agent with RL](rl-training.md) using the same task and rollout configuration.
