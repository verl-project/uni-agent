# Harbor Integration

Beyond its built-in task implementations, Uni-Agent integrates [Harbor](https://harborframework.com/) as a task backend through the Harbor CLI, adding support for Harbor-compatible datasets, agents, sandbox providers, and verifiers. Model-backed evaluations can call an external API directly or route through the Uni-Agent Gateway to capture rollout trajectories.

This guide demonstrates three evaluation workflows, each with a dedicated task configuration shown below:

| Benchmark | Agent | Model | Model Access |
| --- | --- | --- | --- |
| SWE-bench Verified | Oracle | — | — |
| Terminal-Bench 2.1 | Terminus-2 | GLM-5.2-733B-A40B | External API Service |
| SWE-bench Pro | OpenHands | Qwen3.6-35B-A3B | Uni-Agent Gateway |

## Install Harbor

Install Harbor in every Ray worker environment, using the extra for your sandbox provider:

```bash
pip install "harbor>=0.16.1"         # Docker
pip install "harbor[modal]>=0.16.1"  # Modal
pip install "harbor[e2b]>=0.16.1"    # E2B
```

See the [Harbor sandbox guide](https://harborframework.com/docs/run-jobs/cloud-sandboxes) for other providers and authentication instructions.

## Prepare a Dataset

Select a dataset from [Harbor Hub](https://hub.harborframework.com/), or index a local dataset that follows Harbor's [task format](https://harborframework.com/docs/tasks):

=== "SWE-bench Verified"

    ```bash
    python -m uni_agent.tasks.harbor.preprocess \
        --dataset-ref swe-bench/swe-bench-verified \
        --local-save-dir ~/data/uni_agent
    ```

=== "Terminal-Bench 2.1"

    ```bash
    python -m uni_agent.tasks.harbor.preprocess \
        --dataset-ref terminal-bench/terminal-bench-2-1 \
        --local-save-dir ~/data/uni_agent
    ```

=== "SWE-bench Pro"

    ```bash
    python -m uni_agent.tasks.harbor.preprocess \
        --dataset-ref scale-ai/swe-bench-pro \
        --local-save-dir ~/data/uni_agent
    ```

=== "Custom dataset"

    ```bash
    python -m uni_agent.tasks.harbor.preprocess \
        --task-root /path/to/your/harbor/tasks \
        --local-save-dir ~/data/uni_agent
    ```

Each command writes a parquet file under `~/data/uni_agent`. Harbor Hub tasks are stored under `~/data/uni_agent/harbor`.

Use `--max-instances N` to limit preprocessing to the first N tasks.

## Configure Harbor

The evaluation commands below use these task configurations:

=== "Oracle"

    `examples/quickstart/harbor/task_config_oracle.yaml`

    ```yaml
    - name: harbor
      agent:
        name: oracle
      harbor_env: modal
      timeout_multiplier: 1.0
    ```

=== "GLM-5.2 + Terminus-2"

    `examples/quickstart/harbor/task_config_glm52.yaml`

    ```yaml
    - name: harbor
      agent:
        name: terminus-2
        timeout_sec: 14400
        kwargs:
          parser_name: json
          temperature: 1.0
          reasoning_effort: max
          max_turns: 500
          suppress_max_turns_warning: true
          model_info:
            max_input_tokens: 262144
            max_output_tokens: 49152
          llm_call_kwargs:
            top_p: 1.0
            max_tokens: 49152
      harbor_env: modal
      timeout_multiplier: 1.0
      override_cpus: 4
      override_memory_mb: 8192
    ```

=== "Qwen3.6 + OpenHands"

    `examples/quickstart/harbor/task_config_openhands.yaml`

    ```yaml
    - name: harbor
      agent:
        name: openhands
      harbor_env: modal
      timeout_multiplier: 1.0
    ```

Supported fields are:

- `name`: must be `harbor`.
- `agent.name`: a built-in Harbor agent, such as `terminus-2`, `claude-code`, `codex`, `openhands`.
- `agent.kwargs`: agent-specific parameters passed as repeated Harbor `--agent-kwarg` options.
- `agent.timeout_sec`: an optional Harbor agent timeout in seconds.
- `harbor_env`: the Harbor environment backend, such as `docker`, `modal`, or `e2b`.
- `timeout_multiplier`: a positive multiplier applied to Harbor's task timeouts; the default is `1.0`.
- `override_cpus` / `override_memory_mb`: optional environment resource overrides.
- `agent.model`: the model endpoint, API key, and model name injected by the inference runner at runtime.

Do not set the Uni-Agent `sandbox` field because Harbor manages the task environment.

## Agent Evaluation

### Run Oracle Solution

Use Harbor's `oracle` agent to validate the task environment and verifier with the task's reference solution:

```bash
NUM_WORKERS=8 \
GLOBAL_CONCURRENCY=128 \
python examples/inference/parallel_infer_api.py \
    --data-path ~/data/uni_agent/harbor_swe-bench_swe-bench-verified.parquet \
    --task-config examples/quickstart/harbor/task_config_oracle.yaml \
    --base-url http://unused.invalid/v1 \
    --log-dir /path/to/log_dir
```

The oracle ignores the required `--base-url` placeholder.

### External API

The following example evaluates `GLM-5.2` on Terminal-Bench 2.1 with Terminus-2 through an external API:

```bash
BASE_URL="https://ark.cn-beijing.volces.com/api/v3" \
API_KEY="replace-with-api-key" \
MODEL="openai/glm-5-2-260617" \
NUM_WORKERS=8 \
python examples/inference/parallel_infer_api.py \
    --data-path ~/data/uni_agent/harbor_terminal-bench_terminal-bench-2-1.parquet \
    --task-config examples/quickstart/harbor/task_config_glm52.yaml \
    --log-dir /path/to/log_dir
```

The endpoint must be reachable from the Harbor agent process.

!!! success "Result"
    Terminus-2 with `GLM-5.2-733B-A40B` achieved a **67.4** score on Terminal-Bench 2.1.

### Gateway Rollout

!!! note "Not yet validated"
    This Harbor + Gateway workflow has not yet been validated end to end.

The following example evaluates `Qwen3.6-35B-A3B` on SWE-bench Pro with OpenHands through the Uni-Agent Gateway:

```bash
ray job submit --no-wait \
    --runtime-env examples/quickstart/inference/runtime_env.yaml \
    --working-dir . \
    -- python3 examples/inference/parallel_infer_verl.py \
    --data-path ~/data/uni_agent/harbor_scale-ai_swe-bench-pro.parquet \
    --model-path Qwen/Qwen3.6-35B-A3B \
    --task-config examples/quickstart/harbor/task_config_openhands.yaml \
    --engine vllm \
    --tool-parser qwen3_coder \
    --tensor-parallel-size 4 \
    --nnodes 8 \
    --n-gpus-per-node 4 \
    --log-dir /mnt/shared/uni_agent_logs \
    --concurrency 512
```

!!! note "Dataset availability"
    Ensure that `--data-path` points to a readable Parquet file at the same path on every Ray worker. Use a shared filesystem path or replicate the dataset to each worker.

The runner launches the model engine, injects a session-scoped Gateway endpoint, and captures token-level trajectories. OpenHands must be able to reach the Gateway from its Harbor environment.

## Results and Rewards

Set `--log-dir` to persist each rollout's logs and Harbor artifacts under a unique log ID:

```text
<log-dir>/<log-id>/
├── task.log
├── trajectory.json  # Gateway Rollout only
└── harbor/
    ├── result.json
    ├── trial.log
    ├── agent/
    ├── verifier/
    └── artifacts/
```

Use a shared `--log-dir` when Ray workers run on different nodes.

The adapter reads `result.json` and returns a Uni-Agent `TaskResult`.

For scalar reporting, it follows Harbor's primary reward convention:

1. Use the `reward` key when present.
2. Otherwise, use the only reward when exactly one key exists.
3. Mark evaluation incomplete when multiple keys exist without `reward`, when
   no verifier reward exists, or when the trial records an exception.

For multi-step tasks, Harbor first computes the configured top-level aggregate
reward. Uni-Agent reads that top-level result and includes per-step summaries in
`eval_report`.

Uni-Agent's per-rollout `task.log` records orchestration messages. Harbor's
trial directory contains the complete agent, verifier, and artifact output.
