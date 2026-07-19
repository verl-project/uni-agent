# Run Agent Inference

Uni-Agent supports two inference modes:

1. **External API mode:** connect an agent to an existing model endpoint.
2. **veRL rollout mode:** launch a rollout engine using `verl` and run inference.

This guide demonstrates both modes, using SWE-Bench task as an example:

1. Run an ReAct Agent with an external modal API,
2. Run `Claude Code` with verl rollout engine.

## Prepare Data

This guide uses SWE-Bench Verified as the running example. Preprocess a small subset first:

```bash
python -m uni_agent.tasks.swe_bench.preprocess --local-save-dir ~/data/swe_agent
```

The command writes `~/data/swe_agent/swe_bench_verified.parquet`.

Each row contains the prompt and a provider-agnostic task definition under `extra_info.tools_kwargs.task`. The selected sandbox backend maps the task image at runtime.

## Task Configuration

Both inference modes use the same task configuration format. Choose an agent implementation and configure its sandbox, interaction limits, and model parameters.

=== "ReAct"

    ReAct is a white-box agent: Uni-Agent owns the interaction loop and exposes an explicit list of tools to the model. See [`task_config.yaml`](https://github.com/verl-project/uni-agent/blob/main/examples/inference/task_config.yaml).

    ```yaml
    - name: swe_bench
      log_dir: /tmp/uni_agent_logs/swe_bench
      sandbox:
        provider: modal
      agent:
        name: react
        max_steps: 200
        tools:
          - name: stateful_shell
            command_timeout: 180
          - name: str_replace_editor
          - name: submit
        model:
          temperature: 0.8
          top_p: 0.9
          max_total_tokens: 65536
    ```

=== "Claude Code"

    Claude Code is a black-box agent harness: the complete CLI runs inside the sandbox and connects to the configured model endpoint. See [`task_config_claude_code.yaml`](https://github.com/verl-project/uni-agent/blob/main/examples/inference/task_config_claude_code.yaml).

    ```yaml
    - name: swe_bench
      log_dir: /tmp/uni_agent_logs/swe_bench_claude_code
      sandbox:
        provider: "your-provider"
        runtime_timeout: 7200
        sandbox_kwargs:
          memory_gb: 8
      agent:
        name: claude_code
        workdir: /testbed
        max_turns: 200
        run_timeout: 4800
        verbose: true
        model:
          temperature: 1.0
          top_p: 0.95
          max_total_tokens: 131072
    ```

Common fields:

- `name`: task name used to route each dataset row.
- `sandbox`: backend and lifecycle settings for the task environment.
- `agent.name`: agent implementation or harness to launch.
- `agent.model`: sampling parameters and total token budget.
- `log_dir`: per-session execution logs.

Agent-specific fields configure the interaction loop. ReAct declares its tools and `max_steps`; Claude Code configures its working directory, turn limit, and process timeout.

## External API

Use this mode when you already have an OpenAI-compatible endpoint. The endpoint may be a local vLLM/SGLang server or a hosted model API. The inference driver does not load model weights or require GPUs.

### Start a Model Server

For example, start vLLM:

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --served-model-name Qwen3-Coder-30B-A3B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --tensor-parallel-size 4
```

The tool-call parser must match the model's chat template.

### Run Parallel Inference

Point `parallel_infer_api.py` at the endpoint:

```bash
BASE_URL=http://localhost:8000/v1 \
MODEL=Qwen3-Coder-30B-A3B-Instruct \
GLOBAL_CONCURRENCY=32 \
NUM_WORKERS=4 \
python examples/inference/parallel_infer_api.py \
    --data-path ~/data/swe_agent/swe_bench_verified.parquet \
    --task-config examples/inference/task_config.yaml \
    --limit 8
```

For an authenticated endpoint, also set `API_KEY` or pass `--api-key`.

The script uses Ray actors to run tasks concurrently, but all model requests go directly to the external endpoint. It reports resolved tasks, wrong answers, timeout/errors, pass rate, and average evaluation time.

Useful controls:

- `GLOBAL_CONCURRENCY`: maximum number of in-flight tasks across all workers.
- `NUM_WORKERS`: number of Ray inference actors.
- `--limit`: number of dataset rows to run.
- `--n`: rollout attempts per task.

## verl Rollout Engine

Use this mode when you have a local model checkpoint and want inference to match the RL rollout path. `parallel_infer_verl.py` asks `verl` to launch and manage the rollout engine, then sends agent sessions through the Uni-Agent training stack.

The execution path is:

```text
verl LLMServerManager (vLLM or SGLang)
    -> AgentFrameworkRolloutAdapter
    -> Uni-Agent Gateway sessions
    -> Task Runner and sandbox
    -> TransferQueue trajectories and rewards
```

This mode requires the standard `verl` inference environment, GPUs, and TransferQueue.

Run a small single-node example:

```bash
python examples/inference/parallel_infer_verl.py \
    --data-path ~/data/swe_agent/swe_bench_verified.parquet \
    --model-path ~/models/Qwen3-Coder-30B-A3B-Instruct \
    --task-config examples/inference/task_config.yaml \
    --engine vllm \
    --tool-parser qwen3_coder \
    --tensor-parallel-size 4 \
    --n-gpus-per-node 4 \
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

The task reports its reward to the Gateway. The Agent Framework writes the reward and token-level trajectory to TransferQueue, and the script reads back `rm_scores` to produce the inference summary.

## Ray Cluster

The verl-managed entry point can be submitted as a Ray job:

```bash
ray job submit --no-wait \
    --runtime-env examples/inference/runtime_env.yaml \
    --working-dir . \
    -- python3 examples/inference/parallel_infer_verl.py \
    --data-path ~/data/swe_agent/swe_bench_verified.parquet \
    --model-path /path/to/model \
    --task-config examples/inference/task_config.yaml \
    --tool-parser qwen3_coder \
    --tensor-parallel-size 4 \
    --nnodes 2 \
    --n-gpus-per-node 8 \
    --concurrency 128
```

Use the Ray Runtime Environment to distribute the repository, expose the bundled `verl` source, install optional dependencies, and inject sandbox credentials.

## Choose a Mode

- Start with **external API mode** when validating a task, agent, or hosted model endpoint.
- Use **verl-managed rollout mode** when benchmarking a local checkpoint, collecting training-format trajectories, or validating the exact rollout path before RL training.

Next, [train an agent with RL](rl-training.md) using the same task and rollout configuration.
