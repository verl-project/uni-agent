# Run the Agent Aware Router

The **Agent Aware Router** routes requests across multiple inference replicas to maximize KV-cache hit rates while keeping replicas evenly loaded.

For the routing algorithm, strategy internals, and design rationale, see [Agent Aware Router](../concepts/agent-aware-router.md).

## Environment

Install the runtime environment first — see the [Installation guide](installation.md).

## Prepare Data

This guide uses SWE-Bench Verified as the running example. Preprocess it first:

```bash
python -m uni_agent.tasks.swe_bench.preprocess --local-save-dir /path/to/swe_agent
```

> The command writes `/path/to/swe_agent/swe_bench_verified.parquet`. It downloads `princeton-nlp/SWE-bench_Verified` from the Hugging Face Hub; if the Hub is unreachable from your network, set `HF_ENDPOINT` to an alternate mirror (e.g. `HF_ENDPOINT=https://hf-mirror.com`) and rerun.

## Task Configuration

The `react` agent drives a ReAct (reason + act) loop: the model reasons, calls a host-side tool, and repeats until it submits. Use the ready-made config at `examples/quickstart/agent_aware_router/task_config_react.yaml`:

```yaml
- name: swe_bench
  sandbox:
    provider: modal
  agent:
    name: react
    max_steps: 100
    tools:
      - name: str_replace_editor
      - name: stateful_shell
        command_timeout: 180
        env_vars:
          PIP_PROGRESS_BAR: "off"
          PAGER: "cat"
          TQDM_DISABLE: "1"
          GIT_PAGER: "cat"
      - name: submit
    model:
      temperature: 0.8
      top_p: 0.9
      max_total_tokens: 65536
```

To use the modal sandbox, configure its service endpoint and credentials — see [Launch a Sandbox](launch-sandbox.md).

## Run the Router

`--task-config` is required — it selects the task / agent / model configuration per row. Use `--help` for the full flag list and defaults:

```bash
bash examples/quickstart/agent_aware_router/run_infer.sh --help
```

Smoke test (1 sample, kv-events on):

```bash
bash examples/quickstart/agent_aware_router/run_infer.sh \
    --model-path /path/to/Qwen3-8B \
    --data-path /path/to/swe_agent/swe_bench_verified.parquet \
    --task-config examples/quickstart/agent_aware_router/task_config_react.yaml \
    --max-samples 1 \
    --concurrency 4
```

> `--device` selects the accelerator backend — `gpu` (default) or `ascend` (Ascend NPUs). 

Full run (omit `--max-samples` to run the whole dataset):

```bash
bash examples/quickstart/agent_aware_router/run_infer.sh \
    --model-path /path/to/Qwen3-8B \
    --data-path /path/to/swe_agent/swe_bench_verified.parquet \
    --task-config examples/quickstart/agent_aware_router/task_config_react.yaml \
    --tensor-parallel-size 2 --n-gpus-per-node 8 \
    --load-threshold 0.9
```

Next, you can [train an agent with RL](rl-training.md) using the same task and rollout configuration.
