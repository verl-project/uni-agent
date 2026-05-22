# Evaluate on Terminal-Bench v2

[Terminal-Bench v2](https://github.com/laude-institute/terminal-bench-2) is an 89-task suite of long-horizon terminal tasks. Each task ships its own Docker image, resource profile, agent / verifier timeout budgets, and `solution` + `tests` directories.

This page shows how to:

1. Preprocess Terminal-Bench v2 into a Uni-Agent parquet.
2. Sanity-check the parquet by running the gold solutions.
3. Run parallel inference with a model and collect rewards.

The runnable scripts live under `examples/data_preprocess/terminal_bench_v2.py` and `examples/agent_interaction/`.

**Reference result:**

| **Model**       | Inference Config                          | **Uni-Agent**     |
| --------------- | ----------------------------------------- |:-----------------:|
| Qwen3.6-35B-A3B | temp=1.0, top_p=0.95, tp=8, 200K context  | **42.53** (Avg@1) |

---

## Why a dedicated guide

Unlike SWE-Bench, where every task shares the same image family, tool list, and timeout budget, **every Terminal-Bench v2 task has its own deployment config** — different Docker image, CPU / memory request, `agent.timeout_sec`, `verifier.timeout_sec`. Encoding all of that into one YAML would mean either one YAML per task or a long stack of `--override` flags.

Uni-Agent solves this with **per-sample agent config**: the preprocessing script emits a complete `tools_kwargs` (env + reward + interaction + tools) into each parquet row's `extra_info`, and `UniAgentLoop._init_config` deep-merges those over the agent-loop YAML at run time. For Terminal-Bench v2 the YAML degrades to a thin shell that only carries `_target_`, `name`, `concurrency`, and `log_dir`; everything else comes from the dataset row. See `examples/data_preprocess/terminal_bench_v2.py` for the exact `tools_kwargs` shape.

---

## Step 1: Preprocess the dataset

The preprocessor clones [`terminal-bench-2`](https://github.com/laude-institute/terminal-bench-2) at a pinned commit, then for each task packs `solution/` and `tests/` into deterministic tar.gz blobs and writes one parquet row with:

- `extra_info.tools_kwargs.env` — full `AgentEnvConfig` (modal deployment, per-task image, CPU / memory, timeouts, `env_variables`).
- `extra_info.tools_kwargs.reward` — `name="terminal_bench_v2"`, `metadata` (task config + solution / tests archives + workdir), `eval_timeout`.
- `extra_info.tools_kwargs.interaction` — `action_timeout` (= `task.agent.timeout_sec`) and a generous `max_turns` safety net.
- `extra_info.tools_kwargs.tools` — `execute_bash`, `str_replace_editor`, `submit`.

Currently only the Modal deployment backend is supported:

```bash
DEPLOYMENT=modal python examples/data_preprocess/terminal_bench_v2.py \
    --local-save-dir ~/data/swe_agent
```

This writes `~/data/swe_agent/terminal_bench_v2_modal.parquet`. Two tasks (`qemu-alpine-ssh`, `qemu-startup`) are currently skipped because Modal sandbox creation does not work for them; the remaining 87 rows are included.

---

## Step 2: Verify the parquet with gold solutions

Before spending GPU time on inference, run the included gold solutions through the same Modal deployment + reward spec to confirm the parquet is healthy. `parallel_verify_terminal_bench.py` starts each task's sandbox, applies the gold `solve.sh`, runs `test.sh`, and aggregates pass / fail / timeout counts:

```bash
python examples/agent_interaction/parallel_verify_terminal_bench.py \
    --data-path ~/data/swe_agent/terminal_bench_v2_modal.parquet \
    --num-workers 8
```

Useful flags:

- `--limit N` — only verify the first `N` rows (smoke test).
- `--task-ids id1,id2` — verify a specific subset by `task_id`.

A healthy parquet should resolve essentially all tasks. Anything in the `fail_tle` (verifier did not complete) bucket points to a deployment or timeout config problem rather than a model problem.

---

## Step 3: Run parallel inference

Once the parquet verifies, run the agent loop with `parallel_infer.py`. The matching agent-loop YAML is intentionally minimal because the parquet carries the per-task config:

```yaml
# examples/agent_interaction/agent_config_terminal_bench.yaml
- name: swe_agent
  _target_: uni_agent.agent_loop.UniAgentLoop
  concurrency: 128
  log_dir: /tmp/terminal_bench_eval
  mask_abnormal_exit_traj: false
```

Submit the inference job (Qwen3.6-35B-A3B at 200K context is the reference config above):

```bash
ray job submit --no-wait \
    --runtime-env $RAY_DATA_HOME/data/swe_agent/runtime_env.yaml \
    --working-dir . \
    -- python3 examples/agent_interaction/parallel_infer.py \
    --data-path $RAY_DATA_HOME/data/swe_agent/terminal_bench_v2_modal.parquet \
    --agent-config-path examples/agent_interaction/agent_config_terminal_bench.yaml \
    --model-path $RAY_DATA_HOME/models/Qwen3.6-35B-A3B --tp 8 \
    --prompt-length 8192 \
    --response-length 204800 \
    --temperature 1.0 --top-p 0.95 --n 1 \
    --num-workers 8 --nnodes 1
```

Notes:

- `concurrency` (in the YAML) and `--num-workers` together bound how many Modal sandboxes are alive at once. Modal's per-account sandbox quota is usually the binding constraint — start low and ramp up.
- The dataset's `interaction.action_timeout` is already set to each task's declared `agent.timeout_sec`; do not override it from the CLI unless you intend to truncate task budgets.
- Per-task trajectories, rewards, and logs land under `log_dir/<run_id>/` (one directory per sample).

---

## Where to look next

- `uni_agent/reward/terminal_bench.py` — the `terminal_bench_v2` reward spec (uploads gold / tests archives, runs `test.sh`, parses `reward.json`).
- `uni_agent/agent_loop.py` — `UniAgentLoop._init_config`, the merge between the YAML and per-sample `tools_kwargs`.
- `examples/agent_interaction/agent_config_modal.yaml` vs. `agent_config_terminal_bench.yaml` — contrast a "YAML carries defaults" agent config with a "YAML is a thin shell" one.
