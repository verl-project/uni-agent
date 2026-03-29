# Parallel Agent Interaction

Another important class of agent tasks is repeated environment interaction: give the model a real task, let it inspect the workspace, call tools, make edits, and keep going for multiple turns until it either succeeds or runs out of budget.

This page focuses on this pattern. After setting up an agent environment, we show how to run **model-environment interaction** at scale. We use the **SWE agent** workflow as an example: prepare data, run the agent loop with multiple workers, and perform verification.

All runnable code for this page lives under `examples/agent_interaction`.

**Model performance on swe-bench-verified using uni-agent:**

| **Model** | Inference Config | **Results (Avg@N)** | Hardware | Time |
|-------|------------------|-----------------|----------|------|
| Qwen3-Coder-30B-A3B-Instruct | temp=0.8, topp=0.9, tp=4, 100 turns, 64k context | 49.2 (N=4) | A100 Node x 4 | 91 min |
| Qwen3-Coder-480B-A35B-Instruct |                  | todo | A100 Node x 4 |      |
| Qwen3-Coder-Next |                  | todo | A100 Node x 4 |      |

---

## Step 1: Prepare the dataset

We start from data, because a parallel interaction workload is more than just a prompt string. Each sample needs not only the task description, but also the environment and reward metadata required to run the interaction and verification correctly.

Use `prepare_data.py` to fetch [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) and build a Parquet file in the format the framework expects. Each row includes prompts and `extra_info.tools_kwargs`, which the agent loop uses when starting each task.

```bash
DEPLOYMENT=vefaas python examples/agent_interaction/prepare_data.py --local-save-dir ~/data/swe_agent
```

The keys under `tools_kwargs` are defined per sample and must match what the agent loop and the `RewardSpec` expect:

```python
"tools_kwargs": {
    "env": {
        "image": image_name,
        "post_setup_cmd": reset_script,
    },
    "reward": {
        "name": "swe_bench",
        "metadata": example,
    },
},
```

- **`tools_kwargs.env`**: Per-sample environment setup:
  - `image` is the Docker image for that instance, for example one provided by VEFAAS.
  - `post_setup_cmd` is the shell command run after the environment starts, for example `git checkout <base_commit>` followed by cleanup commands to restore the codebase to the correct state.
- **`tools_kwargs.reward`**: Keys must match the selected `RewardSpec`'s constructor parameters (e.g. `name`, `metadata`).

By default, this writes `~/data/swe_agent/swe_bench_verified.parquet`. Use this path as `--data-path` in the next step.

---

## Step 2: Run parallel inference

Once the dataset is ready, the main script is `parallel_infer.py`. This is where the workload starts to look like a real agent system rather than a single local demo: Uni-Agent loads your model, starts multiple workers, gives each worker its own sandbox, and runs the full interaction loop over the dataset.

Each sample can be run with multiple rollouts (`--n`), and the results are aggregated into a mean reward score, that is, a pass rate.

### Single-Node

```bash
python examples/agent_interaction/parallel_infer.py \
    --data-path ~/data/swe_agent/swe_bench_verified.parquet \
    --model-path ~/models/Qwen3-Coder-30B-A3B-Instruct \
    --agent-config-path examples/agent_interaction/agent_config.yaml \
    --num-workers 8 \
    --max-turns 100 \
    --max-samples 4
```

- **`--num-workers`**: Number of parallel agent environments (sandboxes). More workers use more concurrent tasks; tune to your deployment and quota.
- **`--max-samples`**: Limit how many dataset rows to run (`-1` = no limit, full dataset).
- **`--n`**: Number of rollouts per prompt (default 1). Increase for multiple samples per instance.

### Multi-node / Ray job submission

To run on a Ray cluster, submit the same script with `ray job submit` and provide a runtime environment YAML. In practice, you must set all four keys in that file: `VEFAAS_FUNCTION_ID`, `VEFAAS_FUNCTION_ROUTE`, `VOLCE_ACCESS_KEY`, and `VOLCE_SECRET_KEY`. See `examples/agent_interaction/runtime_env.yaml` for an example.

```bash
ray job submit --no-wait \
    --runtime-env examples/agent_interaction/runtime_env.yaml \
    --working-dir . \
    -- python3 examples/agent_interaction/parallel_infer.py \
    --data-path ~/data/swe_agent/swe_bench_verified.parquet \
    --model-path ~/models/Qwen3-Coder-30B-A3B-Instruct \
    --agent-config-path examples/agent_interaction/agent_config.yaml \
    --nnodes 4 \
    --n-gpus-per-node 8 \
    --max-samples -1
```

Edit `runtime_env.yaml` to set your credentials, and do not commit real secrets.

### Agent config

To make the setup easier, Uni-Agent groups the environment, tool, and interaction parameters into a single agent config. During inference, the script reads this YAML file, for example `examples/agent_interaction/agent_config.yaml`.

Below is an annotated version. Each comment explains the corresponding key and the module that uses it.

```yaml
# examples/agent_interaction/agent_config.yaml

- name: swe_agent
  # Agent name for logs and output paths.

  _target_: uni_agent.agent_loop.UniAgentLoop
  # Class to instantiate (Hydra-style). Keep as is for the standard agent loop.

  concurrency: 512
  # Max concurrent agent loops. AgentLoopManager uses this as the semaphore size.

  log_dir: /tmp/swebench_qwen3_coder
  # Base dir for run logs; each run writes to log_dir/run_id.

  interaction:
    # Passed to MultiTurnInteraction (uni_agent.interaction.interaction).
    action_timeout: 300   # Seconds per tool/action call; command is cancelled if exceeded.
    max_turns: 100        # Max turns (model -> action -> observation) per episode.

  env:
    # Passed to AgentEnvConfig -> AgentEnv. image and post_setup_cmd overridden per sample from tools_kwargs.
    deployment:
      # Deployment backend configuration. image is set at runtime from tools_kwargs.
      type: vefaas
      command: curl -fsSL https://pjw-test-empty.tos-cn-beijing.ivolces.com/bin/tos_swe_rex.sh | bash -s -- {token}
      timeout: 600        # Runtime operation timeout (seconds).
      function_id: xxxxxx
      function_route: xxxxxx
    env_variables:
      # Env vars set in the sandbox after start.
      PIP_PROGRESS_BAR: "off"
      PIP_CACHE_DIR: "~/.cache/pip"
      PAGER: "cat"
      MANPAGER: "cat"
      LESS: "-R"
      TQDM_DISABLE: "1"
      GIT_PAGER: "cat"

  tools:
    # Tools installed into the sandbox and exposed to the model.
    - name: str_replace_editor
    - name: execute_bash
    - name: submit

  reward:
    # Base reward configuration. Sample-specific reward metadata is merged from tools_kwargs.reward.
    eval_timeout: 600
```

---

## Step 3: Parallel verification (optional)

If you want to **re-run evaluation only**, for example to apply the gold patch and compute the reward again without running the model, use `parallel_verify_swe.py`. It loads the same Parquet dataset, starts Ray actors, and runs each sample in an agent environment to determine whether the instance is resolved, along with optional metrics such as execution time. This is useful for validating results or rescoring them with different reward settings.

**Prerequisites:** Same as for inference: VEFAAS, or your chosen deployment, must be configured, and the required credentials must be set, for example `VEFAAS_FUNCTION_ID`, `VEFAAS_FUNCTION_ROUTE`, `VOLCE_ACCESS_KEY`, and `VOLCE_SECRET_KEY`.

**Run:**

1. Set the data path inside the script, or add CLI arguments, to point to your prepared Parquet file, for example `~/data/swe_agent/swe_bench_verified.parquet`.
2. Run the script from the repository root:

```bash
python examples/agent_interaction/parallel_verify_swe.py
```

The script uses a fixed number of Ray workers, for example 8, and a semaphore, for example 64, to cap the number of concurrent environments. It prints aggregate statistics, including total samples, success count, failure count, timeout count, and average execution time, as well as the instance IDs for failures.

---

## Quick Reference

| Step              | Script                    | Purpose |
|-------------------|---------------------------|--------|
| Prepare data | `prepare_data.py` | Download SWE-bench Verified and write a Parquet file with `tools_kwargs` |
| Parallel inference | `parallel_infer.py` | Run the agent loop with many workers and compute the mean reward score |
| Parallel verify | `parallel_verify_swe.py` | Re-run evaluation only, including gold patch application and reward computation, in parallel |

| Argument / config   | Meaning |
|---------------------|--------|
| `--data-path`       | Path to prepared Parquet dataset. |
| `--model-path`      | Local model checkpoint for inference. |
| `--agent-config-path` | YAML for agent loop (tools, env, reward). |
| `--num-workers`     | Number of parallel agent sandboxes. |
| `--max-turns`       | Max interaction turns per episode. |
| `--max-samples`     | Cap samples (`-1` = no limit). |
| `--n`               | Rollouts per prompt. |

For more options, run `python examples/agent_interaction/parallel_infer.py --help`.
