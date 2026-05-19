# Parallel Agent Interaction

After you can launch a single agent environment, the next step is to run many agent tasks in parallel. In this setting, each sample gets its own sandbox, the model interacts with that sandbox over multiple turns, and Uni-Agent collects the resulting trajectories and rewards.

This page uses a **SWE agent** workflow as the running example. You will prepare SWE-Bench data, run model-environment interaction with multiple workers, and verify the generated solutions.

The inference and verification scripts for this page live under `examples/agent_interaction`.

**Reference results on SWE-Bench Verified with Uni-Agent:**

| **Model**                      | Inference Config                                   | **Uni-Agent** |
| ------------------------------ | -------------------------------------------------- |:-------------:|
| Qwen3-Coder-30B-A3B-Instruct   | temp=0.8, topp=0.9, tp=4, 100 turns, 128K context  | **49.2** (Avg@4) |
| Qwen3-Coder-480B-A35B-Instruct | temp=0.8, topp=0.9, tp=16, 500 turns, 256K context | **64.2** (Avg@4) |
| Qwen3-Coder-Next               | temp=0.8, topp=0.9, tp=16, 300 turns, 128K context | **67.6** (Avg@4) |
| Qwen3.5-4B                     | temp=0.8, topp=0.9, tp=4, 100 turns, 64K context   | **45.2** (Avg@1) |
| Qwen3.5-9B                     | temp=1.0, topp=0.7, tp=4, 100 turns, 64K context   | **53.8** (Avg@1) |
| Qwen3.5-35B-A3B                | temp=1.0, topp=0.7, tp=4, 300 turns, 128K context  | **68.4** (Avg@1) |

`Avg@N` reports the average pass rate over `N` rollouts per task.

---

## Step 1: Prepare the dataset

Start with the dataset. A parallel interaction sample needs the prompt, the sandbox setup, and the reward metadata required for verification.

Use `examples/data_preprocess/swe_bench_verified.py` to fetch [SWE-Bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) and build a Parquet file in the format Uni-Agent expects. Set `DEPLOYMENT` to match the sandbox backend you plan to use, because the preprocessing step writes backend-specific image names. The commands below use Modal as the example backend.

```bash
DEPLOYMENT=modal python examples/data_preprocess/swe_bench_verified.py --local-save-dir ~/data/swe_agent
```

The script writes `~/data/swe_agent/swe_bench_verified_<deployment>.parquet`, for example `~/data/swe_agent/swe_bench_verified_modal.parquet`.

---

## Step 2: Run parallel inference

Once the dataset is ready, use `parallel_infer.py` to run the agent loop over many samples. Uni-Agent loads the model, starts multiple agent workers, creates a sandbox for each active task, and reports the mean reward score.

### Single-Node

```bash
DATA_PATH=~/data/swe_agent/swe_bench_verified_modal.parquet
AGENT_CONFIG=examples/agent_interaction/agent_config_modal.yaml

python examples/agent_interaction/parallel_infer.py \
    --data-path $DATA_PATH \
    --model-path ~/models/Qwen3-Coder-30B-A3B-Instruct \
    --agent-config-path $AGENT_CONFIG \
    --num-workers 8 \
    --max-turns 100 \
    --max-samples 4
```

- `--num-workers`: number of parallel agent environments. Tune this to your GPU resources and sandbox quota.
- `--max-samples`: cap the number of dataset rows to run. Use `-1` for the full dataset.
- `--n`: number of rollouts per prompt.

### Multi-node / Ray job submission

To run on a Ray cluster, submit the same script with `ray job submit` and provide a runtime environment YAML. Put backend credentials in that file, for example `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` for Modal, or `VEFAAS_FUNCTION_ID`, `VEFAAS_FUNCTION_ROUTE`, `VOLCE_ACCESS_KEY`, and `VOLCE_SECRET_KEY` for veFaaS. See `examples/agent_interaction/runtime_env.yaml` for an example.

```bash
ray job submit --no-wait \
    --runtime-env examples/agent_interaction/runtime_env.yaml \
    --working-dir . \
    -- python3 examples/agent_interaction/parallel_infer.py \
    --data-path ~/data/swe_agent/swe_bench_verified_modal.parquet \
    --model-path ~/models/Qwen3-Coder-30B-A3B-Instruct \
    --agent-config-path examples/agent_interaction/agent_config_modal.yaml \
    --nnodes 4 \
    --n-gpus-per-node 8 \
    --max-samples -1
```

Edit `runtime_env.yaml` to set your credentials, and do not commit real secrets.

### Agent config

Uni-Agent groups the environment, tool, and interaction parameters into a single agent config. This example uses `examples/agent_interaction/agent_config_modal.yaml`; use `examples/agent_interaction/agent_config_vefaas.yaml` if you run on veFaaS.

Below is the main shape of the config:

```yaml
# examples/agent_interaction/agent_config_modal.yaml

- name: swe_agent
  _target_: uni_agent.agent_loop.UniAgentLoop
  concurrency: 64
  log_dir: /tmp/swebench_qwen3_coder
  mask_abnormal_exit_traj: false

  interaction:
    action_timeout: 300
    max_turns: 100

  env:
    deployment:
      type: modal
      startup_timeout: 600
      runtime_timeout: 300
      deployment_timeout: 3600
      # If your machine needs a proxy to connect to Modal
      proxy: http://<proxy-host>:<proxy-port>
    env_variables:
      PIP_PROGRESS_BAR: "off"
      PIP_CACHE_DIR: "~/.cache/pip"
      PAGER: "cat"
      MANPAGER: "cat"
      LESS: "-R"
      TQDM_DISABLE: "1"
      GIT_PAGER: "cat"

  tools:
    - name: str_replace_editor
    - name: execute_bash
    - name: submit

  reward:
    eval_timeout: 600
```

- `concurrency` limits the number of in-flight agent loops.
- `interaction` controls per-action timeout and max turns.
- `env.deployment` defines the default sandbox backend. Per-sample fields such as image and post-setup command come from `tools_kwargs.env`. If your machine needs a proxy to connect to Modal, set `proxy` to your proxy address.
- `tools` lists the tools installed into each sandbox and exposed to the model.
- `reward` provides default reward settings. Per-sample reward metadata comes from `tools_kwargs.reward`.
