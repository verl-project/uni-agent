# Agent Reinforcement Learning

In the previous pages, we focused on inference-time agent behavior: how an agent calls tools, interacts with environments, and solves tasks through multi-turn interaction. A major advantage of Uni-Agent is that the same interaction stack can be connected directly to training engines such as `verl`, so you can move from running agents to training them without any modification.

That is where agent reinforcement learning becomes interesting. Once the agent can already run real rollouts, the next step is to optimize it with large-scale training. This is also where the systems challenge appears: each sample may involve multi-turn reasoning, tool calls, environment interaction, and test execution, and episode latency can vary a lot across tasks.

This page introduces the training scripts under `examples/agent_train`, explains how **agent config** is defined and used, and compares the **synchronous** and **fully asynchronous** training recipes built on top of `verl`.

All runnable scripts referenced in this document live under `examples/agent_train`.

---

## Training Overview

The two launcher scripts are:

| Script | Mode| Best for |
|--------|------|----------|
| `train_sync.sh` | Synchronous | First runs, simpler debugging, predictable update rhythm |
| `train_fully_async.sh` | Fully asynchronous | Large-scale runs, better utilization when episode latency is uneven |

Both scripts train an agent policy with the same overall components:

1. Load prompts from a Parquet dataset.
2. Run multi-turn agent rollouts in parallel sandboxes.
3. Compute rewards from the task outcome.
4. Update the policy with GRPO-style training.
5. Periodically evaluate on the validation set.

Both scripts also share the same idea of **agent configuration**:

- The training script sets high-level trainer, rollout, model, and cluster parameters.
- `AGENT_CONFIG_PATH` points to a YAML file that defines the agent loop itself: interaction limits, sandbox config, tools, and reward settings.
- Sample-specific fields from the dataset, such as environment image or reward metadata, are merged in at runtime.

That separation is important: the shell script controls the **training system**, while the YAML controls the **agent interaction behavior inside each rollout**.

---

## Before You Launch

Both scripts are designed to be launched from the repository root so Ray can package both `verl/` and `uni_agent/`.

The common path variables are:

| Variable | Meaning | Default |
|----------|---------|---------|
| `RAY_DATA_HOME` | Root directory for models, data, checkpoints, and runtime env files | `${HOME}/verl` |
| `MODEL_PATH` | Policy model checkpoint | `${RAY_DATA_HOME}/models/Qwen3-30B-A3B-Instruct-xml-template` |
| `CKPTS_DIR` | Output directory for training logs and checkpoints | `${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}` |
| `TRAIN_FILE` | Training dataset in Parquet format | `${RAY_DATA_HOME}/data/swe_agent/r2e_gym_subset_filtered.parquet` |
| `TEST_FILE` | Validation dataset in Parquet format | `${RAY_DATA_HOME}/data/swe_agent/swe_bench_verified.parquet` |
| `RUNTIME_ENV` | Ray runtime environment YAML | `${RAY_DATA_HOME}/data/swe_agent/runtime_env.yaml` |
| `AGENT_CONFIG_PATH` | Agent loop config YAML | `examples/agent_interaction/agent_config.yaml` |

Prepare the training and validation datasets first:

```bash
export RAY_DATA_HOME=~/verl

# Training set: r2e-gym-subset
python examples/agent_train/prepare_data.py \
    --local-save-dir "${RAY_DATA_HOME}/data/swe_agent"

# Validation set: SWE-bench Verified
python examples/agent_interaction/prepare_data.py \
    --local-save-dir "${RAY_DATA_HOME}/data/swe_agent"
```

This writes:

- `${RAY_DATA_HOME}/data/swe_agent/r2e_gym_subset_filtered.parquet` as `TRAIN_FILE`
- `${RAY_DATA_HOME}/data/swe_agent/swe_bench_verified.parquet` as `TEST_FILE`

---

## Launch Sync/Async Training

Then launch one of the scripts from the repo root:

```bash
bash examples/agent_train/train_sync.sh
```

or:

```bash
bash examples/agent_train/train_fully_async.sh
```

If you use VEFAAS or other remote environments, make sure the required credentials and deployment settings are already present in `runtime_env.yaml` or the job environment. See the environment setup document for the sandbox-side details.


Agent training uses `actor_rollout_ref.rollout.agent.agent_loop_config_path` to locate the YAML file that defines the agent loop. In the default scripts, this path is:

```bash
examples/agent_interaction/agent_config.yaml
```

At runtime, the config is consumed in layers:

1. The training script passes `agent_loop_config_path` into the rollout config.
2. `uni_agent/agent_loop.py` loads the YAML file and reads the first agent definition from it.
3. The trainer injects rollout-side model objects such as the client, tokenizer, and sampling parameters.
4. Per-sample fields from the dataset, especially `tools_kwargs.env` and `tools_kwargs.reward`, are merged into the config before each rollout starts.

This means the YAML defines the **base agent template**, while the dataset can still customize the sandbox image, setup commands, and reward metadata for each sample.

### Annotated agent config

Below is the default config used by the training scripts, with explanations for each section:

```yaml
# examples/agent_interaction/agent_config.yaml

- name: xxx_agent
  # Agent name. This should match the `agent_name` field in the dataset.
  # Each sample uses this name to select the corresponding agent config.

  _target_: uni_agent.agent_loop.UniAgentLoop
  # Agent loop class. Keep this value unless you are replacing the rollout logic.

  concurrency: 512
  # Global concurrency budget inside the agent loop. The implementation divides
  # this across rollout workers to cap how many environments run at once.

  log_dir: /tmp/swebench_qwen3_coder
  # Base directory for per-run logs such as trajectories and interaction results.

  interaction:
    action_timeout: 300
    # Max time for one tool/action call inside a rollout.

    max_turns: 100
    # Max number of model-environment turns per episode.

  env:
    deployment:
      type: vefaas
      command: curl -fsSL https://pjw-test-empty.tos-cn-beijing.ivolces.com/bin/tos_swe_rex.sh | bash -s -- {token}
      timeout: 600
      function_id: s1lwmntk
      function_route: https://sd48oga819ma6giekqtvg.apigateway-cn-beijing.volceapi.com
    env_variables:
      PIP_PROGRESS_BAR: "off"
      PIP_CACHE_DIR: "~/.cache/pip"
      PAGER: "cat"
      MANPAGER: "cat"
      LESS: "-R"
      TQDM_DISABLE: "1"
      GIT_PAGER: "cat"
  # Sandbox template. The dataset can still override pieces such as image and
  # post-setup commands for each sample.

  tools:
    - name: str_replace_editor
    - name: execute_bash
    - name: submit
  # Tools installed into each sandbox before interaction starts.

  reward:
    eval_timeout: 600
  # Base reward config. Dataset-provided reward metadata is merged into this.
```

### What each top-level section means

| Key | Purpose | Typical tuning advice |
|-----|---------|-----------------------|
| `name` | Logical name of the agent loop entry | Usually keep as is unless you define multiple agent types |
| `_target_` | Python class that implements the loop | Change only when extending the framework |
| `concurrency` | Cap on total simultaneous agent episodes | Raise when your backend can sustain more sandboxes |
| `log_dir` | Directory for run logs and cached rollout outputs | Point to fast, large local storage |
| `interaction` | Turn limit and action timeout | Increase `max_turns` for harder tasks; keep timeouts conservative |
| `env` | Sandbox deployment template and env vars | Match your actual deployment backend and credentials |
| `tools` | Tool list exposed to the model | Keep only tools the task truly needs |
| `reward` | Reward-side settings shared by all samples | Most often `eval_timeout` and reward backend options |

### How this differs from trainer config

It helps to separate three config layers:

| Layer | Controlled by | Examples |
|-------|---------------|----------|
| Training system | `train_sync.sh` / `train_fully_async.sh` | node counts, batch sizes, optimizer, rollout engine, parallelism |
| Agent loop | `AGENT_CONFIG_PATH` YAML | tools, sandbox, interaction limits, reward base config |
| Per-sample task data | Dataset row under `extra_info.tools_kwargs` | container image, repo reset command, reward metadata |

If you want to change **how many GPUs or workers** the run uses, edit the shell script. If you want to change **how the agent behaves inside each environment**, edit the YAML. If you want task-specific sandbox or reward details, edit the dataset generation pipeline.
