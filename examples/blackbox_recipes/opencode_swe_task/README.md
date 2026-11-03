# OpenCode SWE-rebench recipe (E2B)

This recipe trains a policy with the OpenCode CLI as the in-sandbox agent on
SWE-rebench tasks. Sandboxes come from an E2B-compatible backend. OpenCode
**1.19.3** is mounted read-only at `/opt/opencode` from the sidecar image in
[`examples/sidecar/opencode`](../../sidecar/opencode). The OpenCode adapter is
described in [`examples/opencode`](../../opencode/README.md).

| File | Purpose |
|---|---|
| `task_config_opencode_e2b.yaml` | Task defaults: E2B sandbox, image map, sidecar mount, agent, prompt |
| `run_train.sh` | Megatron + V1 async GRPO launcher (Ascend NPU defaults) |

## How a rollout runs

1. `run_task` resolves the sample's task against the task config and binds
   `agent.model.base_url` to the rollout's Gateway session
   (`http://<ray-node-ip>:<port>/sessions/<id>/v1`).
2. `E2BSandbox` creates a sandbox from the mapped task image. With
   `image_as_template: true` the mapped image reference is passed as the E2B
   template ID; no template is built through the E2B Template API.
3. With `backend_image_mounts: true` the backend provides the declared
   `image_mounts`. Startup only checks that `/opt/opencode` exists and fails the
   rollout if it does not. Set `image_mounts_metadata_key` to also send the
   declared mounts to the backend as JSON sandbox metadata.
4. OpenCode runs `run --pure --format json` in `/testbed` and streams Chat
   Completions requests to the session URL. There is no reverse tunnel: the
   sandbox network must reach every Ray node IP on the Gateway's ports.
5. The SWE-rebench verifier runs the FAIL_TO_PASS/PASS_TO_PASS tests in the same
   sandbox (`eval_timeout: 900`) and returns reward 1 or 0.

## Prepare images

Build the sidecar on an x86_64 host and push it next to the task images:

```bash
docker build --platform linux/amd64 --target tool \
  -t <registry>/swe-rebench/opencode-tool:1.19.3 examples/sidecar/opencode
docker push <registry>/swe-rebench/opencode-tool:1.19.3
```

Push the SWE-rebench task images (`swerebench/sweb.eval.x86_64.<tag>`) to the
same registry, then edit the two placeholders in the task config:

- `image_map[0].to`: where the task images live, or the backend's template names.
- `image_mounts[0].image`: the sidecar reference.

The task images default `PATH` to `/opt/conda/bin`. The task config prepends
`/opt/miniconda3/envs/testbed/bin` through `sandbox_kwargs.envs`, so OpenCode's
shell tool and the verifier use the testbed environment.

## Prepare data

The preprocessor accepts local parquet files, so the cluster does not need
Hugging Face access:

```bash
python -m uni_agent.tasks.swe_rebench.preprocess \
  --data-files /data/swe-rebench/filtered-00000-of-00001.parquet \
  --local-save-dir /data/uni_agent \
  [--instance-ids EmilStenstrom__conllu-64,NOhs__dkey-18 --output-name swe_rebench_2task.parquet]
```

`--instance-ids` keeps the given order and fails if an ID is missing or duplicated
in the dataset. An explicitly empty selection or file list is rejected.

Dynamic reward-group filtering defaults to disabled (`FILTER_GROUPS=False`) so
uniform rewards, including an all-failing smoke dataset, cannot cause repeated
generation without a training batch. Enable it only with a suitable dataset and
a retry limit supported by the deployed trainer.

## Configure the sandbox backend

The e2b SDK reads its connection settings from the environment of each rollout
worker. `run_train.sh` forwards every non-empty variable below through verl's
`ray.init` runtime environment, because `ray job submit` does not pass the
submitting shell's environment to the driver.

| Variable | Purpose |
|---|---|
| `E2B_API_KEY` | Required API key |
| `E2B_DOMAIN` | Backend domain (`api.<domain>`, `<port>-<id>.<domain>`) |
| `E2B_API_URL` / `E2B_SANDBOX_URL` | Explicit control-plane and sandbox URLs |
| `E2B_DEBUG` / `E2B_HTTP_VERSION` | Optional SDK settings |

Ray records the submitted runtime environment, so treat job metadata as
sensitive. `PRINT_CMD_ONLY=1` prints the command with the key redacted.

## Train

Start Ray on every node, then run from the repository root:

```bash
E2B_API_KEY=... E2B_DOMAIN=... \
MODEL_PATH=/models/Qwen3.6-35B-A3B \
TRAIN_DATA=/data/uni_agent/swe_rebench_filtered.parquet \
bash examples/blackbox_recipes/opencode_swe_task/run_train.sh
```

The launcher defaults to `colocate_async` on 2 nodes with 16 NPUs each. All
sizes are environment overrides (`NNODES`, `N_GPUS_PER_NODE`, `GEN_TP`,
`TRAIN_TP/PP/CP/EP/ETP`, `TRAIN_BATCH_SIZE`, `N`, `PPO_MINI_BATCH_SIZE`); extra
arguments are appended as Hydra overrides. Keep
`PROMPT_LENGTH + RESPONSE_LENGTH` at least `agent.context_limit` (65536).
`SESSION_TIMEOUT_SECONDS` (5400) covers `agent.run_timeout` (3600) plus
verification and sandbox startup.

For a two-task smoke run, keep the default dynamic-sampling filter disabled so a
batch where every rollout gets the same reward does not stall:

```bash
TRAIN_DATA=/data/uni_agent/swe_rebench_2task.parquet \
TRAIN_BATCH_SIZE=2 PPO_MINI_BATCH_SIZE=2 N=4 FILTER_GROUPS=False \
TOTAL_TRAINING_STEPS=2 SAVE_FREQ=-1 \
bash examples/blackbox_recipes/opencode_swe_task/run_train.sh
```

## Validation status

- CPU tests: adapter (`tests/uni_agent/agents/test_opencode_agent.py`) and E2B
  options (`tests/uni_agent/sandbox/test_e2b_backend_options.py`).
- Real OpenCode 1.18.18 and 1.19.3 CLIs: CLI probe and task smoke through
  LocalSandbox.
- Local Docker, two SWE-rebench tasks with the 1.19.3 sidecar mounted: dataset gold
  patch (reward 1), OpenCode applying the gold patch through its bash tool
  against a scripted endpoint (reward 1), and OpenCode finishing without edits
  (reward 0).
- Not yet validated: the internal E2B backend, a real policy through the
  Gateway, and NPU training with this launcher.
