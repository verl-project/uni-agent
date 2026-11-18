# Hermes black-box recipe

Runs the pinned [Hermes Agent](https://github.com/NousResearch/hermes-agent)
(`5eb99eb2844b22ebb723711b8e6a0bbb80bb5f04`) inside a task sandbox. The host
adapter passes the framework's messages and session endpoint to the sidecar.
SWE-bench owns verification; the framework owns rewards and token trajectories.
Only terminal/file tools are enabled. Each episode has an isolated Hermes home;
memory, delegation, fallback providers and context compression are disabled.

## Build and configure

```bash
bash examples/blackbox_recipes/hermes_swe_task/build_tool.sh --tag my-hermes-tag
# Optional: add --registry <registry/namespace> to publish the built image.
```

The image is a filesystem sidecar, mounted at `/opt/hermes` through
`sandbox.image_mounts` using the `image` and `mount_path` fields. Provider tunnel
settings remain under `sandbox.sandbox_kwargs`. Update
`task_config_hermes.yaml` with the image reference you built, preferably a digest,
and a task image mapping accessible to your sandbox service. The included image
reference records the original build; it is not a guarantee of registry access.
The task image must already contain the `testbed` environment. The sandbox selects
it through `sandbox.sandbox_kwargs.env`: `CONDA_DEFAULT_ENV=testbed` and
`CONDA_PREFIX=/opt/miniconda3/envs/testbed`. Its `add_to_path` list prepends
`/opt/miniconda3/envs/testbed/bin` and `/opt/miniconda3/bin` to the task image's
existing PATH, so task commands use the preinstalled interpreter and tools.
These sandbox settings do not download, create or install an environment. For a
different task image, adjust or omit these settings together in
[`task_config_hermes.yaml`](task_config_hermes.yaml). Hermes itself continues to
run with the sidecar's absolute Python path.

The recipe keeps
`agent.sampling_params_override` because its `temperature`, `top_p`, and
`max_tokens_per_turn` values define the Hermes request/budget policy. The
standard sidecar uses the default `tool_python`, `runner_script`, and
`approval_mode` values; they are intentionally omitted from the sample task
YAML. The defaults point to `/opt/hermes/bin/python`,
`/opt/hermes/bin/run_hermes.py`, and unattended approval respectively. Custom
sidecar layouts may override the path fields.
Use the current `openyuanrong-sandbox` / `yr_sandbox` provider supported by the
repository; this recipe does not include the legacy SDK bridge.

## Prepare the inference environment and data

Follow the [installation guide](../../../docs/source/quickstart/installation.md)
in a dedicated Python environment. This entrypoint needs a compatible engine
environment with PyTorch, vLLM or SGLang, Ray Jobs and TransferQueue.
Use the `verl` commit pinned by this checkout;
do not replace it with an unrelated release branch. From the repository root:

```bash
git submodule update --init --recursive
python -m pip install -e .
python -m pip install --no-deps -e ./verl
python -m pip install swebench==4.1.0 openyuanrong-sandbox
```

The `--no-deps` command registers the pinned `verl` source; it assumes its engine
dependencies are already installed. Install a sandbox SDK version compatible
with your service on every execution node; see the
[OpenYuanrong provider setup](../../../docs/source/quickstart/launch-sandbox.md).
Hermes and its Python dependencies live in the sidecar, not this host environment.

Prepare one SWE-bench Verified sample for a smoke run:

```bash
python -m uni_agent.tasks.swe_bench.preprocess \
    --local-save-dir "$HOME/data/hermes" --max-instances 1
```

This writes `$HOME/data/hermes/swe_bench_verified.parquet`. Omit
`--max-instances 1` to prepare the full dataset. Each row contains
`extra_info.tools_kwargs.task`, including the task name, metadata and canonical
`sandbox.image`; `sandbox.image_map` in the task YAML maps that image to your
service's registry. The sandbox service must be able to pull both the mapped
SWE image and the Hermes sidecar image.

## Configure the provider and Ray

The sandbox SDK reads `OPENYUANRONG_SERVER_ADDRESS` and `OPENYUANRONG_TOKEN`
from the environment of the Ray process that creates the sandbox. Store these
shell assignments in a protected file outside the checkout, for example
`$HOME/.config/uni-agent/openyuanrong.env`, using your service's values. Configure
`OPENYUANRONG_TLS`, `OPENYUANRONG_GATEWAY_ADDRESS`, `OPENYUANRONG_GATEWAY_TLS`,
`OPENYUANRONG_TLS_VERIFY` and `OPENYUANRONG_TUNNEL_SSL_VERIFY` there only as required
by your deployment. The gateway session model endpoint and key are injected by
Uni-Agent; do not add them to the task YAML.

Load the provider environment **before starting Ray on the head and worker nodes**.
Exporting variables only in the submitting shell does not configure an already
running remote Ray cluster. For an existing cluster, use its administrator's
secret injection mechanism. The following is a single-node example for a
dedicated eight-GPU cluster; adjust the GPU count and tensor parallelism to your
hardware:

```bash
# Activate the prepared inference environment first.
set +x
chmod 600 "$HOME/.config/uni-agent/openyuanrong.env"
set -a
. "$HOME/.config/uni-agent/openyuanrong.env"
set +a
export N_GPUS_PER_NODE=8 TENSOR_PARALLEL_SIZE=8
ray start --head --num-gpus="${N_GPUS_PER_NODE}" \
    --dashboard-host=127.0.0.1 --dashboard-port=8265
```

Ray workers inherit the cluster environment. Keep all Ray ports on a trusted
network and the Jobs API private; see [Ray security](https://docs.ray.io/en/latest/ray-security/index.html)
and [cluster startup options](https://docs.ray.io/en/latest/cluster/cli.html#ray-start).
For a remote head, use the deployment's protected Jobs API address instead of
the loopback address below.

Create a minimal runtime-env file outside the uploaded checkout. It contains
only non-secret job settings; the provider credentials remain in the node
environment:

```bash
umask 077
mkdir -p "$HOME/.config/uni-agent"
cat > "$HOME/.config/uni-agent/hermes-runtime-env.yaml" <<'YAML'
env_vars:
  PYTHONPATH: "verl"
YAML
chmod 600 "$HOME/.config/uni-agent/hermes-runtime-env.yaml"
```

The launcher supplies `--working-dir .` to Ray Jobs. The runtime environment
applies to the job driver and its tasks/actors; see
[Ray environment inheritance](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environment-specified-by-both-job-and-driver).
Do not put credentials in this YAML, the task config or the uploaded directory.

## Inference

Run the launcher from the prepared environment. Data and model paths must be
available to the execution workers. `OUTPUT_DIR`, `RESULT_PATH` (default:
`$OUTPUT_DIR/result.json`) and `LOG_DIR` (default: `$OUTPUT_DIR/logs`) must use
shared storage mounted at the same absolute paths on the submitting host, Ray
job driver and workers. The launcher reads and validates the result and
trajectory files on the submitting host after the Ray job finishes. This
launcher requires shared storage; deployments without it must retrieve the
artifacts before running local validation.

```bash
DATA_PATH="$HOME/data/hermes/swe_bench_verified.parquet" \
MODEL_PATH=/absolute/model \
OUTPUT_DIR=/absolute/new-run-directory \
RAY_RUNTIME_ENV="$HOME/.config/uni-agent/hermes-runtime-env.yaml" \
RAY_API_SERVER_ADDRESS=http://127.0.0.1:8265 \
LANGUAGE_MODEL_ONLY=1 \
bash examples/blackbox_recipes/hermes_swe_task/run_infer_hermes.sh
```

Set `TOOL_PARSER` for the policy model; its default is `qwen3_coder`. For this
Qwen3.5-oriented text-only recipe, `LANGUAGE_MODEL_ONLY=1` is the default; set
it to `0` only for a model/engine that does not support the option. Configure `NNODES`,
`N_GPUS_PER_NODE` and `TENSOR_PARALLEL_SIZE` for available hardware (defaults: 1).
`LIMIT`, `N`, `CONCURRENCY` and `GATEWAY_COUNT` default to 1. `TASK_CONFIG` can
override the recipe config. `RAY_API_SERVER_ADDRESS` is required and must be
the Ray Jobs API address reachable from the submitting host; the launcher does
not assume a localhost/default port.
The launcher waits for submission, rejects stale result/log files, and checks
the scored-session count plus one finished trajectory entry per session. It
returns nonzero if inference fails or the result is incomplete.

## Episode budgets

The task YAML owns the agent and sandbox limits; the launcher owns the rollout
capacity. Defaults are:

| Setting | Default | Scope |
|---|---|---|
| `agent.max_iterations` | `100` | Hermes loop iteration limit |
| `agent.terminal_timeout` | `600 s` | One terminal command |
| `agent.run_timeout` | `7200 s` | Host-side timeout; Hermes receives `run_timeout - 600 s` to finish cleanly |
| `eval_timeout` | `900 s` | SWE-bench verification |
| `sandbox.runtime_timeout` | `10800 s` | Sandbox lifetime, including execution and verification |
| `agent.sampling_params_override.max_tokens_per_turn` | `8192` | Per-request output cap, passed as `limits.max_tokens` to Hermes and as HTTP `max_tokens` for the Qwen/custom Chat Completions route |
| `PROMPT_LENGTH` | `4096` | Rollout/data prompt budget; contributes to total trajectory capacity |
| `RESPONSE_LENGTH` | `65536` | Rollout continuation capacity, not a per-request generation cap |

The adapter reserves 600 seconds inside `agent.run_timeout` for Hermes to emit a
structured final result before the host stops the runner. Keep the runner timeout
within the sandbox lifetime, with enough room for startup, verification and cleanup.
Terminal commands run within the agent episode; the terminal and agent budgets
are nested limits, not durations to add together. This inference entrypoint does
not set a separate framework session timeout. The iteration limit counts Hermes
loop iterations; a final summary request can follow exhaustion, so it is not an
exact HTTP request-count limit. Exhausting a Hermes iteration/time budget remains
unfinished; increasing the sandbox lifetime alone does not extend those inner limits.

The shared inference entrypoint sets `max_model_len` and the Gateway's total
trajectory capacity to `PROMPT_LENGTH + RESPONSE_LENGTH` (`69632` by default).
Before each model call, the Gateway clamps `max_tokens` to the smaller of the
requested output cap and the remaining trajectory capacity after the encoded
context. Prior model replies and tool messages therefore consume capacity;
`8192` is not granted afresh regardless of history. Context compression is
disabled in this recipe.

## Completion

`finished=True` means Hermes returned an unambiguous completed response; it does
not mean SWE-bench is solved. Budget exhaustion, timeout and malformed results
remain unfinished. Ordinary inference may legitimately return reward 0.

The launcher records the inference result and framework logs. It does not turn a
task reward into a launcher success/failure gate; inspect the task result and
framework trajectory for the run you care about. A single-task acceptance run
must separately verify one scored session, one raw trajectory entry,
`finished=true`, and the task reward. Non-empty failure summaries are redacted and
length-limited before being included in `TaskResult.extra_info.agent_error`.
Raw Hermes logs stay inside the sandbox.
