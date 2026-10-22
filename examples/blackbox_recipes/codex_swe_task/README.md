# Codex SWE-bench recipe

This recipe runs the Codex CLI as an inference-only sandbox agent. The CLI is
mounted at `/opt/codex`, reads one task prompt from stdin, and calls the
session's native Responses endpoint. Use a checkout that includes the Responses
Gateway merged in [PR #224](https://github.com/verl-project/uni-agent/pull/224).
The Agent invokes the native executable directly. `agent.codex_bin` defaults to
`/opt/codex/vendor/x86_64-unknown-linux-musl/bin/codex`; set it to the full
in-sandbox binary path if you use another sidecar layout. Provider and catalog
settings are passed through Codex's native `-c` overrides, with a separate
temporary `CODEX_HOME` for each episode.

## Prerequisites

Follow the [installation guide](https://uni-agent.readthedocs.io/en/latest/quickstart/installation.html)
for Uni-Agent, the checkout's pinned `verl`, and SWE-bench dependencies. Each Ray
node needs the same inference environment and access to the model and data paths.
Install the OpenYuanRong SDK following the
[sandbox guide](https://uni-agent.readthedocs.io/en/latest/quickstart/launch-sandbox.html).
The recorded service validation used `openyuanrong-sandbox==0.10.2rc9`; select an
SDK version compatible with your sandbox service. The service must be able to
pull both the task image and the sidecar image.

## Build the sidecar

The build script keeps the image name, tag, registry, and Codex version
separate. It builds locally by default and pushes only when `--registry` is
provided:

```bash
bash examples/blackbox_recipes/codex_swe_task/build_tool.sh \
  --tag 0.147.0-pr168-13e1f45-catalog \
  --version 0.147.0 \
  --registry registry.example/namespace
```

Record the resulting digest in
`examples/blackbox_recipes/codex_swe_task/task_config_codex.yaml`. The task
image owns `/testbed` and its environment. In the task YAML's existing `sandbox`
block, configure `sandbox_kwargs` as follows. The Codex process and its tools
inherit this environment:

```yaml
sandbox_kwargs:
  proxy_port: 38197
  env:
    CONDA_DEFAULT_ENV: testbed
    CONDA_PREFIX: /opt/miniconda3/envs/testbed
  add_to_path:
    - /opt/miniconda3/envs/testbed/bin
    - /opt/miniconda3/bin
```

These paths belong to the task image, not the host. `add_to_path` prepends them
while preserving the sandbox's existing `PATH`.

The task config mounts the sidecar with `sandbox.image_mounts`, using `image`
for the image reference and `mount_path` for `/opt/codex`. Provider-specific
settings such as `proxy_port` remain in `sandbox.sandbox_kwargs`.

## Prepare data and model metadata

Prepare SWE-bench Verified from the repository root:

```bash
python -m uni_agent.tasks.swe_bench.preprocess --local-save-dir ~/data/swe_agent
```

This writes `~/data/swe_agent/swe_bench_verified.parquet` with the current
`extra_info.tools_kwargs.task` format. The launcher uses these prepared rows
without converting legacy data. Make the resulting file available to the Ray
workers at the path passed as `DATA_PATH`.

For Qwen3.5-9B, add `model_catalog_path` to the existing `agent` block in the task
YAML, or use a worker-readable copy of the YAML through `TASK_CONFIG`:

```yaml
agent:
  name: codex
  run_timeout: 7200
  model_catalog_path: examples/blackbox_recipes/codex_swe_task/model_catalog_qwen35.json
```

The supplied [catalog](model_catalog_qwen35.json) is for Codex **0.147.0**. Its
`slug` is `Qwen3.5-9B`, which must match the served model name (the model directory
basename in this launcher). The adapter copies the catalog into the sandbox and
passes Codex's `model_catalog_json` override. Its
`supports_parallel_tool_calls: true` matches the Gateway; Codex's fallback for an
unknown model sends `false`, which the Gateway rejects.

The catalog's `context_window` is **69632**, matching the default backend capacity:
4096 prompt tokens plus `RESPONSE_LENGTH=65536`. Codex keeps its own effective-window
reserve within this declaration. If you change the response budget, update the
catalog's context declaration to match. `RESPONSE_LENGTH` is the trajectory
response capacity, not a per-request output limit. For another model or CLI
version, use matching model metadata and instructions.

## Configure the Ray environment

The OpenYuanRong provider requires `OPENYUANRONG_SERVER_ADDRESS` and
`OPENYUANRONG_TOKEN`. Keep their values in a protected file outside the checkout,
readable only by the cluster user (mode `600`), for example:

```bash
OPENYUANRONG_SERVER_ADDRESS="<sandbox-service-address>"
OPENYUANRONG_TOKEN="<sandbox-token>"
```

On **each Ray node**, load that file in the project environment **before starting
that node's Ray processes**, then start Ray using your normal cluster settings:

```bash
set -a
. /absolute/private/openyuanrong.env
set +a
```

Ray workers inherit the node environment. Exporting these variables only in the
job-submission shell does not update an already-running Ray node. Keep actual
credentials out of runtime-env YAML because Ray records the submitted runtime
environment.

Set deployment-specific options in the same protected node environment as needed:

| Variable | Purpose |
|---|---|
| `OPENYUANRONG_TLS` | Sandbox service TLS (`1`/`0`) |
| `OPENYUANRONG_GATEWAY_ADDRESS` | Separate sandbox gateway address |
| `OPENYUANRONG_GATEWAY_TLS` | Gateway TLS/WSS (`1`/`0`) |
| `OPENYUANRONG_TLS_VERIFY` | SDK certificate verification (`1`/`0`) |
| `OPENYUANRONG_TUNNEL_SSL_VERIFY` | Reverse-tunnel certificate verification (`1`/`0`) |

A runtime-env file can contain non-secret settings such as:

```yaml
excludes:
  - "/.git/"
env_vars:
  PYTHONPATH: "verl"
  TOKENIZERS_PARALLELISM: "false"
```

The launcher packages the checkout with `--working-dir .`, including the catalog.
The task YAML, model, data and output paths must also be available to the workers.

## Run inference

Run from the repository root after configuring the task YAML and Ray nodes:

```bash
DATA_PATH=/absolute/data/swe_bench_verified.parquet \
MODEL_PATH=/absolute/models/Qwen3.5-9B \
OUTPUT_DIR=/absolute/new-run-directory \
RAY_RUNTIME_ENV=/absolute/codex-runtime-env.yaml \
RAY_API_SERVER_ADDRESS=http://ray-head.example:8265 \
bash examples/blackbox_recipes/codex_swe_task/run_infer_codex.sh
```

`LIMIT`, `N`, `CONCURRENCY`, and `GATEWAY_COUNT` default to `1`. Hardware and
`RESPONSE_LENGTH` are explicit environment overrides. `LANGUAGE_MODEL_ONLY` and
`DISABLE_THINKING` accept `1/0` or `true/false` and default to enabled for the
text-only Qwen setup.

The task YAML sets both the agent timeout and sandbox lifetime to 7200 seconds;
the sandbox lifetime also covers task verification. This launcher does not set a
framework session timeout. It does not disable Codex's native text compaction:
long runs may replace history and produce additional trajectory segments. The
Gateway rejects opaque `compaction`/`compaction_summary` items. Keep the catalog,
backend capacity and task time budgets consistent with the intended run.

`OUTPUT_DIR/result.json` stores scores and `OUTPUT_DIR/logs/` stores task and
trajectory artifacts. The launcher rejects pre-existing results/logs and requires
the expected number of scored sessions, with one finished trajectory per session.
A compacted run that produces multiple segments will fail that check. Reward 0
is a valid inference result; strict single-task SWE acceptance separately requires
reward `1.0`, verifier `resolved=True`, one session and exactly one raw trajectory
with `finished=true`. When the dataset has fewer rows than `LIMIT`, the expected
session count uses the actual `num_prompts` multiplied by `N`.

The Codex adapter accepts an optional leading system message for framework
compatibility but passes only the single user task to the CLI. Put effective task
instructions in the user message.

## Scope and tests

The recipe contains the Codex agent adapter, its sidecar image, task YAML, and
the shared inference entrypoint. It has no training entrypoint or
machine-specific credential file. Focused agent tests are in
`tests/uni_agent/agents/test_codex_agent.py`. Launcher regression tests are in
`tests/uni_agent/tasks/test_codex_infer_launcher.py`; they exercise the real
shell script with simulated Ray results. Both use CPU and level-0 test markers.
