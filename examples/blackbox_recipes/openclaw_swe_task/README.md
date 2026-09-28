# OpenClaw SWE-bench recipe

This recipe runs the pinned OpenClaw CLI inside a task sandbox and sends its
model requests through the Uni-Agent gateway. OpenClaw keeps its native JSON
and SQLite trajectory audit; the framework owns the task reward. The recipe is
inference-only and does not add a training entrypoint.

## Build the sidecar

The build script uses separate image name, tag, registry, and OpenClaw version
arguments. It builds locally by default and publishes only with an explicit
registry:

```bash
bash examples/blackbox_recipes/openclaw_swe_task/build_tool.sh \
  --tag 2026.9.2 \
  --version 2026.9.2 \
  --registry registry.example/namespace
```

The sidecar is mounted at `/opt/openclaw`. Record the published digest in
`task_config_openclaw.yaml`; the task image still owns `/testbed` and its own
Python environment.

## Run inference

Prepare SWE-bench rows with the repository preprocessor. The prompt comes from
the row at runtime; the YAML contains task and sandbox defaults only. Submit a
Ray Job with a protected runtime-env file and an explicit Ray Jobs address:

```bash
DATA_PATH=/absolute/preprocessed-swe-bench.parquet \
MODEL_PATH=/absolute/model \
OUTPUT_DIR=/absolute/new-run-directory \
RAY_RUNTIME_ENV=/absolute/protected-runtime-env.yaml \
RAY_API_SERVER_ADDRESS=http://ray-head.example:8265 \
bash examples/blackbox_recipes/openclaw_swe_task/run_infer_openclaw.sh
```

`LIMIT`, `N`, `CONCURRENCY`, and `GATEWAY_COUNT` default to `1`; model and
hardware settings are explicit environment overrides. `OUTPUT_DIR/result.json`
contains framework scores and `OUTPUT_DIR/logs/` contains task logs and
trajectories. A result of reward 0 is valid ordinary inference. For a
single-task acceptance run, verify one scored session, one raw framework
trajectory with `finished=true`, and the OpenClaw SQLite audit separately.

The agent accepts an optional system message followed by exactly one user
message and writes the combined prompt to a private episode file. Credentials
and diagnostics are redacted before they are returned to the framework.

## Scope and tests

The recipe contains the OpenClaw adapter, its sidecar image, task YAML, and the
shared inference entrypoint. Focused agent tests are in
`tests/uni_agent/agents/test_openclaw_agent.py` with the repository's CPU and
level-0 markers. They use a fake sandbox and do not claim to test registry
access, a real OpenYuanRong service, or a GPU rollout.
