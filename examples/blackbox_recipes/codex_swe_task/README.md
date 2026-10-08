# Codex SWE-bench recipe

This recipe runs the Codex CLI as an inference-only sandbox agent. The CLI is
mounted at `/opt/codex`, reads one task prompt from stdin, and calls the
session's native Responses endpoint. The Responses gateway implementation is
tracked separately in PR #224; this recipe does not add gateway code.

## Build the sidecar

The build script keeps the image name, tag, registry, and Codex version
separate. It builds locally by default and pushes only when `--registry` is
provided:

```bash
bash examples/blackbox_recipes/codex_swe_task/build_tool.sh \
  --tag 0.147.0-direct-stdin \
  --version 0.147.0 \
  --registry registry.example/namespace
```

Record the resulting digest in
`examples/blackbox_recipes/codex_swe_task/task_config_codex.yaml`. The task
image owns `/testbed` and its Conda environment; `agent.conda_env_path` and
`agent.path` describe that task image and do not assume a host installation.

## Run inference

Input rows must already be prepared by the repository's SWE-bench preprocessor
(`extra_info.tools_kwargs.task`). The launcher submits a Ray Job, so the Ray
Jobs address and a protected runtime-env file are explicit inputs. The runtime
environment must make the checkout, `verl`, provider settings, and the task
dependencies available to the driver and workers.

```bash
DATA_PATH=/absolute/preprocessed-swe-bench.parquet \
MODEL_PATH=/absolute/model \
OUTPUT_DIR=/absolute/new-run-directory \
RAY_RUNTIME_ENV=/absolute/protected-runtime-env.yaml \
RAY_API_SERVER_ADDRESS=http://ray-head.example:8265 \
bash examples/blackbox_recipes/codex_swe_task/run_infer_codex.sh
```

`LIMIT`, `N`, `CONCURRENCY`, and `GATEWAY_COUNT` default to `1` in this
launcher; the shared inference entrypoint retains its upstream defaults.
Hardware and `RESPONSE_LENGTH` are explicit environment overrides; the prompt
budget uses the shared 4096-token default. `LANGUAGE_MODEL_ONLY`
and `DISABLE_THINKING` accept `1/0` or `true/false` and default to enabled for
the text-only Qwen setup. `OUTPUT_DIR/result.json` stores scores and
`OUTPUT_DIR/logs/` stores framework/task trajectories. The launcher rejects a
pre-existing result or log file and checks that the expected number of sessions
was scored with one finished trajectory entry per session; reward 0 remains a
valid inference result. `LIMIT` is an upper bound on selected rows: when the
dataset has fewer rows, the expected session count is the actual `num_prompts`
reported in the result multiplied by the rollout count.

The launcher does not claim that `finished` or reward means SWE-bench solved.
For a single-task acceptance run, inspect the result and the raw trajectory in
`logs/` and verify one session, one trajectory entry, `finished=true`, and the
task reward separately.

The Codex adapter follows the mini-swe-agent input contract: it accepts an
optional leading system message but passes only the single user task to the
CLI. Put effective instructions in the user message.

## Scope and tests

The recipe contains the Codex agent adapter, its sidecar image, task YAML, and
the shared inference entrypoint. It has no training entrypoint or
machine-specific credential file. Focused agent tests are in
`tests/uni_agent/agents/test_codex_agent.py`. Launcher regression tests are in
`tests/uni_agent/tasks/test_codex_infer_launcher.py`; they exercise the real
shell script with simulated Ray results. Both use CPU and level-0 test markers.
