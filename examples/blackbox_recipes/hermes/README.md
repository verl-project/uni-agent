# Hermes black-box recipe

Runs the pinned [Hermes Agent](https://github.com/NousResearch/hermes-agent)
(`5eb99eb2844b22ebb723711b8e6a0bbb80bb5f04`) inside a task sandbox. The host
adapter passes the framework's messages and session endpoint to the sidecar.
SWE-bench owns verification; the framework owns rewards and token trajectories.
Only terminal/file tools are enabled. Each episode has an isolated Hermes home;
memory, delegation, fallback providers and context compression are disabled.

## Build and configure

```bash
bash examples/blackbox_recipes/hermes/build_tool.sh --tag <your-tag>
# Optional: add --registry <registry/namespace> to publish the built image.
```

The image is a filesystem sidecar, mounted at `/opt/hermes`. Update
`task_config_hermes.yaml` with the image reference you built, preferably a digest,
and a task image mapping accessible to your sandbox service. The included image
reference records the original build; it is not a guarantee of registry access.
Set `agent.conda_env_path` to the absolute Conda environment path in your task
image, or omit it to inherit PATH. Tool Python and runner paths are configurable.
Use the current `openyuanrong-sandbox` / `yr_sandbox` provider supported by the
repository; this recipe does not include the legacy SDK bridge.

## Inference

Install Uni-Agent and the repository's inference dependencies, including the
pinned `verl` submodule. Use the standard SWE-bench preprocessor documented by
the task; input rows must already contain `extra_info.tools_kwargs.task`.
Start/configure Ray separately. Provide a protected runtime-env YAML outside
the checkout with `PYTHONPATH: verl` under `env_vars`, plus the provider's
`OPENYUANRONG_SERVER_ADDRESS` and `OPENYUANRONG_TOKEN`. Ray applies these to the
job driver and workers. Protect Ray's control plane and do not check credentials
into source control or place that file in the uploaded working directory.
Data, model and output paths must be available to the execution workers.

```bash
DATA_PATH=/absolute/preprocessed-swe-bench.parquet \
MODEL_PATH=/absolute/model \
OUTPUT_DIR=/absolute/new-run-directory \
RAY_RUNTIME_ENV=/absolute/protected-runtime-env.yaml \
LANGUAGE_MODEL_ONLY=1 \
bash examples/blackbox_recipes/hermes/run_infer_hermes.sh
```

Set `TOOL_PARSER` for the policy model; its default is `qwen3_coder`. For Qwen3.5
text-only inference use `LANGUAGE_MODEL_ONLY=1`. Configure `NNODES`,
`N_GPUS_PER_NODE` and `TENSOR_PARALLEL_SIZE` for available hardware (defaults: 1).
`LIMIT`, `N`, `CONCURRENCY` and `GATEWAY_COUNT` default to 1. `TASK_CONFIG` and
`RAY_API_SERVER_ADDRESS` can override the recipe and Ray endpoint.
The launcher waits for submission and returns nonzero if inference fails.

## Completion

`finished=True` means Hermes returned an unambiguous completed response; it does
not mean SWE-bench is solved. Budget exhaustion, timeout and malformed results
remain unfinished. Ordinary inference may legitimately return reward 0.

The launcher records the inference result and framework logs. It does not turn a
task reward into a launcher success/failure gate; inspect the task result and
framework trajectory for the run you care about. Hermes-local diagnostics stay
inside the sandbox.
