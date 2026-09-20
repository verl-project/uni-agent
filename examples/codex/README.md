# Codex recipe

This recipe runs Codex as a sandboxed agent through uni-agent's standard task
runner. The Codex CLI runs from the mounted sidecar, reads its prompt from
stdin, and sends requests to the session's native `/v1/responses` endpoint.
The recipe therefore depends on the Responses gateway adapter and its protocol
tests in `uni_agent/gateway`.

## Sidecar

Build the Codex sidecar with a pinned CLI release and the registry used by your
sandbox provider:

```bash
bash examples/codex/build_tool.sh \
  --version 0.147.0 \
  --registry registry.example
```

The task config mounts the image at `/opt/codex`. It provides the task-image
Conda environment path and PATH entries, so the agent does not assume a host's
Conda installation layout. Leave `agent.conda_env_path` unset for a task image
without Conda.

## Inference

Prepare a Parquet dataset in the format consumed by the task runner
(`extra_info.tools_kwargs.task`), then supply its path and the model checkpoint:

```bash
DATA_PATH=/path/to/prepared.parquet \
MODEL_PATH=/path/to/model \
OPENYUANRONG_SERVER_ADDRESS=your-server-address \
OPENYUANRONG_TOKEN=your-token \
bash examples/codex/run_infer_codex.sh
```

The launcher uses `examples/codex/task_config_codex.yaml` and the shared
`examples/inference/parallel_infer_verl.py` entrypoint. It does not convert
legacy Parquet rows; prepare legacy data with the task's preprocessing before
launching. The launcher defaults to one input sample and one rollout (`LIMIT=1`,
`N=1`); override either value for a larger run. `CONCURRENCY`, `RESULT_PATH`,
and the engine sizing options can also be passed as environment variables.
Text-only Qwen3.5 runs enable `LANGUAGE_MODEL_ONLY=true` and
`DISABLE_THINKING=true` by default; set either to `false` for another model.
