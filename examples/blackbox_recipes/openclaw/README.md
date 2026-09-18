# OpenClaw recipe

This recipe runs the pinned OpenClaw CLI (`2026.9.2`) inside a task sandbox and sends its model requests through the Uni-Agent gateway. It uses the dataset row for the user prompt, the SWE-bench task for scoring, and a fail-closed SQLite audit to reject fallback, recovery, branching, or incomplete tool traces.

## Tool runtime

The immutable OpenClaw tool image is:

    swr.cn-east-3.myhuaweicloud.com/openyuanrong/openclaw-tool@sha256:90cef4641f82a955692385c48cb4db01a3bd73fc6f68f37962a58717ce547c13

The `tool` image is a scratch image mounted at `/opt/openclaw` by the OpenYuanRong provider in the included SWE-bench config. The task image supplies the benchmark repository and its own Python environment; the agent command prepends `/opt/openclaw/bin` while preserving the task image PATH.

    bash examples/blackbox_recipes/openclaw/build_tool.sh

The task image must be reachable by the provider's image mapping and include the normal SWE-bench dependencies. The sidecar image reference in `config/openclaw_swe_bench.yaml` is pinned by digest; rebuild and publish a new digest when the OpenClaw version changes.

## SWE-bench inference

Preprocess SWE-bench data with the repository's `uni_agent.tasks.swe_bench` pipeline. The resulting rows provide the task image and metadata; the prompt stays in the row and is injected by the framework at runtime. The YAML only supplies task, sandbox, and agent defaults.

Use an environment with the repository's inference dependencies installed, provide the OpenYuanRong credentials through the caller's environment, and provide paths explicitly:

    DATA_PATH=/path/to/val.parquet \
    MODEL_PATH=/path/to/model \
    OUTPUT_DIR=/path/to/output \
    OPENYUANRONG_SERVER_ADDRESS=your-server-address \
    OPENYUANRONG_TOKEN=your-token \
    bash examples/blackbox_recipes/openclaw/run_infer_openclaw.sh

`LIMIT` defaults to one sample and `N` to one rollout per sample. The script records verifier scores in `result.json` and validates the corresponding task logs and finished trajectories; reward 0 is a normal task result and does not make the launcher fail. Optional `N_GPUS_PER_NODE` and `TENSOR_PARALLEL_SIZE` values are passed through to the inference driver.

## Tests and deployment limits

The focused CPU tests cover prompt-file handoff, CLI errors and timeouts, fallback rejection, and SQLite single-trajectory auditing. They do not validate the real OpenClaw CLI, OpenYuanRong service, sidecar registry access, task image mapping, or GPU rollout.
