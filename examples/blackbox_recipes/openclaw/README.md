# OpenClaw recipe

This recipe runs the pinned OpenClaw CLI (`2026.9.2`) inside a task sandbox and sends its model requests through the Uni-Agent gateway. It uses the dataset row for the user prompt, the SWE-bench task for scoring, and a fail-closed SQLite audit to reject fallback, recovery, branching, or incomplete tool traces.

## Tool runtime

The immutable OpenClaw tool image is:

    swr.cn-east-3.myhuaweicloud.com/openyuanrong/openclaw-tool@sha256:90cef4641f82a955692385c48cb4db01a3bd73fc6f68f37962a58717ce547c13

The `tool` image is a scratch image intended for providers that can mount a sidecar at `/opt/openclaw`. The current SWE-bench config uses the Docker provider, which does not mount that image automatically. Each task image used with this config must already contain `/opt/openclaw/bin/openclaw` and the task's Python dependencies. The `runtime` build target can be used as the source for a derived task image:

    BUILD_TARGET=runtime RUNTIME_IMAGE=openclaw-recipe-runtime:2026.9.2 bash examples/blackbox_recipes/openclaw/build_tool.sh

For example, a per-instance SWE-bench image can add the runtime with `COPY --from=openclaw-recipe-runtime:2026.9.2 /opt/openclaw /opt/openclaw`; publish or load that derived image and point the dataset's sandbox image at it.

The task container also needs glibc 2.35 or newer. OpenYuanRong sidecar mounting has not been validated for this recipe.

## SWE-bench inference

Preprocess SWE-bench data with the repository's `uni_agent.tasks.swe_bench` pipeline. The resulting rows provide the task image and metadata; the prompt stays in the row and is injected by the framework at runtime. The YAML only supplies task, sandbox, and agent defaults.

Use an environment with the repository's inference dependencies installed and provide paths explicitly:

    DATA_PATH=/path/to/val.parquet \
    MODEL_PATH=/path/to/model \
    OUTPUT_DIR=/path/to/output \
    bash examples/blackbox_recipes/openclaw/run_infer_openclaw.sh

`LIMIT` defaults to one sample and `N` to one rollout per sample. The script records verifier scores in `result.json` and writes a structural rollout summary to `inference_summary.json`; reward 0 is a normal task result and does not make the launcher fail. Optional `N_GPUS_PER_NODE` and `TENSOR_PARALLEL_SIZE` values are passed through to the inference driver.

The Docker task config relies on Docker's default `pull_policy: missing`, so public task images can be pulled when needed. It currently uses host networking because the gateway endpoint is bound to the Ray node address. This weakens isolation between benchmark code and the host network, so the current config is only suitable for a controlled deployment. A bridge-network route that exposes only the per-session gateway, and automatic tool-runtime injection into each task image, still need a verified implementation before this config is portable to arbitrary sandbox hosts.

## Tests and deployment limits

The focused CPU tests cover prompt-file handoff, CLI errors and timeouts, fallback rejection, and SQLite single-trajectory auditing. They do not validate the real OpenClaw CLI, Docker gateway routing, task image assembly, or GPU rollout. The Docker image must provide the tool runtime at `/opt/openclaw`, and the host-network gateway route is the current tested deployment assumption.
