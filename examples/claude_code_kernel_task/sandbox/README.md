# Ascend Triton Agent sandbox layer

Build files for the evaluator image used by [the recipe](../README.md):

- Local `template/` and `tools/` files are combined with pinned upstream assets and `rl-adaptations.patch` in a temporary build context.
- The assembled `template/` becomes `/opt/triton-agent-template`, copied into each session workspace; `tools/` becomes `/opt/triton-agent-tools`, linked by the workspace verifier.
- Task execution, reward calculation and runtime-installed hooks remain in `uni_agent/tasks/kernel_bench/`.

## Skills source

The skills and reference material come from [CANNBot Skills PR #205](https://gitcode.com/cann/cannbot-skills/pull/205), with local RL adaptations: prepared-task instructions, compact verifier feedback, a unified verification/benchmark entry point, and integration with best-snapshot tracking and early stopping. They are not an unmodified upstream copy.

`build_image.sh` pins upstream commit `49451c99df99d0dc31f2a8acb920e46e1ee52105`, extracts the required skills/references and verifier scripts, and applies `rl-adaptations.patch`. Upstream material remains subject to its original licence.

## Build

The base image must already contain Claude Code, Python, Bash, `timeout`, sudo/visudo, Ascend/CANN, torch-npu, Triton Ascend, verifier dependencies and the non-root `claude` user. The verifier expects `/usr/local/Ascend/cann/set_env.sh`. The build host needs Git, tar and access to GitCode (or a local upstream checkout, below). The Dockerfile only copies the assembled files and configures permissions; it does not download packages or create users.

```bash
cd examples/claude_code_kernel_task/sandbox
DOCKER_HOST=ssh://root@npu-host-01 \
BASE_IMAGE='<your-prepared-ascend-claude-image>:<tag>' \
bash build_image.sh
```

Replace the placeholder with your prepared base image; `BASE_IMAGE` is required and has no default. The output defaults to `triton-claude-code-env:latest`, matching the task YAML. If you set a different `OUTPUT_IMAGE`, update the YAML too. Use a distinct base-image tag to avoid overwriting it. Set `SANDBOX_USER` if the existing user has a different name.

For offline or repeated builds, set `CANNBOT_SOURCE` to a local Git repository containing the pinned commit:

```bash
# Fetch once on a connected machine; copy this repository to the build host if needed.
git init cannbot-skills
git -C cannbot-skills fetch --depth=1 https://gitcode.com/cann/cannbot-skills.git \
  49451c99df99d0dc31f2a8acb920e46e1ee52105
CANNBOT_SOURCE="$PWD/cannbot-skills" \
BASE_IMAGE='<your-prepared-ascend-claude-image>:<tag>' \
bash build_image.sh
```

The script reads the pinned Git objects, not the checkout's working files. Always build through `build_image.sh`, rather than `docker build .`; generated assets are temporary and are not written into the recipe. No network access is needed inside the sandbox.

## Verify

Check the image layout and sudo policy:

```bash
docker --host ssh://root@npu-host-01 run --rm --entrypoint bash \
  triton-claude-code-env:latest -lc \
  'test "$(id -u)" != 0 && sudo -n -l /opt/triton-agent-tools/verify_once.sh smoke >/dev/null && test -d /opt/triton-agent-template'
```

For actual NPU verification, use the complete device/driver mounts and host-local locks described in the parent README. In a prepared session container, with no concurrent verifier running:

```bash
docker --host ssh://root@npu-host-01 exec -w /workspace CONTAINER \
  bash tools/verify_once.sh OP_NAME
```

The Task prepares the operator, implementation and case files. Check `output/verify/verify_result_summary.json` for the verification summary and `verify_result.raw.log` for detailed errors. The wrapper benchmarks fully correct candidates and saves the best snapshot; final Task collection reads those artifacts.
