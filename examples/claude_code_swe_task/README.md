# Claude Code SWE Training Recipe

This example trains Qwen3.5-9B with GRPO while Claude Code solves SWE-bench
and SWE-reBench tasks inside OpenYuanrong sandboxes. The Claude Code executable
is supplied by a sidecar image and linked into the sandbox by
`sandbox.startup_commands`.

## Files

- `Dockerfile.claude-code-tool`: builds the Claude Code sidecar image.
- `build_tool.sh`: builds and optionally pushes the sidecar image.
- `task_config_claude_code_openyuanrong.yaml`: configures the agent and the
  OpenYuanrong sandboxes.
- `run_train.sh`: launches the two-node Ascend Qwen3.5-9B GRPO recipe.

## Build the tool image

```bash
bash examples/claude_code_swe_task/build_tool.sh \
    --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
```

Keep the pushed image tag synchronized with
`sandbox.sandbox_kwargs.mounts[].image_url` in the task configuration.

## Launch training

Start the two-node Ray cluster first. The default recipe expects 16 NPUs per
node. Then run the launcher from the repository root:

```bash
DATA_DIR=/path/to/shared/data \
RUNTIME_DIR=/path/to/shared/runtime \
bash examples/claude_code_swe_task/run_train.sh
```

`DATA_DIR` supplies the default model and dataset paths. `RUNTIME_DIR` supplies
the Ray runtime environment, checkpoint directory, and agent log directory.
All paths can also be overridden directly with `MODEL_PATH`, `TRAIN_FILE`,
`TEST_FILE`, `RUNTIME_ENV`, `CKPTS_DIR`, and `AGENT_LOG_DIR`.

The Ray runtime environment must provide the OpenYuanrong client dependencies
and credentials required by the sandbox provider. See
`task_config_claude_code_openyuanrong.yaml` for the sandbox images, mounted tool
image, startup command, and agent timeouts.

The launcher defaults to the validated two-node configuration: 64 prompts per
step, 8 responses per prompt, TP2/PP2/CP8 for training, and TP2 rollout replicas.
These defaults remain overridable through the environment variables defined at
the top of `run_train.sh`.
