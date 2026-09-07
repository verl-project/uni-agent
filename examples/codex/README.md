# Codex Qwen3.5 recipe

This directory adds the Codex black-box agent to the existing `uni_agent`
framework. A runner uses the mounted sidecar at `/opt/codex/bin/run_agent.sh`
and talks to the session's native Responses endpoint:

```text
verl/main_ppo -> AgentFrameworkRolloutAdapter -> run_task -> CodexAgent
  -> /opt/codex/bin/run_agent.sh -> /v1/responses -> typed runner reward
```

## Training

The only Codex training entry point is the community-shaped launcher below. It
uses `ray job submit --no-wait` and `python3 -m verl.trainer.main_ppo`; it does
not create a local Ray cluster or provide a second launcher.

```bash
DATA_DIR=/home/zxh/data \
RUNTIME_DIR=/home/zxh/runtime \
NNODES=1 NGPUS_PER_NODE=8 \
CONCURRENCY=1 GEN_TP=8 TP=8 PP=1 CP=1 \
TRAIN_PROMPT_BSZ=1 N_RESP_PER_PROMPT=1 PPO_MINI_BATCH_SIZE=1 \
MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=122880 \
MAX_TOKENS_PER_TURN=8192 \
TASK_CONFIG=examples/codex/task_config_codex.yaml \
MASK_UNFINISHED_EPISODE=True \
bash examples/codex/train_qwen3p5_codex.sh
```

The default recipe targets Qwen3.5-9B on one eight-GPU node, with a 128K total
trajectory, an 8192-token per-turn cap, `qwen3_coder`, and one rollout
session. `CONCURRENCY` maps directly to
`agent_runners.task.max_concurrent_sessions` and must be a positive integer.
`MAX_TOKENS_PER_TURN` is also positive and independent of the total trajectory
capacity.

Text-only Qwen3.5 runs set
`+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True`.
Thinking is disabled by default, with `qwen3` as the reasoning parser. Set
`VLLM_LANGUAGE_MODEL_ONLY=False` only for an explicitly multimodal recipe.
The sample uses `vanilla_mbridge=False`, disables Megatron gradient
accumulation fusion, and uses 16384 max batched tokens because the pinned
remote186 environment has no Apex CUDA extension and the long-context vLLM
configuration is validated with these settings.

The RUNTIME_ENV YAML must provide worker environment variables for the
OpenYuanrong provider: OPENYUANRONG_SERVER_ADDRESS,
OPENYUANRONG_TOKEN, and OPENYUANRONG_TUNNEL_SSL_VERIFY. Add
AKERNEL_SDK_LD_PRELOAD only when the provider environment needs the optional
libffi compatibility hook. Keep credentials in that runtime-env file or a
secret provider; the launcher does not put them in the Ray command line.

## Sidecar

Build the fixed Codex sidecar with:

```bash
bash examples/codex/build_tool.sh \
  --version 0.147.0 \
  --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
```

The image is mounted at `/opt/codex`; `run_agent.sh` writes the native
Responses configuration, reads the task prompt from stdin, and forwards any
extra positional arguments to `codex exec` without shell evaluation.
