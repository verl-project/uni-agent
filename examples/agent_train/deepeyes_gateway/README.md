# DeepEyes Gateway Training Example

This example wires the DeepEyes multimodal tool-use recipe into the Uni-Agent
gateway framework path on `verl.trainer.main_ppo_sync`.

## Layout

- `uni_agent.recipes.deepeyes_gateway.agent_runner`: gateway-backed DeepEyes
  tool loop.
- `uni_agent.recipes.deepeyes_gateway.dataset`: dataset adapter that emits
  `raw_prompt`, `tools_kwargs`, and reward fields without local prompt
  tokenization.
- `uni_agent.recipes.deepeyes_gateway.reward`: self-contained `compute_score`
  wrapper for the DeepEyes LLM-as-a-judge reward.
- `configs/deepeyes_gateway_grpo.yaml`: recipe config using
  `uni_agent.trainer.framework.entry.AgentFrameworkRolloutAdapter`.
- `configs/image_zoom_in_tool_config.yaml`: image zoom-in tool config.
- `run_deepeyes_gateway_grpo.sh`: example full-data launch script.

## Prerequisites

- Run from the Uni-Agent repository with the `verl` trainer dependencies
  available.
- Launch an OpenAI-compatible judge service and set `LLM_AS_A_JUDGE_BASE`.
- Prepare a DeepEyes parquet dataset with image payloads.
- Reserve training GPUs separately from the judge GPU.

Example judge service:

```bash
CUDA_VISIBLE_DEVICES=7 \
python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/judge-model \
  --host 127.0.0.1 \
  --port 18901 \
  --served-model-name qwen3-4b-judge \
  --dtype float16 \
  --trust-remote-code \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.75 \
  --enforce-eager
```

## Launch

```bash
bash examples/agent_train/deepeyes_gateway/run_deepeyes_gateway_grpo.sh
```

Common overrides:

```bash
MODEL_PATH=/path/to/policy-model \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/val.parquet \
LLM_AS_A_JUDGE_BASE=http://127.0.0.1:18901/v1 \
PROJECT_NAME=my_project \
EXPERIMENT_NAME=my_run \
TOTAL_TRAINING_STEPS=20 \
bash examples/agent_train/deepeyes_gateway/run_deepeyes_gateway_grpo.sh
```

The script resolves the config directory relative to its own location, then
launches from the repository root so `uni_agent.*` recipe imports are stable.

## Notes

- No parquet data files are included in this example.
- The image tool implementation is still loaded from `verl.tools` by the tool
  config; the gateway framework adapter and recipe imports use `uni_agent.*`.
- Reward scoring returns `0.0` if the judge service or reward dependencies are
  unavailable.
