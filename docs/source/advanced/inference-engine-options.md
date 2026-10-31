# Inference Engine Options

Follow the [inference quickstart](../quickstart/agent-inference.md#verl-rollout-engine-mode) to prepare your data, task configuration, and rollout environment. This guide covers optional graph and compilation settings for the native engines.

## How options are applied

`--engine-kwargs` is merged into `rollout.engine_kwargs.<engine>` after the
backend-specific defaults. Nested objects, lists, and booleans are preserved.
Omitting it keeps the existing behavior. Native options must be supported by the
installed framework and consistent with common options such as `--enforce-eager`.

## vLLM on CUDA

For a single active request, the following vLLM configuration keeps model
compilation disabled and captures only batch-size-1 decode CUDA graphs:

```bash
python examples/inference/parallel_infer_verl.py \
    --data-path /path/to/tasks.parquet --model-path /path/to/model \
    --task-config /path/to/task.yaml --engine vllm \
    --concurrency 1 --gateway-count 1 --limit 1 --n 1 \
    --engine-kwargs '{"compilation_config":{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}}'
```

## vLLM on Ascend

For Ascend vLLM deployments that require `VLLM_COMPILE` for ACL graph
initialization, use the native compilation path together with decode graphs:

```bash
python examples/inference/parallel_infer_verl.py \
    --data-path /path/to/tasks.parquet --model-path /path/to/model \
    --task-config /path/to/task.yaml --engine vllm \
    --concurrency 1 --gateway-count 1 --limit 1 --n 1 \
    --engine-kwargs '{"compilation_config":{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1],"pass_config":{"enable_sp":false}}}'
```

This is a combined compilation-and-graph configuration. The Ascend platform
selects its own compiler backend; it is not the same setting as CUDA graph-only
execution. Tensor parallelism is configured separately with `--tensor-parallel-size`.

## SGLang

For SGLang, capture batch-size-1 decode graphs while leaving prefill graph
capture and model-level `torch.compile` disabled:

```bash
python examples/inference/parallel_infer_verl.py \
    --data-path /path/to/tasks.parquet --model-path /path/to/model \
    --task-config /path/to/task.yaml --engine sglang \
    --concurrency 1 --gateway-count 1 --limit 1 --n 1 \
    --engine-kwargs '{"cuda_graph_config":{"decode":{"backend":"full","bs":[1],"max_bs":1},"prefill":{"backend":"disabled"}},"enable_torch_compile":false}'
```

## Versions and workload limits

These option names were validated with vLLM 0.24.0 and SGLang 0.5.20. Do not
combine the graph examples with `--enforce-eager`. Capture sizes must match the
intended workload; `[1]` is a single-request example, not a general throughput
default. Startup/capture cost and generated trajectories can change, so measure
complete tasks and verify rewards as well as serving speed.

## Mini-SWE-Agent launcher

The mini-swe-agent launcher forwards the same JSON through `ENGINE_KWARGS_JSON`:

```bash
ENGINE=sglang ENFORCE_EAGER=0 \
ENGINE_KWARGS_JSON='{"cuda_graph_config":{"decode":{"backend":"full","bs":[1],"max_bs":1},"prefill":{"backend":"disabled"}},"enable_torch_compile":false}' \
bash examples/mini_swe_agent/run_infer_mini_swe_agent.sh
```
