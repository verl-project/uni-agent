# DeepSWE v1.1 Evaluation

Evaluate the 113 [DeepSWE v1.1](https://github.com/datacurve-ai/deep-swe) tasks
with mini-swe-agent 2.4.6, Harbor, and E2B. This recipe serves
DeepSeek-V4-Flash-0731 through verl and vLLM on one 8-GPU node (TP=4).

## Prepare the environment

Run these commands inside a GPU environment with Python 3.12, `uv`, and a
DeepSeek-V4-capable vLLM installation. The tested environment used 8 B200 GPUs,
PyTorch 2.11.0+cu130, and vLLM 0.24.0. GPU dependencies must already be installed.

```bash
git clone https://github.com/verl-project/uni-agent.git
cd uni-agent

# Use the DeepSeek-V4-compatible verl revision tested with this recipe.
git clone https://github.com/yyDing1/verl.git ../verl-dsv4
git -C ../verl-dsv4 checkout 3b58d4d408b90d36b0784f046bc062a270c689b7
export VERL_ROOT="$(cd ../verl-dsv4 && pwd)"

uv pip install --python "$(command -v python)" --no-deps -e "$VERL_ROOT" -e .
uv pip install --python "$(command -v python)" \
  "harbor[e2b] @ git+https://github.com/harbor-framework/harbor.git@dba6b9de4bd014d0a9d430b6eb85f6a4ed883b35"
uv pip install --python "$(command -v python)" --no-deps \
  --reinstall-package TransferQueue --reinstall-package transformers --reinstall-package tokenizers \
  TransferQueue==0.1.11 transformers==5.16.0 tokenizers==0.23.1

export E2B_API_KEY="<your-e2b-api-key>"
export E2B_API_URL="http://<your-e2b-api-host>:<port>"
export E2B_SANDBOX_URL="http://<your-e2b-sandbox-host>:<port>"
export MODEL_PATH="/path/to/DeepSeek-V4-Flash-0731"
```

This recipe uses self-hosted E2B. Set both URLs to the full HTTP(S) endpoints
provided by your deployment. The sandbox must be able to reach the model host's Gateway ports.
Run the following commands in this same environment; `uv run --frozen` can
restore incompatible dependency versions from another project's lockfile.

## Prepare the data

```bash
git clone https://github.com/datacurve-ai/deep-swe.git ../deep-swe
git -C ../deep-swe checkout 0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea
export DATA_PATH="$PWD/data/deepswe_v11.parquet"
python examples/deepswe_v11/prepare_data.py \
  --task-root ../deep-swe/tasks \
  --output "$DATA_PATH"
```

The Parquet file stores absolute task paths, so keep the `deep-swe` checkout
available at that location when running evaluation.

## Run evaluation

```bash
# Verify one task with one rollout first.
LIMIT=1 CONCURRENCY=1 N=1 bash examples/deepswe_v11/run_eval.sh

# Evaluate all 113 tasks, with four rollouts each (452 total).
CONCURRENCY=64 N=4 bash examples/deepswe_v11/run_eval.sh
```

Results are written to `outputs/deepswe_v11/<RUN_ID>/results.json`, with
per-rollout logs under `logs/` in the same run directory. The score is the mean
reward across rollouts. Override `OUTPUT_DIR`, `RUN_ID`, `TENSOR_PARALLEL_SIZE`,
`N_GPUS_PER_NODE`, or `GATEWAY_COUNT` as needed. Omit `LIMIT` for the full dataset.

Optional: verify a reference solution without loading the model:

```bash
python examples/inference/parallel_infer_api.py \
  --data-path "$DATA_PATH" \
  --task-config examples/deepswe_v11/task_config_oracle.yaml \
  --base-url http://unused.invalid/v1 \
  --limit 1 --concurrency 1 \
  --log-dir outputs/deepswe_v11/oracle/logs \
  --result-path outputs/deepswe_v11/oracle/results.json
```
