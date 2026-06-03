# Minimal concurrency repro

A **local, CPU-only** repro (no GPU, no full training stack).

## Install

```bash
git clone https://github.com/verl-project/uni-agent.git && cd uni-agent && git checkout yy/modal_concurrency
python3 -m venv .venv
source .venv/bin/activate
# verl submodule + other deps
git submodule update --init --recursive
pip install --no-deps -e ./verl
pip install datasets modal ray loguru pydantic swe-rex boto3 swebench
# authorize modal
modal token set --token-id <TOKEN_ID> --token-secret <TOKEN_SECRET>
```

## Reproduce

```bash
# 1. Preprocess: pull SWE-bench_Verified from HuggingFace and preprocess
#    Output: reproduce/swe_bench_verified_modal.parquet
DEPLOYMENT=modal python examples/data_preprocess/swe_bench_verified.py --local-save-dir ./reproduce/

# 2. Run: start Ray, run the samples concurrently. Each sample spins up
#    a Modal sandbox -> applies gold patch -> evaluates -> closes the sandbox.
GLOBAL_CONCURRENCY=64 DEPLOYMENT=modal python -m reproduce.run
GLOBAL_CONCURRENCY=256 DEPLOYMENT=modal python -m reproduce.run
```

## Logs

Per-sample logs at `reproduce/logs/<run_id>.log` (sandbox lifecycle, Modal sandbox id + console link).