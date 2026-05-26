# Plan E Integration Validation

These tests are host-capability checks for `llm_router`. They are allowed to
skip on a CPU-only or Mooncake-less development host, but Plan E completion
requires real pass results on the target GPU/Mooncake/vLLM/verl environment.

## CPU Regression

```bash
pytest llm_router/ -v --tb=short
ruff check llm_router
```

## GPU Validation

Use an idle GPU:

```bash
CUDA_VISIBLE_DEVICES=2 pytest llm_router/tests/test_manager_parity.py -v -s --tb=short
```

## Mooncake Host Validation

```bash
CUDA_VISIBLE_DEVICES=2 pytest llm_router/connector/tests/test_mooncake_store.py -v -s --tb=short
```

If `mooncake` is missing, the real TransferEngine tests skip. The fake-buffer
allocator tests still run and protect the `llm_router` LRU/free-list behavior.

## One-Shot Harness

```bash
bash llm_router/tests/integration/run_plan_e_validation.sh
```

The harness writes logs to `artifacts/plan-e/<timestamp>/`.
