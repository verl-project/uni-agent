# Verify Oracle Solutions

Before large-scale training or inference, run each task's solution through its normal verifier.

This validates two prerequisites:

1. **Sandbox scalability.** RL training requires many sandboxed tasks to run in parallel. This can be estimated from `batch_size × group_size` plus asynchronous in-flight or stale samples.
2. **Verifier correctness.** The verifier produces the reward signal that drives RL optimization. Validate it against oracle solutions first to ensure that correct outcomes consistently receive the expected reward.

## Prepare Data

Preprocess a small subset for an end-to-end smoke test:

```bash
python -m uni_agent.tasks.swe_bench.preprocess --local-save-dir ~/data/uni_agent
```

This writes `~/data/uni_agent/swe_bench_verified.parquet`, which contains the gold patch used by oracle mode.

## Task Configuration

Use the provided Task Config `examples/quickstart/oracle/task_config_oracle.yaml`, which contains:

```yaml
- name: swe_bench
  run_oracle_solution: true
  sandbox:
    provider: modal
    runtime_timeout: 3600
```

The complete file also contains entries for `swe_bench_multilingual`, `swe_rebench`, and `terminal_bench`.

The runner uses the sandbox provider and runtime timeout from the resolved Task Config. Set `SANDBOX_PROVIDER` only when you need to override the configured provider for a run.

## Run Oracle Verification

Run the prepared samples:

```bash
python examples/inference/parallel_run_oracle.py \
    --data-path ~/data/uni_agent/swe_bench_verified.parquet \
    --task-config examples/quickstart/oracle/task_config_oracle.yaml \
    --num-workers 4 \
    --concurrency 8 \
    --limit 8 \
    --result-path ~/data/uni_agent/swe_bench_verified_oracle.json
```

The runner distributes samples across Ray actors. `--concurrency` is the global maximum number of in-flight tasks; `--num-workers` controls the number of actors. `--limit N` runs only the first `N` dataset rows.

For each SWE-Bench sample, the Task:

1. Starts the configured sandbox image.
2. Applies the gold patch instead of launching an Agent.
3. Runs the standard SWE-Bench verifier.
4. Returns the Task reward and verification details.

You can remove `--limit` and run the full dataset at the desired concurrency:

```bash
python examples/inference/parallel_run_oracle.py \
    --data-path ~/data/uni_agent/swe_bench_verified.parquet \
    --task-config examples/quickstart/oracle/task_config_oracle.yaml \
    --num-workers 8 \
    --concurrency 500 \
    --result-path ~/data/uni_agent/swe_bench_verified_oracle.json
```

## Review the Results

The test should finish without execution errors and should normally report every oracle sample as solved:

```text
----------------- oracle summary -----------------
  solved        8   (100.0%)
  completed     8
  error         0
  total         8
------------- avg 64.0s | wall 99.3s -------------
```

The full-dataset oracle verification results are summarized below:

```text
----------------- oracle summary -----------------
  solved      492   (98.4%)
  completed   500
  error         0
  total       500
------------ avg 15.3s | wall 304.6s -------------
```

!!! note "Oracle failures at scale"
    Across SWE-Bench, SWE-Bench Verified, Terminal-Bench, and similar benchmarks, a small number of oracle failures can be acceptable after triage. Common causes include invalid gold patches, flaky samples, environment drift, and transient sandbox issues.


The result JSON file passed through `--result-path` contains:

- `summary`: solved count and rate, execution counts, average verifier time, and total wall time.
- `results`: instance ID, log ID, solved status, verifier execution time, and any execution error for each sample.

Each sample also writes a task log under `--log-dir` (default: `/tmp/eval_gold_patch`). When an oracle sample is unsolved or errors, inspect that log before starting training or inference.

After the oracle baseline passes, continue to [Run Agent Inference](agent-inference.md) with the same dataset and sandbox configuration.
