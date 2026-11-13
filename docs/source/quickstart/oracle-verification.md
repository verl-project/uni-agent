# Verify Oracle Solutions

Before large-scale training or inference, run each task's solution through its normal verifier.

For a single SWE-rebench Docker instance with missing imports or a failed oracle,
see [SWE-rebench diagnostics](swe-rebench-diagnostics.md) for recorded gold/no-patch
controls and per-instance installation checks.

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
- `results`: source row index, task name, resolved sandbox configuration, instance ID, log ID, admission outcome, solved status, reward, verifier details, and any execution error for each sample.

Each sample also writes a task log under `--log-dir` (default: `/tmp/eval_gold_patch`). When an oracle sample is unsolved or errors, inspect that log before starting training or inference.

## Export Oracle-passing Training Data

Use the same dataset, image and instance installation that training will use.
For SWE-rebench, the dataset's existing `filtered` split is not a guarantee that
its gold patch still passes in today's environment. Validate before training:

```bash
python examples/inference/parallel_run_oracle.py \
    --data-path ~/data/uni_agent/swe_rebench_filtered.parquet \
    --task-config examples/quickstart/oracle/task_config_oracle.yaml \
    --num-workers 4 --concurrency 8 \
    --result-path ~/data/uni_agent/swe_rebench_oracle.json \
    --passed-data-path ~/data/uni_agent/swe_rebench_oracle_passed.parquet
```

`--passed-data-path` requires an audit JSON and a new output file. The runner
exports only rows with outcome `passed` from **this run**, preserving the input
schema, row contents and order. It selects by source row index, so repeated
instance IDs do not admit other rows accidentally. With `--limit N`, only the
first N rows are eligible; untested rows are never included. If none pass, the
output parquet is empty with the original schema.

The outcomes distinguish admission decisions from root causes:

| Outcome | Meaning | Training admission |
| --- | --- | --- |
| `passed` | Task reports resolved (or full reward 1 when resolution is absent), without a reported execution failure | Include this row |
| `oracle_failed` | Task returns an unresolved verifier result | Exclude; inspect logs for a bad sample/image, dependency drift, flakiness, or a solution failure |
| `execution_error` | Task raises, reports incomplete execution/missing test output, or its Oracle command fails | Exclude pending triage/retry; do not call this a confirmed invalid sample |

`status` remains `completed` or `error`; the summary separately counts
`oracle_failed` rows. `solved_rate` retains all attempted rows as its denominator,
including execution errors. An execution error cannot be admitted even if a
partial verifier result says resolved. When `resolved` is absent, only reward
`1.0` passes; partial rewards alone do not pass. An explicit `resolved` result
takes priority over the numeric reward.

This is a conservative gate over the existing Task contract, not a new verifier.
For example, SWE-rebench's script can return `eval_completed=true` after an
installation/collection error because its final cleanup command succeeds. Such
a row remains excluded when unresolved, but the runner cannot automatically infer
its root cause. Read the retained `extra_info` and per-sample logs. SWE-bench
and SWE-rebench stop before verification if the gold patch fails to apply; the
Oracle audit records this as an `execution_error`, retaining the instance ID,
exit code and command stdout/stderr. Such rows never enter the passing export.
No scoring rule, patch content, installation command or default dependency is
changed by this gate.

Keep the result JSON, input dataset identity/checksum, image digests and Task
Config with the exported parquet. The JSON records the resolved sandbox per row,
the config path and provider override; the exported rows themselves are unchanged.
Reuse the same environment overrides for training. Do not add case-specific
repairs or dependency constraints to the original-environment validation. A pass
in a separately modified environment is separate evidence, and must never be
combined with this run's passing rows. This command does not import historical
result files or single-instance diagnostic output.

After the oracle baseline passes, continue to [Run Agent Inference](agent-inference.md) with the same dataset and sandbox configuration.
