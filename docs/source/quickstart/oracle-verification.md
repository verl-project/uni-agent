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

## Local Docker Smoke Test

Use `examples/quickstart/oracle/task_config_docker.yaml` to run the same verifier
against local Docker containers. Install the oracle runner dependencies in your
Uni-Agent environment (`pip install ray datasets swebench==4.1.0 pyyaml`) and start
the Docker daemon. No GPU or model endpoint is needed for oracle mode.

The dataset supplies each sample's image; the Task Config selects Docker and sets
pull/start budgets. Canonical SWE-Bench images are `linux/amd64`. ARM hosts need
Docker's amd64 emulation and may take longer. Start with one sample and one worker:

```bash
python -m uni_agent.tasks.swe_bench.preprocess \
    --local-save-dir /tmp/uni-agent-oracle --max-instances 1

SANDBOX_STARTUP_TIMEOUT=1200 python examples/inference/parallel_run_oracle.py \
    --data-path /tmp/uni-agent-oracle/swe_bench_verified.parquet \
    --task-config examples/quickstart/oracle/task_config_docker.yaml \
    --num-workers 1 --concurrency 1 --limit 1 \
    --result-path /tmp/uni-agent-oracle/results.json
```

The outer startup budget includes image pulling, so it must allow enough time for
the configured 900-second pull and 120-second start budgets. Images can also be
pulled in advance. The example disables container networking; if a sample's
verifier needs to install dependencies from the internet, adjust `run_args` for
that sample. Avoid shared writable `/testbed` mounts between rollouts.

For provider regression tests, use a local image with Bash and coreutils:

```bash
docker pull ubuntu:22.04
UNI_AGENT_DOCKER_TEST_IMAGE=ubuntu:22.04 \
    python -m pytest tests/uni_agent/sandbox/test_docker_integration.py -q
```

These tests check workspace isolation, file transfer, Agent-to-reward visibility,
cleanup on agent/reward errors, timeout and cancellation, and name collisions.
They are opt-in (`cpu`, `level1`); normal CPU CI runs the daemon-free lifecycle tests.

To also exercise the actual SWE-Bench verifier, save a raw dataset row and pull its image:

```bash
python - <<'PY'
import json
from pathlib import Path
from datasets import load_dataset

dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
sample = next(row for row in dataset if row["instance_id"] == "psf__requests-1142")
Path("/tmp/swe-bench-sample.json").write_text(json.dumps(sample))
PY
docker pull --platform linux/amd64 swebench/sweb.eval.x86_64.psf_1776_requests-1142
UNI_AGENT_SWE_BENCH_SAMPLE=/tmp/swe-bench-sample.json \
    python -m pytest tests/uni_agent/sandbox/test_docker_integration.py -k real_swe_bench -q
```

The negative control must receive reward `0` with no patch, and the gold patch must
receive reward `1`. Both use `SWEBenchTask.run()` and the standard reward implementation,
with separate containers removed after each episode. The test expects the image
to have been pulled already and does not call an LLM.

A local smoke test validates the environment and verifier path. Training-scale
support additionally needs Linux/GPU training runs and sustained rollout concurrency
measurements; increase concurrency only after checking memory, disk usage, and cleanup.

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
