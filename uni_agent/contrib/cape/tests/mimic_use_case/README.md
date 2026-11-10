# CAPE mimic use-case — workloads & harness

Assets for the CAPE multi-user GPU-pool mimic described in this standalone project.
The same two workloads are submitted by every user; per-user/run differences come from env vars
(`CAPE_OUTPUT`, `CAPE_RUN_TAG`, …) and CAPE binds.

## Workloads

### A — `gemm_silu.py` (PyTorch)

BF16 `silu(A @ B)` at `M=N=K=8192` on CUDA. Self-contained; imports torch lazily.

```bash
CAPE_OUTPUT=/path/to/user/out python gemm_silu.py
```

Writes `gemm_silu_result.txt` (status, run_tag, device, `CUDA_VISIBLE_DEVICES`, checksum).
Exit 0 = PASS, 1 = FAIL, 2 = no GPU. Image: `nvcr.io/nvidia/pytorch:26.06-py3`.

### B — `cutedsl_add.py` (CUTLASS CuTe DSL)

A single self-contained file: the upstream element-wise-add kernel + jit launch (trimmed from
NVIDIA/cutlass `examples/python/CuTeDSL/cute/ampere/kernel/elementwise/elementwise_add.py`,
BSD-3-Clause header preserved) merged with a minimal BF16 runner. Self-verifies `c == a + b`.

```bash
pip install nvidia-cutlass-dsl
CAPE_OUTPUT=/path/to/user/out python cutedsl_add.py
```

Writes `cutedsl_add_result.txt` (incl. run_tag). Provenance: the kernel was transcribed from
upstream via a web tool (no egress on the dev host) — verify it against the installed
`nvidia-cutlass-dsl` on the run node.

## Harness

CLI-driven through the implementation selected by `CAPE_BIN` (`rscape` by default).
Set `CAPE_BIN=pycape` for the maintained Python/venv path.
The harness uses explicit `pycape controller init/start --state-dir`, authenticated `pycape node enroll`, async
`pycape run`, and `pycape status/logs/wait/cancel/admin` query/control verbs.

- **`start.sh`** — orchestrator. Sets hyper-parameters (env), brings up the pool, launches `U` users
  at arrival times drawn from a distribution, optionally injects/removes capacity mid-run, samples
  the utilization endpoint, waits for the queue to drain, tears down. Writes the evidence bundle to
  the caller-selected `RESULTS_DIR` (default: `cape/evidence/cape-use-case/...`).
- **`pool.sh`** `up|inject|drain|remove|down` — controller + node agents; `inject`/`remove` mimic
  allocations joining/leaving (dynamic capacity) by running agents over disjoint GPU-UUID subsets.
- **`user.sh`** `<uid> [arrival_delay]` — one user's two requests, each `pycape run` **without
  `--wait`** (async); same `--session-id` for warm reuse; per-user `--bind` for the output (the
  §12 R2 same-path/per-user-bind isolation test); records jobid + submit latency to `submits.csv`.
- **`workload.sh`** `<A|B> <run_tag> <out_dir>` — runs **inside** the container: installs packages
  (the CuTe DSL for B) and launches the workload, writing the per-user output with `CAPE_RUN_TAG`.
- **`analyze.py`** — computes the §12 gates from the bundle and writes `summary.json`, `summary.md`, and `isolation_checks.json`.

### Quick start

```bash
STAMP=$(date +%Y%m%d-%H%M%S)
RUN_STAMP="$STAMP" G=2 U=50 SESSION_MODE=shared \
  ARRIVAL_DIST=poisson ARRIVAL_WINDOW=30 bash start.sh
python analyze.py "$PWD/evidence/shared-G2-R100-$STAMP"
# sweep utilization: rerun with G=1,2,4,8 (use-case §6).
```

**Dynamic pool size** — `CAPACITY_SCHEDULE` is a `;`-separated list of `AT:ACTION:ARGS` events fired
`AT` seconds after the pool comes up (ARGS go to `pool.sh`). E.g. start with 1 GPU, add 8 at 60s,
add 4 at 120s (`gpus_total` grows 1→9→13 in `utilization_series.csv`):

```bash
G=1 CAPACITY_SCHEDULE="60:inject:node-1 alloc-1 8;120:inject:node-2 alloc-2 4" bash start.sh
# add ";180:remove:alloc-1" to drop the 8-GPU allocation at 180s.
```

Each `inject` joins a new node/allocation; on a single box it draws from that box's free GPU UUIDs,
so growing to N GPUs needs N to physically exist — in a real cluster each `inject` is a new
allocation's agent joining (the design's dynamic-capacity model).

Key knobs (env): `G`, `U`, `SESSION_MODE` (shared|distinct), `ARRIVAL_DIST` (burst|uniform|poisson),
`ARRIVAL_WINDOW`, `INTER_REQUEST_GAP`, `SAMPLE_INTERVAL`, `CAPACITY_SCHEDULE`, `CAPE_IMAGE`,
`APPTAINER_BIN`, `CAPE_STATE_DIR`, `CAPE_ADMIN_TOKEN_FILE`. Authentication is mandatory.

## `CAPE_RUN_TAG`

Both workloads write `CAPE_RUN_TAG` into their output. The workspace-isolation test (use-case §12 R2)
gives each user the **same** in-container output path but a **different** host bind, then checks each
host file holds exactly its owner's tag → proves no cross-user clobbering.

## Notes

- Tiny kernels (sub-second) — the mimic measures pool work-conservation + isolation, not per-GPU
  packing (MPS/MIG is V1).
- Same image for both → a user's two requests reuse one warm session under the same `--session-id`.
