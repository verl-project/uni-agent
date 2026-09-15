# Agent Aware Router Examples

Experiment matrix drivers for the KV-cache-aware router: sweep the router over
(concurrency × context × load-threshold) against SWE-bench Verified, retrying each
run until it completes.

For setup and concepts, see the documentation:

- [Run the Agent Aware Router](https://uni-agent.readthedocs.io/en/latest/quickstart/agent-aware-router.html)
- [Agent Aware Router concepts](https://uni-agent.readthedocs.io/en/latest/concepts/agent-aware-router.html)

## Files

- `single-node.sh`: single-node matrix driver (the main entry).
- `multi-node.sh`: the same matrix on a multi-node Ray cluster.
- `run_infer.sh` / `run_infer.py`: one-shot inference driver the matrix calls per run (`--help` for the full flag set).
- `task_config_mini_swe_agent.yaml`: task/agent config used by the matrix (mini-swe-agent in an openyuanrong sandbox).
- `uni_agent/agent_aware_router/insight/`: the Grafana dashboard json (kvc-router metrics); the drivers inject it into rl-insight at startup.

## Environment

Packages:

- [verl](https://github.com/verl-project/verl)
- [uni-agent (router branch)](https://github.com/verl-project/uni-agent)
- [rl-insight](https://github.com/verl-project/rl-insight)

rl-insight needs a one-time install before the driver can start it:

```bash
rl-insight server install
```

> The driver starts/stops the rl-insight server around each run; set `VERL_RL_INSIGHT_ENABLE=0` to disable it.

OpenYuanrong sandbox (the only reverse-tunnel provider) — export these before running (both are read by the drivers and forwarded to the workers):

```bash
export OPENYUANRONG_SERVER_ADDRESS="<server-address>"
export OPENYUANRONG_TOKEN="<token>"
export OPENYUANRONG_TUNNEL_SSL_VERIFY="0"   # optional, defaults to 0
```

> For the platform itself, see the [OpenYuanRong docs](https://docs.openyuanrong.org/zh-cn/latest/index.html). The sandbox is backed by [AKernel](https://github.com/inclusionAI/AKernel/blob/main/AGENTS.md) (`akernel_sdk`), which is where `<server-address>` / `<token>` come from.

## Grafana dashboard

The kvc-router metrics dashboard (`verl_agentic_rollout.json`) ships in `uni_agent/agent_aware_router/insight/`. Both drivers copy it into rl-insight's installed package before `rl-insight server start` (idempotent — heals pip reinstalls).

## Single-node usage

Prerequisites: a host with >= 8 GPUs (`--n-gpus-per-node` is pinned at 8), a local
model path, and a preprocessed dataset:

```bash
python -m uni_agent.tasks.swe_bench.preprocess --local-save-dir /path/to/swe_agent
```

```bash
DEVICE=gpu TP=2 MODEL=/path/to/model DATASET=/path/to/swe_agent/swe_bench_verified.parquet \
CONCURRENCYS="16" CONTEXTS="16384" LTS="0.7" MAX_SAMPLES=4 N=2 bash examples/agent_aware_router/single-node.sh
```

That smoke run sweeps one matrix cell; the defaults sweep
`CONCURRENCYS="16 24 32 128"` × `CONTEXTS="16384 32768 64000 128000"` × `LTS="0.7 0.9"`.
Other knobs (defaults): `DEVICE=ascend` (`gpu` for NVIDIA), `TP=4`, `MAX_SAMPLES=64`,
`RES_LEN=8000`, `N=8`.

Each run writes `infer-${DEVICE}-kvcaware-lt${lt}-prompt${MAX_SAMPLES}x${N}-${CONCURRENCY}x${CONTEXT}.log`
and is retried until the `inference summary` sentinel lands in its log; before every
attempt the script kills leftover `run_infer`/ray processes and clears the device.
A sticky baseline arm is planned to complete the sticky-vs-kvcaware comparison.

## Multi-node usage

Same matrix on N nodes: the invoking host runs the Ray head, and each entry in
`WORKERS` joins as a worker via `ssh <worker> docker exec <container> ray start ...`.

```bash
DEVICE=gpu MODEL=/path/to/model DATASET=/path/to/swe_bench_verified.parquet \
WORKERS="root@<worker-ip>" WORKER_CONTAINER=<container> NNODES=2 TP=2 \
MAX_SAMPLES=4 N=2 CONCURRENCYS="16" CONTEXTS="16384" LTS="0.7" \
bash examples/agent_aware_router/multi-node.sh
```

Knobs (defaults keep the original Ascend 6-node layout):

| Variable | Default | Meaning |
|---|---|---|
| `DEVICE` | `ascend` | `gpu` selects NVIDIA (CUDA_VISIBLE_DEVICES, nvidia-smi cleanup); `ascend` keeps HCCL/ASCEND env and davinci cleanup |
| `WORKERS` | 5× `root@10.22.22.2x` | space-separated `user@host` list joining as Ray workers |
| `WORKER_CONTAINER` | `hgq-verl-ascend` | container name `docker exec` enters on each worker |
| `NNODES` | `6` | head + workers; keep in sync with `len(WORKERS) + 1` |
| `TP` | `4` | tensor parallel size; replicas = NNODES × gpus-per-node / TP |
| `MAX_SAMPLES`/`N`/`RES_LEN`/`GPU_MEM_UTIL` | `64`/`8`/`8000`/`0.8` | per-run inference knobs (same as single-node) |
| `CONCURRENCYS`/`CONTEXTS`/`LTS` | Ascend full sweep | matrix axes, space-separated inside quotes |

Requirements:

- **Passwordless ssh** from the head container to each worker host:
  `ssh <user>@<worker> docker exec <container> true` must succeed; the driver
  checks this up front and fails fast otherwise.
- **Identical worker images**: vLLM replica server actors are scheduled onto
  workers, so every worker must run the same image as the head (same vLLM/verl
  versions) with the repo at the same path. `uni-agent`/`verl` must be
  importable inside the worker image (e.g. editable install) — the driver does
  not inject `PYTHONPATH`; worker env comes only from the `docker exec -e`
  whitelist.
- **No per-attempt timeout**: the retry loop only fires when `run_infer` exits,
  so a wedged run must be killed by hand on all nodes (`pkill -9 -f 'run_infer|ray::|VLLM::'`,
  `ray stop -f`) and the driver relaunched.
- **Device-aware cleanup**: on GPU the script never `fuser -k /dev/nvidia*`
  (it would hit every container on the host); it only clears ray processes and
  port 9092, so verify the accelerator is actually free between attempts.

Each run writes `infer-{sticky|kvcaware-lt${lt}}-prompt${MAX_SAMPLES}x${N}-${CONCURRENCY}x${CONTEXT}-n${NNODES}.log`
under the repo root and archives rl-insight data + trajectories to
`archive/${EXP_ID}/` after the `inference summary` sentinel lands.
