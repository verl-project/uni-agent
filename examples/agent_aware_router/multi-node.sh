#!/usr/bin/env bash
# Multi-node experiment driver (Ascend or GPU via DEVICE).
#
# N nodes (this host = Ray head + N-1 ssh workers via `docker exec ${WORKER_CONTAINER}`).
# rl-insight stays up for the whole matrix; the Ray cluster is rebuilt per attempt
# (clean -> ray up -> run_infer.sh, retry until "inference summary").
#
# Device-specific defaults (override via env):
#   DEVICE=ascend WORKERS="root@10.22.22.22 ..." WORKER_CONTAINER=hgq-verl-ascend TP=4
#   DEVICE=gpu    WORKERS="root@<worker-ip>"     WORKER_CONTAINER=<gpu-image-container> TP=2
# Workers must run the same image as the head (vLLM server actors are scheduled
# onto them) with uni-agent/verl editable-installed — no PYTHONPATH is injected.
# Matrix params are env-overridable like single-node.sh:
#   CONCURRENCYS/CONTEXTS/LTS/MAX_SAMPLES/N/RES_LEN.
# Note: cross-replica KV sharing via mooncake is deliberately NOT enabled
# (run_infer.sh only attaches the connector with --enable-mooncake).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# =====================================================================
# Configuration
# =====================================================================
DEVICE="${DEVICE:-ascend}"

WORKERS=(${WORKERS:-root@10.22.22.22 root@10.22.22.23 root@10.22.22.24 root@10.22.22.25 root@10.22.22.26})

WORKER_CONTAINER="${WORKER_CONTAINER:-hgq-verl-ascend}"
NNODES="${NNODES:-6}"                    # head + workers; keep in sync with WORKERS
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
TP="${TP:-4}"                            # ascend: 48 NPU / 4 = 12 replicas
RAY_PORT=6379
RL_INSIGHT_PORT=18080

: "${MODEL:?Set MODEL to a local model path}"
: "${DATASET:?Set DATASET to a swe_bench parquet (uni_agent.tasks.swe_bench.preprocess)}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${REPO_ROOT}/archive}"   # trajectories / rl-insight data per experiment
MAX_SAMPLES="${MAX_SAMPLES:-64}"
RES_LEN="${RES_LEN:-8000}"
N="${N:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"

if [ "$DEVICE" = "ascend" ]; then
    export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-26100}
    export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
fi

# OpenYuanrong sandbox creds (only reverse-tunnel provider); passed to workers
# via `docker exec -e`.
: "${OPENYUANRONG_SERVER_ADDRESS:?Set OPENYUANRONG_SERVER_ADDRESS}"
: "${OPENYUANRONG_TOKEN:?Set OPENYUANRONG_TOKEN}"

# =====================================================================
# Helpers
# =====================================================================
log() { echo "[$(date +%H:%M:%S)] $*"; }

head_ip() {
    hostname -I | awk '{print $1}'
}

worker_exec() {
    local host=$1; shift
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${host}" \
        "docker exec ${WORKER_CONTAINER} $*"
}

# worker_exec + env injected via `docker exec -e` (baked into the raylets).
# Ascend-only vars are emitted only on ascend; GPU relies on the worker
# container's editable installs (uni-agent/verl), not PYTHONPATH.
worker_exec_with_env() {
    local host=$1 hip=$2; shift 2
    local ascend_env=""
    if [ "$DEVICE" = "ascend" ]; then
        ascend_env="-e HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-26100} \
        -e RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 \
        -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
    fi
    ssh -o StrictHostKeyChecking=no "${host}" "docker exec \
        -e VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO} \
        -e VERL_RL_INSIGHT_ENABLE=1 \
        -e RL_INSIGHT_SERVER_URL=http://${hip}:${RL_INSIGHT_PORT} \
        -e OPENYUANRONG_SERVER_ADDRESS=${OPENYUANRONG_SERVER_ADDRESS} \
        -e OPENYUANRONG_TOKEN=${OPENYUANRONG_TOKEN} \
        -e OPENYUANRONG_TUNNEL_SSL_VERIFY=${OPENYUANRONG_TUNNEL_SSL_VERIFY:-0} \
        -e SANDBOX_NAME_PREFIX=${SANDBOX_NAME_PREFIX:-mini-swe-} \
        ${ascend_env} \
        -e PYTHONHASHSEED=0 \
        ${WORKER_CONTAINER} $*"
}

# Fan-out a command across all workers in parallel.
worker_exec_all() {
    local cmd=$1
    for w in "${WORKERS[@]}"; do
        log "  -> ${w}: ${cmd}"
        worker_exec "${w}" "${cmd}" &
    done
    wait
}

# =====================================================================
# Step 1: passwordless-SSH + container reachability check
# =====================================================================
step1_ssh_check() {
    log "=== Step 1: SSH + container reachability check (${#WORKERS[@]} workers) ==="
    local ok=1
    for w in "${WORKERS[@]}"; do
        if worker_exec "${w}" "true" 2>/dev/null; then
            log "  ok ${w} (container ${WORKER_CONTAINER})"
        else
            log "  FAIL ${w} (ssh or docker exec ${WORKER_CONTAINER} failed)"
            ok=0
        fi
    done
    [[ ${ok} -eq 1 ]] || { log "ERROR: not all workers reachable; fix ssh/docker first."; exit 1; }
}

# =====================================================================
# Per-attempt cluster teardown / bring-up (all 6 nodes)
# =====================================================================
# Cleanup order: kill driver -> ray stop -> kill stray ray:: -> free accelerators.
# Ascend frees /dev/davinci* + :9092; GPU only verifies memory (no fuser -k on
# /dev/nvidia* — killing those wedges the driver for every container on the host).
KILL_DRIVER_CMD="ps aux | grep -E 'run_infer[.]sh|run_infer[.]py' | grep -v grep | awk '{print \$2}' | xargs -r kill -9 2>/dev/null || true"
KILL_RAY_CMD="ps aux | grep -E 'ray[:][:]|raylet|gcs_server' | grep -v grep | awk '{print \$2}' | xargs -r kill -9 2>/dev/null || true"
if [ "$DEVICE" = "ascend" ]; then
    FUSER_CMD="bash -lc 'fuser -k /dev/davinci* 2>/dev/null || true; fuser -k 9092/tcp 2>/dev/null || true'"
    ACCEL_STATUS_CMD="npu-smi info 2>/dev/null | tail -3 || true"
else
    FUSER_CMD="bash -lc 'fuser -k 9092/tcp 2>/dev/null || true'"
    ACCEL_STATUS_CMD="nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true"
fi
NODE_CLEANUP="${KILL_DRIVER_CMD}; ray stop -f 2>/dev/null || true; ${KILL_RAY_CMD}; ${FUSER_CMD}"

clean_all_nodes() {
    log "  cleaning all ${NNODES} nodes (kill by pid, ray stop, free accelerators)"
    eval "${NODE_CLEANUP}"
    worker_exec_all "${NODE_CLEANUP}"
}

setup_ray_cluster() {
    log "  starting ray cluster (nnodes=${NNODES}, device=${DEVICE})"
    export OPENYUANRONG_TUNNEL_SSL_VERIFY="${OPENYUANRONG_TUNNEL_SSL_VERIFY:-0}"
    export SANDBOX_NAME_PREFIX="${SANDBOX_NAME_PREFIX:-mini-swe-}"
    export PYTHONHASHSEED=0
    if [ "$DEVICE" = "ascend" ]; then
        export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
    else
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    fi
    # idle-worker reaper must be disabled at ray start --head; ray.init's
    # _system_config is ignored when connecting to a running cluster.
    ray start --head --port="${RAY_PORT}" --temp-dir=/tmp/ray_head \
        --system-config='{"idle_worker_killing_time_threshold_ms": 2147483647}'

    # Custom --temp-dir hides the cluster from bare ray.init() discovery.
    export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

    log "  workers joining cluster (inside ${WORKER_CONTAINER})..."
    for w in "${WORKERS[@]}"; do
        log "    -> ${w} joining"
        worker_exec_with_env "${w}" "${HEAD_IP}" \
            "ray start --address=${HEAD_IP}:${RAY_PORT} --temp-dir=/tmp/ray_worker"
        sleep 3
    done
    wait

    log "waiting for cluster to settle..."
    sleep 10
    ray status || true
    log "alive nodes: $(ray nodes 2>/dev/null | grep -c alive)"
    ray list nodes 2>/dev/null | grep -E "ALIVE" | sed -E 's/^ *[0-9]+ +([0-9a-f]+) +([0-9.]+).*/  \1 \2/'
    local nconn
    nconn=$(ray list nodes 2>/dev/null | grep -c ALIVE)
    [[ "${nconn}" -eq "${NNODES}" ]] || { log "ERROR: got ${nconn} nodes, expected ${NNODES}"; return 1; }
    log "  cluster up: ${nconn}/${NNODES} nodes alive"
}

# Copy the router dashboard (uni_agent/agent_aware_router/insight/) into
# rl-insight's installed package before start. Idempotent every run — heals
# pip reinstalls and picks up json updates.
ensure_router_dashboard() {
    local dash_dir
    dash_dir="$(python -c 'import pathlib, rl_insight; print(pathlib.Path(rl_insight.__file__).parent / "config/services/grafana/dashboards")' 2>/dev/null)" || return 0
    [ -d "$dash_dir" ] || return 0
    mkdir -p "$dash_dir"
    cp -v "$REPO_ROOT"/uni_agent/agent_aware_router/insight/*.json "$dash_dir"/ || true
}

# =====================================================================
# Step 2: rl-insight server — started once, up for the whole matrix
# =====================================================================
step2_rl_insight() {
    log "=== Step 2: rl-insight server on head (${HEAD_IP}:${RL_INSIGHT_PORT}), up for the whole matrix ==="
    export VERL_RL_INSIGHT_ENABLE=1
    export RL_INSIGHT_SERVER_URL="http://${HEAD_IP}:${RL_INSIGHT_PORT}"
    ensure_router_dashboard
    rl-insight server stop
    rl-insight server start --detach 2>/dev/null || true   # already-running is fine
    trap 'rl-insight server stop 2>/dev/null || true' EXIT
    log "rl-insight up; workers scrape via RL_INSIGHT_SERVER_URL=${RL_INSIGHT_SERVER_URL}"
}

# =====================================================================
# Step 3: ascend-exps matrix (each run_infer spans all 6 nodes)
# =====================================================================

archive_rl_insight() {
    local exp_id=${1:-}
    if [ -z "${exp_id}" ]; then
        log " (skip rl-insight archive: EXP_ID empty)"
        return 0
    fi
    local dest="${ARCHIVE_ROOT}/${exp_id}"
    mkdir -p "${dest}"
    local tgz="${dest}/rl-insight-data.tgz"
    log " archiving rl-insight data -> ${tgz}"
    if [ -d /root/.rl-insight/data ]; then
        ( cd /root/.rl-insight/ && tar czf "${tgz}" data ) 2>/dev/null \
            && log " rl-insight archived ($(du -h "${tgz}" 2>/dev/null | cut -f1))" \
            || log " WARNING: rl-insight archive failed"
    fi
}

run_experiment() {
    local log_file=$1
    shift

    local log_mtime_before=$(stat -c %Y "${log_file}" 2>/dev/null || echo 0)

    while ! grep -q "${TARGET}" "${log_file}" 2>/dev/null; do
        # Tolerated failures (cleanup/setup/run) land back in this retry loop;
        # keep `set -e` armed globally instead of `set +e`-ing the whole matrix.
        clean_all_nodes || true
        setup_ray_cluster || true
        eval "${ACCEL_STATUS_CMD}"
        log "  running -> ${log_file}"

        if ! bash "${REPO_ROOT}/examples/agent_aware_router/run_infer.sh" \
            --model-path "${MODEL}" \
            --data-path "${DATASET}" \
            --task-config "${REPO_ROOT}/examples/agent_aware_router/task_config_mini_swe_agent.yaml" \
            --device "${DEVICE}" \
            --nnodes "${NNODES}" \
            --n-gpus-per-node "${N_GPUS_PER_NODE}" \
            --tp "${TP}" \
            --gpu-memory-utilization "${GPU_MEM_UTIL}" \
            --response-length "${RES_LEN}" \
            --max-model-len "${CONTEXT}" \
            --max-samples "${MAX_SAMPLES}" \
            --n "${N}" \
            --shuffle \
            --concurrency "${CONCURRENCY}" \
            --kv-events \
            "$@" > "${log_file}" 2>&1; then
            log "  (run failed, will retry)"
        fi
    done
    log "experiment resolved ${log_file}"
    local log_mtime_after=$(stat -c %Y "${log_file}" 2>/dev/null || echo 0)
    if [ "${log_mtime_after}" != "${log_mtime_before}" ]; then
        archive_rl_insight "${EXP_ID}"
    else
        log " (log unchanged - resolved from previous run, skip archive)"
    fi
}

step3_matrix() {
    log "=== Step 3: ${DEVICE}-exps matrix (nnodes=${NNODES}, tp=${TP}, replicas=$((NNODES*N_GPUS_PER_NODE/TP)), per-run ray cluster) ==="

    local concurrencys=(${CONCURRENCYS:-16 24 32 128 192 256})
    local contexts=(${CONTEXTS:-16384 32768 64000 128000})
    export TARGET="inference summary"

    for CONCURRENCY in "${concurrencys[@]}"; do
        for CONTEXT in "${contexts[@]}"; do
            local EXP_ID="infer-sticky-prompt${MAX_SAMPLES}x${N}-${CONCURRENCY}x${CONTEXT}-n${NNODES}"
            local LOG_FILE="${EXP_ID}.log"
            log "sticky concurrency=${CONCURRENCY} context=${CONTEXT} (traj -> ${ARCHIVE_ROOT}/${EXP_ID}/trajectories)"
            (
                export UNI_AGENT_ROUTER_DEBUG=1
                export UNI_AGENT_ROUTER_SLOW_CUT=least-inflight
                export UNI_AGENT_ROUTER_OVERLOAD_MODE=None
                run_experiment "${LOG_FILE}"
                unset UNI_AGENT_ROUTER_DEBUG
            )

            local lts=(${LTS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9})
            for lt in "${lts[@]}"; do
                local EXP_ID="infer-kvcaware-lt${lt}-prompt${MAX_SAMPLES}x${N}-${CONCURRENCY}x${CONTEXT}-n${NNODES}"
                local LOG_FILE="${EXP_ID}.log"
                log "kvcaware-lt${lt} concurrency=${CONCURRENCY} context=${CONTEXT} (traj -> ${ARCHIVE_ROOT}/${EXP_ID}/trajectories)"
                run_experiment "${LOG_FILE}" \
                    --load-threshold "${lt}"
            done
        done
    done
    log "=== matrix complete ==="
}

# =====================================================================
# Main
# =====================================================================
step1_ssh_check
HEAD_IP=$(head_ip)
export HEAD_IP
step2_rl_insight
step3_matrix
