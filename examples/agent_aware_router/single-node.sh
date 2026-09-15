#!/usr/bin/env bash
# Single-node experiment matrix driver (Ascend by default, GPU-supported).
#
# Sweeps sticky-vs-kvcaware over (concurrency × context), retrying each run until
# the "inference summary" sentinel lands in its log. Requires an OpenYuanrong
# remote sandbox (the only reverse-tunnel provider).
#
# Hosts are assumed to have >= 8 GPUs (--n-gpus-per-node is pinned at 8).
# Every knob is env-overridable for smoke runs, e.g.:
#   DEVICE=gpu TP=2 MODEL=/path/to/model DATASET=/path/to/swe_bench_verified.parquet \
#   CONCURRENCYS="16" CONTEXTS="16384" LTS="0.7" MAX_SAMPLES=4 N=2 bash single-node.sh

set -uo pipefail

# ── Device backend: ascend (default) or gpu ──────────────────────────────
DEVICE="${DEVICE:-ascend}"
TP="${TP:-4}"
if [ "$DEVICE" = "ascend" ]; then
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
fi
export PYTHONHASHSEED=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${MODEL:?Set MODEL to a local model path}"
: "${DATASET:?Set DATASET to a swe_bench parquet (uni_agent.tasks.swe_bench.preprocess)}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"
RES_LEN="${RES_LEN:-8000}"
N="${N:-8}"

echo "please set right TOOL_PARSER for you model. Reference https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/tool_parser.py"
TOOL_PARSER="${TOOL_PARSER:-hermes}"

# OpenYuanrong sandbox creds (only reverse-tunnel provider).
: "${OPENYUANRONG_SERVER_ADDRESS:?Set OPENYUANRONG_SERVER_ADDRESS}"
: "${OPENYUANRONG_TOKEN:?Set OPENYUANRONG_TOKEN}"
export OPENYUANRONG_SERVER_ADDRESS OPENYUANRONG_TOKEN
export OPENYUANRONG_TUNNEL_SSL_VERIFY="${OPENYUANRONG_TUNNEL_SSL_VERIFY:-0}"
export SANDBOX_NAME_PREFIX="${SANDBOX_NAME_PREFIX:-mini-swe-}"

# rl-insight observability — start on entry, stop on exit. One-time `rl-insight server install`.
# Set VERL_RL_INSIGHT_ENABLE=0 to disable.
export VERL_RL_INSIGHT_ENABLE="${VERL_RL_INSIGHT_ENABLE:-1}"
export RL_INSIGHT_SERVER_URL="${RL_INSIGHT_SERVER_URL:-http://127.0.0.1:18080}"

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
ensure_router_dashboard

rl-insight server start --detach 2>/dev/null || true
trap 'rl-insight server stop 2>/dev/null || true' EXIT

TARGET="inference summary"

run_experiment() {
    local log_file=$1
    shift

    while ! grep -q "$TARGET" "$log_file" 2>/dev/null; do
        pkill -9 -f 'run_infer.py|ray::' || true
        ps -aux | grep run_infer.sh | grep -v grep | awk -F ' ' '{print $2}' | xargs -r -I {} kill -9 {} || true
        ray stop || true
        if [ "$DEVICE" = "ascend" ]; then
            fuser -k /dev/davinci* || true
            npu-smi info
        else
            nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
        fi
        bash "${REPO_ROOT}/examples/agent_aware_router/run_infer.sh" \
            --model-path "$MODEL" \
            --data-path "$DATASET" \
            --task-config "${REPO_ROOT}/examples/agent_aware_router/task_config_mini_swe_agent.yaml" \
            --device "$DEVICE" \
            --n-gpus-per-node 8 \
            --tp "$TP" \
            --response-length "$RES_LEN" \
            --max-model-len "$CONTEXT" \
            --max-samples "$MAX_SAMPLES" \
            --n "$N" \
            --shuffle \
            --concurrency "$CONCURRENCY" \
            --kv-events \
            --tool-parser "$TOOL_PARSER" \
            "$@" > "$log_file" 2>&1
    done
}

concurrencys=(${CONCURRENCYS:-16 24 32 128})
contexts=(${CONTEXTS:-16384 32768 64000 128000})
lts=(${LTS:-0.7 0.9})

for CONCURRENCY in "${concurrencys[@]}"; do
    for CONTEXT in "${contexts[@]}"; do
        LOG_FILE="infer-${DEVICE}-sticky-prompt${MAX_SAMPLES}x${N}-${CONCURRENCY}x${CONTEXT}.log"
        echo "Running sticky concurrency=${CONCURRENCY} context=${CONTEXT}"
        (
            export UNI_AGENT_ROUTER_DEBUG=1
            export UNI_AGENT_ROUTER_SLOW_CUT=least-inflight
            export UNI_AGENT_ROUTER_OVERLOAD_MODE=None
            run_experiment "$LOG_FILE"
            unset UNI_AGENT_ROUTER_DEBUG
        )

        for lt in "${lts[@]}"; do
            LOG_FILE="infer-${DEVICE}-kvcaware-lt${lt}-prompt${MAX_SAMPLES}x${N}-${CONCURRENCY}x${CONTEXT}.log"
            echo "Running kvcaware-lt${lt} concurrency=${CONCURRENCY} context=${CONTEXT}"
            run_experiment "$LOG_FILE" --load-threshold "$lt"
        done
    done
done
