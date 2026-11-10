#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# start.sh — set hyper-parameters and drive the whole CAPE mimic pipeline (use-case §11 phases 1-6).
#
# Pipeline: bring up the pool (pool.sh) -> launch U users at arrival times drawn from a distribution
# (user.sh, each async) -> optionally inject/remove capacity mid-run (pool.sh) -> sample the pool's
# utilization endpoint -> wait for the queue to drain -> tear down. Raw evidence lands in RESULTS_DIR
# for analyze.py (use-case §11 bundle / §12 gates).
#
# All commands use the executable selected by CAPE_BIN.
# The CLI layer now provides controller/node/run/status/logs/wait/cancel/admin verbs plus Apptainer
# node-join wiring and bind/workspace/output flags.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/project_root.sh"
source "$HERE/identity.sh"
umask 077
CAPE_ROOT="$(cape_project_root "$HERE")"

# ---- Hyper-parameters (override via env) ------------------------------------------------------
export G="${G:-2}"                              # GPUs in the pool (the §6 sweep var: 1,2,4,8)
export U="${U:-50}"                             # users (each submits 2 requests -> R = 2U)
export SESSION_MODE="${SESSION_MODE:-shared}"   # shared = warm reuse | distinct = independent
export ARRIVAL_DIST="${ARRIVAL_DIST:-poisson}"  # burst | uniform | poisson  (user arrival timing)
export ARRIVAL_WINDOW="${ARRIVAL_WINDOW:-30}"   # seconds to spread arrivals over
export INTER_REQUEST_GAP="${INTER_REQUEST_GAP:-0}"   # seconds between a user's A and B
export GPUS_PER_REQ="${GPUS_PER_REQ:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.5}"       # utilization sampling period P
MAX_WALL="${MAX_WALL:-1800}"                    # safety deadline (s)

# Dynamic-capacity timeline — allocations join/leave over time, like a real cluster.
# CAPACITY_SCHEDULE = ";"-separated "AT:ACTION:ARGS" events; each fires AT seconds after pool bring-up,
# concurrently with user submissions. ACTION/ARGS are passed verbatim to pool.sh.
# Example — start with 1 GPU, add 8 @60s, add 4 @120s, then drop the 8-GPU allocation @180s:
#   G=1 CAPACITY_SCHEDULE="60:inject:node-1 alloc-1 8;120:inject:node-2 alloc-2 4;180:remove:alloc-1"
CAPACITY_SCHEDULE="${CAPACITY_SCHEDULE:-}"

# ---- CAPE / runtime config ---------------------------------------------------------------------
export CAPE_HOST="${CAPE_HOST:-127.0.0.1}"; export CAPE_PORT="${CAPE_PORT:-8765}"
export CAPE_URL="${CAPE_URL:-http://$CAPE_HOST:$CAPE_PORT}"
export CAPE_POOL="${CAPE_POOL:-coding-agent-gpu}"
export CAPE_IMAGE="${CAPE_IMAGE:-docker://nvcr.io/nvidia/pytorch:26.06-py3}"
export APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
CAPE_BIN_SELECTION="${CAPE_BIN:-rscape}"
export MIMIC_DIR="$HERE"; export MIMIC_MOUNT="${MIMIC_MOUNT:-/cape-mimic}"
export AUTH="${AUTH:-on}"
[ "$AUTH" = on ] || { echo "start.sh: authentication is mandatory; AUTH=off is unsupported" >&2; exit 2; }

STAMP="${RUN_STAMP:-run}"     # pass RUN_STAMP=$(date +%Y%m%d-%H%M%S) for a unique dir
export RUN_NONCE="$STAMP"
export RESULTS_DIR="${RESULTS_DIR:-$CAPE_ROOT/evidence/cape-use-case/${SESSION_MODE}-G${G}-R$((2*U))-${STAMP}}"
export USER_ROOT="${USER_ROOT:-$RESULTS_DIR/users}"
export POOL_STATE_DIR="${POOL_STATE_DIR:-$RESULTS_DIR/pool}"
export CAPE_STATE_DIR="${CAPE_STATE_DIR:-$POOL_STATE_DIR/controller-state}"
export CAPE_ADMIN_TOKEN_FILE="${CAPE_ADMIN_TOKEN_FILE:-$POOL_STATE_DIR/secrets/admin.json}"
export SUBMITS_CSV="$RESULTS_DIR/submits.csv"
UTIL_CSV="$RESULTS_DIR/utilization_series.csv"
STATUS_CSV="$RESULTS_DIR/status_transitions.csv"
STOP_FLAG="$RESULTS_DIR/.stop_sampler"

mkdir -p "$RESULTS_DIR" "$USER_ROOT" "$POOL_STATE_DIR"
if [ "$CAPE_BIN_SELECTION" = pycape ]; then
  CAPE_BIN="$(env CAPE_SUBMIT_BIN=pycape CAPE_SUBMIT_BUNDLE="$RESULTS_DIR" \
    bash "$HERE/manual/submit.sh" --resolve-client)" || exit 2
else
  CAPE_BIN="$(cape_resolve_executable "$CAPE_BIN_SELECTION" CAPE_BIN)" || exit 2
fi
cape_inspect_executable "$CAPE_BIN_SELECTION" "$CAPE_BIN" || exit 2
export CAPE_BIN
cape_record_executable start "$CAPE_BIN" "$RESULTS_DIR/implementation.txt"
echo "uid,request,jobid,t_start,t_return,latency,session_id" > "$SUBMITS_CSV"
echo "jobid,state,t,exit_code,node_id,execution_session_id,gpu_uuids,runtime_adapter,isolation_policy,command_id" > "$STATUS_CSV"
echo "[start] G=$G U=$U R=$((2*U)) session=$SESSION_MODE arrivals=$ARRIVAL_DIST/${ARRIVAL_WINDOW}s -> $RESULTS_DIR"

# Credentials are registered online after the persistent controller becomes healthy.
export CAPE_USER_TOKENS="${CAPE_USER_TOKENS:-$POOL_STATE_DIR/user_tokens.csv}"

# ---- background utilization sampler (work-conserving invariant + queue depth) ------------------
sampler() {
  echo "t,gpus_total,gpus_leased,idle,queued,running" > "$UTIL_CSV"
  while [ ! -f "$STOP_FLAG" ]; do
    T="$(date +%s.%N)" CAPE_URL="$CAPE_URL" CAPE_POOL="$CAPE_POOL" python3 - >> "$UTIL_CSV" <<'PY' || true
import json, os, urllib.request
t = os.environ["T"]
try:
    request = urllib.request.Request("%s/v1/pools/%s/utilization" % (os.environ["CAPE_URL"], os.environ["CAPE_POOL"]),
                                     headers={"Authorization": "Bearer " + os.environ["CAPE_ADMIN_TOKEN"]})
    with urllib.request.urlopen(request, timeout=5) as r:
        d = json.load(r)
    tot, le = d.get("gpus_total", 0), d.get("gpus_leased", 0)
    print("%s,%s,%s,%s,%s,%s" % (t, tot, le, tot - le, d.get("queued_requests", 0), d.get("running_requests", 0)))
except Exception:
    print("%s,,,,," % t)
PY
    sleep "$SAMPLE_INTERVAL"
  done
}

# ---- per-user arrival delays from the chosen distribution --------------------------------------
compute_delays() {
  awk -v u="$U" -v w="$ARRIVAL_WINDOW" -v dist="$ARRIVAL_DIST" 'BEGIN{
    srand();
    if (dist=="poisson") { mean=(u>0 ? w/u : 0); cum=0;
      for (i=0;i<u;i++){ r=rand(); if(r<=0) r=1e-9; cum += -mean*log(r); printf "%.3f\n", cum } }
    else if (dist=="uniform") { for (i=0;i<u;i++) printf "%.3f\n", rand()*w }
    else { for (i=0;i<u;i++) print "0" }   # burst
  }'
}

# ---- run -----------------------------------------------------------------------------------------
bash "$HERE/pool.sh" up "$G"
export CAPE_ADMIN_TOKEN="$(cape_read_secret "$CAPE_ADMIN_TOKEN_FILE")"
: > "$CAPE_USER_TOKENS"
chmod 600 "$CAPE_USER_TOKENS"
for i in $(seq 0 $((U-1))); do
  uid="user-$(printf '%03d' "$i")"
  token_file="$POOL_STATE_DIR/secrets/$uid.json"
  cape_create_user "$uid" "$token_file"
  printf '%s,%s\n' "$uid" "$(cape_read_secret "$token_file")" >> "$CAPE_USER_TOKENS"
done
sampler & SAMPLER_PID=$!

# fire the dynamic-capacity timeline (each event runs at its offset, concurrently with submissions)
if [ -n "$CAPACITY_SCHEDULE" ]; then
  OLD_IFS="$IFS"; IFS=';'
  for ev in $CAPACITY_SCHEDULE; do
    [ -z "${ev// /}" ] && continue
    at="${ev%%:*}"; rest="${ev#*:}"; action="${rest%%:*}"; args="${rest#*:}"
    at="${at//[[:space:]]/}"; action="${action//[[:space:]]/}"
    # shellcheck disable=SC2086
    ( sleep "$at"; echo "[start] capacity @${at}s: pool.sh $action $args"; bash "$HERE/pool.sh" $action $args ) &
  done
  IFS="$OLD_IFS"
fi

# launch users at their arrival times (async submissions)
i=0; USER_PIDS=()
while IFS= read -r delay; do
  uid="user-$(printf '%03d' "$i")"
  bash "$HERE/user.sh" "$uid" "$delay" & USER_PIDS+=("$!")
  i=$((i+1))
done < <(compute_delays)

# wait for all submissions to be issued
for p in "${USER_PIDS[@]}"; do wait "$p" || true; done
echo "[start] all $U users submitted ($((2*U)) requests); waiting for the queue to drain..."

# wait until the pool is idle (running==0 && queued==0) for a couple of consecutive samples
deadline=$(( $(date +%s) + MAX_WALL ))
drained=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  read -r q r < <(tail -n 1 "$UTIL_CSV" | awk -F, '{print ($5==""?0:$5), ($6==""?0:$6)}')
  if [ "${q:-0}" = "0" ] && [ "${r:-0}" = "0" ]; then
    drained=$((drained+1)); [ "$drained" -ge 3 ] && break
  else
    drained=0
  fi
  sleep "$SAMPLE_INTERVAL"
done

collect_job_evidence() {
  tail -n +2 "$SUBMITS_CSV" | while IFS=, read -r uid label jobid _t_start _t_return _latency _session_id; do
    [ -z "$jobid" ] && continue
    local token_args=()
    if [ "$AUTH" = "on" ]; then
      tok="$(awk -F, -v u="$uid" '$1==u {print $2; found=1} END{if(!found) exit 1}' "$CAPE_USER_TOKENS")" || tok=""
      [ -n "$tok" ] && token_args=(--token "$tok")
    fi
    raw="$("$CAPE_BIN" status "$jobid" --controller-url "$CAPE_URL" "${token_args[@]}" 2>/dev/null || printf '{}')"
    T_NOW="$(date +%s.%N)" STATUS_RAW="$raw" JOBID="$jobid" python3 - <<'PY_STATUS' >> "$STATUS_CSV"
import csv, json, os, sys
try:
    data = json.loads(os.environ.get("STATUS_RAW", "{}") or "{}")
except Exception:
    data = {}
csv.writer(sys.stdout).writerow([
    os.environ.get("JOBID", ""),
    data.get("state", ""),
    os.environ.get("T_NOW", ""),
    data.get("exit_code", ""),
    data.get("node_id", ""),
    data.get("execution_session_id", ""),
    " ".join(data.get("gpu_uuids") or []),
    data.get("runtime_adapter", ""),
    data.get("isolation_policy", ""),
    data.get("command_id", ""),
])
PY_STATUS
    out_dir="$USER_ROOT/$uid/output"
    mkdir -p "$out_dir"
    "$CAPE_BIN" logs "$jobid" --controller-url "$CAPE_URL" "${token_args[@]}" > "$out_dir/${jobid}.stdout.log" 2> "$out_dir/${jobid}.stderr.log" || true
  done
}

collect_auth_evidence() {
  local out="$RESULTS_DIR/auth_probes.json"
  if [ "$AUTH" != "on" ]; then
    printf '{"status":"UNKNOWN","pass":false,"reason":"AUTH=off"}\n' > "$out"
    return 0
  fi

  local first_uid first_tok target_line target_uid target_job
  first_uid="$(tail -n +2 "$SUBMITS_CSV" | awk -F, 'NF && $1!="" {print $1; exit}')"
  target_line="$(tail -n +2 "$SUBMITS_CSV" | awk -F, -v u="$first_uid" 'NF && $1!="" && $1!=u && $3!="" {print; exit}')"
  target_uid="$(printf '%s' "$target_line" | awk -F, '{print $1}')"
  target_job="$(printf '%s' "$target_line" | awk -F, '{print $3}')"
  first_tok="$(awk -F, -v u="$first_uid" '$1==u {print $2; found=1} END{if(!found) exit 1}' "$CAPE_USER_TOKENS")" || first_tok=""

  local cross_stdout="$RESULTS_DIR/auth_cross_user.stdout"
  local cross_stderr="$RESULTS_DIR/auth_cross_user.stderr"
  local node_stdout="$RESULTS_DIR/auth_node_submit.stdout"
  local node_stderr="$RESULTS_DIR/auth_node_submit.stderr"
  local cross_rc=127 node_rc=127

  if [ -n "$first_tok" ] && [ -n "$target_job" ]; then
    set +e
    "$CAPE_BIN" status "$target_job" --controller-url "$CAPE_URL" --token "$first_tok" > "$cross_stdout" 2> "$cross_stderr"
    cross_rc="$?"
    set -e
  else
    printf 'missing cross-user probe inputs: first_uid=%s target_uid=%s target_job=%s\n' "$first_uid" "$target_uid" "$target_job" > "$cross_stderr"
  fi

  CAPE_PROBE_NODE_TOKEN="$(cape_read_secret "$POOL_STATE_DIR/secrets/agent-node-0.json")" || return 1
  set +e
  "$CAPE_BIN" run --controller-url "$CAPE_URL" --pool "$CAPE_POOL" --user node-probe --session-id node-probe --token "${CAPE_PROBE_NODE_TOKEN:-}" -- bash -lc true > "$node_stdout" 2> "$node_stderr"
  node_rc="$?"
  set -e

  CROSS_RC="$cross_rc" NODE_RC="$node_rc" FIRST_UID="$first_uid" TARGET_UID="$target_uid" TARGET_JOB="$target_job" \
  CROSS_STDOUT="$cross_stdout" CROSS_STDERR="$cross_stderr" NODE_STDOUT="$node_stdout" NODE_STDERR="$node_stderr" python3 - <<'PY_AUTH' > "$out"
import json, os
def read(path):
    try:
        with open(path) as fh:
            return fh.read()[-2000:]
    except Exception:
        return ""
cross_rc = int(os.environ.get("CROSS_RC", "127"))
node_rc = int(os.environ.get("NODE_RC", "127"))
record = {
    "status": "PASS" if cross_rc != 0 and node_rc != 0 else "FAIL",
    "pass": bool(cross_rc != 0 and node_rc != 0),
    "cross_user_status": {
        "source_user": os.environ.get("FIRST_UID"),
        "target_user": os.environ.get("TARGET_UID"),
        "target_job": os.environ.get("TARGET_JOB"),
        "return_code": cross_rc,
        "denied": cross_rc != 0,
        "stdout_tail": read(os.environ.get("CROSS_STDOUT", "")),
        "stderr_tail": read(os.environ.get("CROSS_STDERR", "")),
    },
    "node_agent_submit": {
        "return_code": node_rc,
        "denied": node_rc != 0,
        "stdout_tail": read(os.environ.get("NODE_STDOUT", "")),
        "stderr_tail": read(os.environ.get("NODE_STDERR", "")),
    },
}
print(json.dumps(record, sort_keys=True))
PY_AUTH
}

# ---- stop + collect + teardown --------------------------------------------------------------------
touch "$STOP_FLAG"; wait "$SAMPLER_PID" 2>/dev/null || true
collect_job_evidence
collect_auth_evidence
bash "$HERE/pool.sh" down || true
echo "[start] done. Evidence in $RESULTS_DIR (submits.csv, status_transitions.csv, utilization_series.csv, journal.sqlite, auth_probes.json, users/, pool/)."
echo "[start] next: run analyze.py over $RESULTS_DIR to compute §12 gates."
