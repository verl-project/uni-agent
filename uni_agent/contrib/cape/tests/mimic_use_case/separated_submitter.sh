#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# separated_submitter.sh -- submit mimic users from a non-pool allocation.
#
# This script runs on a submitter allocation that is separate from the GPU pool
# allocation(s). It does not start node agents or a controller.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/project_root.sh"
CAPE_PROJECT_DIR="$(cape_project_root "$HERE")"

: "${CAPE_URL:?CAPE_URL is required}"
: "${CAPE_POOL:=coding-agent-gpu}"
: "${CAPE_IMAGE:=docker://nvcr.io/nvidia/pytorch:26.06-py3}"
: "${AUTH:=on}"
[ "$AUTH" = on ] || { echo "separated_submitter.sh: authentication is mandatory" >&2; exit 2; }
: "${U:=4}"
: "${G:=1}"
: "${SESSION_MODE:=shared}"
: "${ARRIVAL_DIST:=burst}"
: "${ARRIVAL_WINDOW:=0}"
: "${INTER_REQUEST_GAP:=0}"
: "${GPUS_PER_REQ:=1}"
: "${SAMPLE_INTERVAL:=0.5}"
: "${MAX_WALL:=1800}"
: "${RUN_NONCE:=separated}"
: "${RESULTS_DIR:=$CAPE_PROJECT_DIR/evidence/cape-use-case/separated-${SESSION_MODE}-G${G}-R$((2*U))-${RUN_NONCE}}"
: "${USER_ROOT:=$RESULTS_DIR/users}"
: "${CAPE_USER_TOKENS:=$RESULTS_DIR/pool/user_tokens.csv}"
: "${CAPE_PROBE_NODE_TOKEN:?CAPE_PROBE_NODE_TOKEN is required when AUTH=on}"
: "${CAPE_BIN_DIR:=/tmp/cape-separated-bin}"
: "${CAPE_BIN:=rscape}"
: "${CAPE_AGENT_LOG_BUNDLE:=$RESULTS_DIR}"
: "${EXPECTED_RUNTIME_ADAPTER:=${CAPE_REQUEST_RUNTIME_ADAPTER:-apptainer}}"
: "${EXPECTED_ISOLATION_POLICY:=${CAPE_ISOLATION_POLICY:-image-stateful}}"

export MIMIC_DIR="${MIMIC_DIR:-$HERE}"
export MIMIC_MOUNT="${MIMIC_MOUNT:-/cape-mimic}"
export SUBMITS_CSV="${SUBMITS_CSV:-$RESULTS_DIR/submits.csv}"
UTIL_CSV="$RESULTS_DIR/utilization_series.csv"
STATUS_CSV="$RESULTS_DIR/status_transitions.csv"
RUNTIME_LOG_DIR="$RESULTS_DIR/runtime-command-logs"
STOP_FLAG="$RESULTS_DIR/.stop_sampler"

mkdir -p "$RESULTS_DIR" "$USER_ROOT" "$CAPE_BIN_DIR" "$RUNTIME_LOG_DIR"
export PYTHONPATH="${PYTHONPATH:-$CAPE_PROJECT_DIR}"
CAPE_BIN_SELECTION="$CAPE_BIN"
CAPE_BIN="$(cape_resolve_executable "$CAPE_BIN_SELECTION" CAPE_BIN)" || exit 2
cape_inspect_executable "$CAPE_BIN_SELECTION" "$CAPE_BIN" || exit 2
cape_record_executable submitter "$CAPE_BIN" "$RESULTS_DIR/implementation.txt"
export CAPE_URL CAPE_POOL CAPE_IMAGE AUTH U G SESSION_MODE ARRIVAL_DIST ARRIVAL_WINDOW
export INTER_REQUEST_GAP GPUS_PER_REQ RUN_NONCE RESULTS_DIR USER_ROOT CAPE_USER_TOKENS
export CAPE_DIRECT_PATHS CAPE_BIN

echo "uid,request,jobid,t_start,t_return,latency,session_id" > "$SUBMITS_CSV"
echo "jobid,state,t,exit_code,node_id,execution_session_id,gpu_uuids,runtime_adapter,isolation_policy,command_id,log_path" > "$STATUS_CSV"
echo "[submitter] U=$U G=$G -> $RESULTS_DIR"

sampler() {
  echo "t,gpus_total,gpus_leased,idle,queued,running" > "$UTIL_CSV"
  while [ ! -f "$STOP_FLAG" ]; do
    T="$(date +%s.%N)" CAPE_URL="$CAPE_URL" CAPE_POOL="$CAPE_POOL" python3 - >> "$UTIL_CSV" <<'PY' || true
import json, os, urllib.request
t = os.environ["T"]
try:
    with open(os.environ["CAPE_USER_TOKENS"]) as stream:
        credential = next(line.strip().split(",", 1)[1] for line in stream if "," in line)
    request = urllib.request.Request("%s/v1/pools/%s/utilization" % (os.environ["CAPE_URL"], os.environ["CAPE_POOL"]),
                                     headers={"Authorization": "Bearer " + credential})
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

compute_delays() {
  awk -v u="$U" -v w="$ARRIVAL_WINDOW" -v dist="$ARRIVAL_DIST" 'BEGIN{
    srand();
    if (dist=="poisson") { mean=(u>0 ? w/u : 0); cum=0;
      for (i=0;i<u;i++){ r=rand(); if(r<=0) r=1e-9; cum += -mean*log(r); printf "%.3f\n", cum } }
    else if (dist=="uniform") { for (i=0;i<u;i++) printf "%.3f\n", rand()*w }
    else { for (i=0;i<u;i++) print "0" }
  }'
}

sampler & SAMPLER_PID=$!

i=0
USER_PIDS=()
while IFS= read -r delay; do
  uid="user-$(printf '%03d' "$i")"
  bash "$HERE/user.sh" "$uid" "$delay" & USER_PIDS+=("$!")
  i=$((i+1))
done < <(compute_delays)

for p in "${USER_PIDS[@]}"; do wait "$p" || true; done
echo "[submitter] all users submitted; waiting for drain"

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
    token_args=()
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
    data.get("log_path", ""),
])
PY_STATUS
    out_dir="$USER_ROOT/$uid/output"
    mkdir -p "$out_dir"
    "$CAPE_BIN" logs "$jobid" --controller-url "$CAPE_URL" "${token_args[@]}" > "$out_dir/${jobid}.stdout.log" 2> "$out_dir/${jobid}.stderr.log" || true
  done
}

collect_runtime_logs() {
  tail -n +2 "$STATUS_CSV" | while IFS=, read -r \
      jobid _state _t _exit_code node_id _execution_session_id _gpu_uuids \
      _runtime_adapter _isolation_policy command_id log_path; do
    [ -n "$jobid" ] && [ -n "$node_id" ] && [ -n "$command_id" ] || continue
    case "$jobid:$node_id:$command_id" in
      *[!A-Za-z0-9_.:-]*)
        printf '[submitter] refusing unsafe runtime-log identity: %s/%s/%s\n' \
          "$jobid" "$node_id" "$command_id" >&2
        continue
        ;;
    esac
    source_log="$CAPE_AGENT_LOG_BUNDLE/pool/$node_id/agent-logs/$command_id.log"
    if [ ! -r "$source_log" ] && [ -n "$log_path" ] && [ -r "$log_path" ]; then
      source_log="$log_path"
    fi
    if [ -r "$source_log" ]; then
      cp -- "$source_log" "$RUNTIME_LOG_DIR/$jobid-$node_id-$command_id.log"
    else
      printf '[submitter] runtime command log is not readable: %s\n' "$source_log" >&2
    fi
  done
}

collect_auth_evidence() {
  out="$RESULTS_DIR/auth_probes.json"
  if [ "$AUTH" != "on" ]; then
    printf '{"status":"UNKNOWN","pass":false,"reason":"AUTH=off"}\n' > "$out"
    return 0
  fi
  first_uid="$(tail -n +2 "$SUBMITS_CSV" | awk -F, 'NF && $1!="" {print $1; exit}')"
  target_line="$(tail -n +2 "$SUBMITS_CSV" | awk -F, -v u="$first_uid" 'NF && $1!="" && $1!=u && $3!="" {print; exit}')"
  target_uid="$(printf '%s' "$target_line" | awk -F, '{print $1}')"
  target_job="$(printf '%s' "$target_line" | awk -F, '{print $3}')"
  first_tok="$(awk -F, -v u="$first_uid" '$1==u {print $2; found=1} END{if(!found) exit 1}' "$CAPE_USER_TOKENS")" || first_tok=""
  cross_stdout="$RESULTS_DIR/auth_cross_user.stdout"; cross_stderr="$RESULTS_DIR/auth_cross_user.stderr"
  node_stdout="$RESULTS_DIR/auth_node_submit.stdout"; node_stderr="$RESULTS_DIR/auth_node_submit.stderr"
  cross_rc=127; node_rc=127
  if [ -n "$first_tok" ] && [ -n "$target_job" ]; then
    set +e
    "$CAPE_BIN" status "$target_job" --controller-url "$CAPE_URL" --token "$first_tok" > "$cross_stdout" 2> "$cross_stderr"
    cross_rc="$?"
    set -e
  fi
  set +e
  "$CAPE_BIN" run --controller-url "$CAPE_URL" --pool "$CAPE_POOL" --user node-probe --session-id node-probe --token "$CAPE_PROBE_NODE_TOKEN" -- bash -lc true > "$node_stdout" 2> "$node_stderr"
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
print(json.dumps({
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
}, sort_keys=True))
PY_AUTH
}

touch "$STOP_FLAG"; wait "$SAMPLER_PID" 2>/dev/null || true
collect_job_evidence
collect_runtime_logs
collect_auth_evidence
python3 "$HERE/analyze.py" "$RESULTS_DIR" \
  --busy-threshold "${BUSY_THRESHOLD:-0.9}" \
  --expected-runtime-adapter "$EXPECTED_RUNTIME_ADAPTER" \
  --expected-isolation-policy "$EXPECTED_ISOLATION_POLICY"
