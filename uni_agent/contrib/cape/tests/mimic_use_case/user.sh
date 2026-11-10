#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# user.sh — submit one user's two requests asynchronously through CAPE_BIN.
#
#   usage: user.sh <uid> [arrival_delay_s]
#
# Arrival timing: start.sh computes per-user `arrival_delay_s` from the chosen distribution
# (burst / uniform / poisson) so the U users hit the pool spread over time — mimicking real
# arrivals. Within a user, the two requests are gapped by INTER_REQUEST_GAP.
#
# Each run is fired without --wait: it returns a request id immediately. We record the
# submit-return latency + jobid into $SUBMITS_CSV for the analyzer.
#
# Inputs via env (defaults sensible for a single-node run; start.sh exports the real values):
#   CAPE_URL CAPE_POOL CAPE_IMAGE  — controller endpoint, pool, container image
#   MIMIC_DIR                      — host path of this dir (bound read-only into the container)
#   MIMIC_MOUNT                    — in-container path for MIMIC_DIR        (default /cape-mimic)
#   USER_ROOT                      — shared-storage root for per-user dirs  (default ./.cape-mimic-users)
#   SESSION_MODE                   — shared (state-layer reuse) | distinct   (default shared)
#   INTER_REQUEST_GAP              — seconds between a user's A and B         (default 0)
#   GPUS_PER_REQ                   — GPUs per request                         (default 1)
#   RUN_NONCE                      — run id; run_tag = "<uid>@<RUN_NONCE>"
#   SUBMITS_CSV                    — append target (uid,request,jobid,t_start,t_return,latency,session_id)
#   AUTH                           — on|off; if on, CAPE_USER_TOKENS maps uid -> submitter token
#   CAPE_REQUEST_RUNTIME_ADAPTER   — optional explicit adapter for submitted requests
#   CAPE_ISOLATION_POLICY          — request policy; defaults to none for direct paths, image-stateful otherwise
set -euo pipefail

UID_="${1:?usage: user.sh <uid> [arrival_delay_s]}"
ARRIVAL_DELAY="${2:-0}"

: "${CAPE_URL:=http://127.0.0.1:8765}"
: "${CAPE_POOL:=coding-agent-gpu}"
: "${CAPE_IMAGE:=docker://nvcr.io/nvidia/pytorch:26.06-py3}"
: "${MIMIC_DIR:=$(cd "$(dirname "$0")" && pwd)}"
: "${MIMIC_MOUNT:=/cape-mimic}"
: "${USER_ROOT:=$PWD/.cape-mimic-users}"
: "${SESSION_MODE:=shared}"
: "${INTER_REQUEST_GAP:=0}"
: "${GPUS_PER_REQ:=1}"
: "${RUN_NONCE:=$$}"
: "${SUBMITS_CSV:=/dev/null}"
: "${AUTH:=on}"
[ "$AUTH" = on ] || { echo "user.sh: authentication is mandatory" >&2; exit 2; }
: "${CAPE_USER_TOKENS:=}"
: "${CAPE_DIRECT_PATHS:=off}"
: "${CAPE_BIN:=rscape}"
case "$CAPE_BIN" in
  */*) [ -x "$CAPE_BIN" ] || { echo "user.sh: CAPE_BIN is not executable: $CAPE_BIN" >&2; exit 2; } ;;
  *) CAPE_BIN="$(command -v "$CAPE_BIN")" || { echo "user.sh: CAPE_BIN is not on PATH" >&2; exit 2; } ;;
esac
if [ -z "${CAPE_REQUEST_RUNTIME_ADAPTER+x}" ]; then
  if [ "$CAPE_DIRECT_PATHS" = "on" ]; then
    CAPE_REQUEST_RUNTIME_ADAPTER="local-process"
  else
    CAPE_REQUEST_RUNTIME_ADAPTER=""
  fi
fi
if [ -z "${CAPE_ISOLATION_POLICY+x}" ]; then
  if [ "$CAPE_DIRECT_PATHS" = "on" ]; then
    CAPE_ISOLATION_POLICY="none"
  else
    CAPE_ISOLATION_POLICY="image-stateful"
  fi
fi

runtime_args=()
[ -z "$CAPE_REQUEST_RUNTIME_ADAPTER" ] || runtime_args=(--runtime-adapter "$CAPE_REQUEST_RUNTIME_ADAPTER")

run_tag="${UID_}@${RUN_NONCE}"
user_ws="$USER_ROOT/$UID_/workspace"
user_out="$USER_ROOT/$UID_/output"
mkdir -p "$user_ws" "$user_out"

# Optional auth: look up this user's submitter token from the controller-start seed file.
token_args=()
if [ "$AUTH" = "on" ]; then
  tok="$(awk -F, -v u="$UID_" '$1==u {print $2; found=1} END{if(!found) exit 1}' "$CAPE_USER_TOKENS")" || {
    echo "user.sh: missing token for $UID_ in CAPE_USER_TOKENS=$CAPE_USER_TOKENS" >&2; exit 2;
  }
  token_args=(--token "$tok")
fi

# Sleep this user's arrival delay so the fleet follows the configured time distribution.
if [ "$(awk "BEGIN{print ($ARRIVAL_DELAY>0)}")" = "1" ]; then
  sleep "$ARRIVAL_DELAY"
fi

submit() {
  # submit <request_label> <kind> <session_id>
  local label="$1" kind="$2" skey="$3"
  local t_start t_return latency out jobid
  t_start="$(date +%s.%N)"
  if [ "$CAPE_DIRECT_PATHS" = "on" ]; then
    out="$("$CAPE_BIN" run \
        --controller-url "$CAPE_URL" --pool "$CAPE_POOL" \
        --user "$UID_" --session-id "$skey" \
        --image "$CAPE_IMAGE" --isolation-policy "$CAPE_ISOLATION_POLICY" --gpus "$GPUS_PER_REQ" \
        "${runtime_args[@]}" \
        "${token_args[@]}" \
        --cwd "$user_ws" \
        -- bash "$MIMIC_DIR/workload.sh" "$kind" "$run_tag" "$user_out")"
  else
    out="$("$CAPE_BIN" run \
        --controller-url "$CAPE_URL" --pool "$CAPE_POOL" \
        --user "$UID_" --session-id "$skey" \
        --image "$CAPE_IMAGE" --isolation-policy "$CAPE_ISOLATION_POLICY" --gpus "$GPUS_PER_REQ" \
        "${runtime_args[@]}" \
        "${token_args[@]}" \
        --bind "$MIMIC_DIR:$MIMIC_MOUNT:ro" \
        --bind "$user_out:/output:rw" \
        --workspace-target "${CAPE_MANAGED_WORKSPACE_TARGET:-/cape-workspace}" \
        --bind "$user_ws:/workspace:rw" --cwd /workspace \
        -- bash "$MIMIC_MOUNT/workload.sh" "$kind" "$run_tag" /output)"
  fi
  t_return="$(date +%s.%N)"
  jobid="$(printf '%s' "$out" | tr -d '\r' | awk 'NF{print $1; exit}')"
  latency="$(awk "BEGIN{printf \"%.6f\", $t_return-$t_start}")"
  echo "$UID_,$label,$jobid,$t_start,$t_return,$latency,$skey" >> "$SUBMITS_CSV"
  echo "[user $UID_] $label -> jobid=$jobid latency=${latency}s session_id=$skey"
}

if [ "$SESSION_MODE" = "shared" ]; then
  ka="$UID_/task"; kb="$UID_/task"     # same key -> B reuses A's persistent state layer
else
  ka="$UID_/task-A"; kb="$UID_/task-B"  # distinct keys -> independent sessions/leases
fi

submit A gemm_silu "$ka"
if [ "$(awk "BEGIN{print ($INTER_REQUEST_GAP>0)}")" = "1" ]; then
  sleep "$INTER_REQUEST_GAP"
fi
submit B cutedsl_add "$kb"
