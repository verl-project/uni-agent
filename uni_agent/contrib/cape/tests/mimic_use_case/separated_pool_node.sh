#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# separated_pool_node.sh -- run one CAPE pool node from a GPU allocation.
#
# This script is intended to be launched on a GPU allocation through ssh-gw.
# One node may also own the controller (`CAPE_CONTROLLER=on`). Additional pool
# nodes join the same controller URL and wait on the shared STOP_FLAG.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/project_root.sh"
source "$HERE/identity.sh"
umask 077
CAPE_PROJECT_DIR="$(cape_project_root "$HERE")"

: "${CAPE_POOL:=coding-agent-gpu}"
: "${CAPE_HOST:=0.0.0.0}"
: "${CAPE_PORT:=8765}"
: "${CAPE_URL:?CAPE_URL is required, e.g. http://pool-node:8765}"
: "${CAPE_CONTROLLER:=off}"
: "${CAPE_NODE_ID:=$(hostname -s)}"
: "${CAPE_ALLOCATION_ID:=$CAPE_NODE_ID}"
: "${CAPE_NODE_GPU_COUNT:=1}"
: "${CAPE_NODE_CPU_CORES:=$(nproc 2>/dev/null || echo 64)}"
: "${CAPE_NODE_MEMORY_GB:=$(awk '/MemTotal/ {printf "%d", $2/1024/1024; found=1} END {if (!found) printf "64"}' /proc/meminfo 2>/dev/null || echo 64)}"
: "${CAPE_RUNTIME_ADAPTERS:=}"
: "${CAPE_IMAGE:=docker://nvcr.io/nvidia/pytorch:26.06-py3}"
: "${APPTAINER_BIN:=$(command -v apptainer 2>/dev/null || true)}"
: "${BUBBLEWRAP_BIN:=$(command -v bwrap 2>/dev/null || true)}"
: "${PODMAN_BIN:=$(command -v podman 2>/dev/null || true)}"
: "${RUNTIME_CACHE:=/tmp/cape-separated-cache}"
: "${RUNTIME_TMP:=/tmp/cape-separated-tmp}"
: "${RUNTIME_DIR:=/tmp/cape-separated-runtime}"
: "${RESULTS_DIR:?RESULTS_DIR is required}"
: "${POOL_STATE_DIR:=$RESULTS_DIR/pool/$CAPE_NODE_ID}"
: "${CAPE_STATE_DIR:=$RESULTS_DIR/pool/controller-state}"
: "${CAPE_ADMIN_TOKEN_FILE:=$RESULTS_DIR/pool/secrets/admin.json}"
: "${CAPE_NODE_TOKEN:=}"
: "${STOP_FLAG:=$RESULTS_DIR/.stop_pool}"
: "${CAPE_BIN_DIR:=/tmp/cape-separated-bin}"
: "${CAPE_BIN:=rscape}"
: "${CAPE_AGENT_STATE_PATH:=$POOL_STATE_DIR/agent-state.json}"
: "${CAPE_AGENT_CLEANUP_TIMEOUT:=30}"
: "${CAPE_HEARTBEAT_INTERVAL:=1}"

mkdir -p \
  "$POOL_STATE_DIR" \
  "$RUNTIME_CACHE/$CAPE_NODE_ID" \
  "$RUNTIME_TMP/$CAPE_NODE_ID" \
  "$RUNTIME_DIR/$CAPE_NODE_ID" \
  "$CAPE_BIN_DIR"
export PYTHONPATH="${PYTHONPATH:-$CAPE_PROJECT_DIR}"
CAPE_BIN_SELECTION="$CAPE_BIN"
CAPE_BIN="$(cape_resolve_executable "$CAPE_BIN_SELECTION" CAPE_BIN)" || exit 2
cape_inspect_executable "$CAPE_BIN_SELECTION" "$CAPE_BIN" || exit 2
cape_record_executable pool-node "$CAPE_BIN" "$POOL_STATE_DIR/implementation.txt"

log() { printf '[pool-node %s] %s\n' "$CAPE_NODE_ID" "$*" >&2; }

gpu_args=()
if [ -n "${CAPE_GPU_UUIDS:-}" ]; then
  for uuid in $CAPE_GPU_UUIDS; do
    gpu_args+=(--gpu-uuid "$uuid")
  done
else
  while IFS= read -r uuid; do
    [ -n "$uuid" ] && gpu_args+=(--gpu-uuid "$uuid")
    [ "${#gpu_args[@]}" -ge $((2 * CAPE_NODE_GPU_COUNT)) ] && break
  done < <(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | sed 's/^ *//;s/ *$//')
fi

if [ "${#gpu_args[@]}" -eq 0 ]; then
  log "no GPU UUIDs discovered"
  exit 2
fi

controller_pid=""
agent_pid=""

agent_has_no_sessions() {
  [ -s "$CAPE_AGENT_STATE_PATH" ] &&
    grep -Eq '"local_sessions"[[:space:]]*:[[:space:]]*\[[[:space:]]*\]' "$CAPE_AGENT_STATE_PATH"
}

wait_for_agent_cleanup() {
  local deadline
  deadline=$(( $(date +%s) + CAPE_AGENT_CLEANUP_TIMEOUT ))
  while [ "$(date +%s)" -le "$deadline" ]; do
    agent_has_no_sessions && return 0
    [ -n "$agent_pid" ] && kill -0 "$agent_pid" >/dev/null 2>&1 || return 1
    sleep 0.1
  done
  return 1
}

cleanup() {
  cleanup_status="$?"
  trap - EXIT INT TERM
  set +e
  log "cleanup"
  if CAPE_TOKEN="$CAPE_NODE_TOKEN" "$CAPE_BIN" node drain "$CAPE_NODE_ID" --controller-url "$CAPE_URL" --force >/dev/null 2>&1; then
    if [ -n "$agent_pid" ] && ! wait_for_agent_cleanup; then
      log "agent did not acknowledge session cleanup within ${CAPE_AGENT_CLEANUP_TIMEOUT}s"
      cleanup_status=1
    fi
  else
    log "allocation removal failed"
    cleanup_status=1
  fi
  if [ -n "$agent_pid" ]; then
    kill "$agent_pid" >/dev/null 2>&1
    wait "$agent_pid" >/dev/null 2>&1
  fi
  if [ -n "$controller_pid" ]; then
    kill "$controller_pid" >/dev/null 2>&1
    wait "$controller_pid" >/dev/null 2>&1
  fi
  exit "$cleanup_status"
}
trap cleanup EXIT INT TERM

if [ "$CAPE_CONTROLLER" = "on" ]; then
  log "starting controller on $CAPE_HOST:$CAPE_PORT"
  cape_init_instance "$CAPE_STATE_DIR" "$CAPE_ADMIN_TOKEN_FILE"
  "$CAPE_BIN" controller start --host "$CAPE_HOST" --port "$CAPE_PORT" --state-dir "$CAPE_STATE_DIR" --token-file "$CAPE_ADMIN_TOKEN_FILE" > "$POOL_STATE_DIR/controller.log" 2>&1 &
  controller_pid="$!"
  for _ in $(seq 1 100); do
    curl -sf "http://127.0.0.1:$CAPE_PORT/health" >/dev/null 2>&1 && break
    sleep 0.2
  done
  cape_ensure_pool "$CAPE_POOL" "$CAPE_ADMIN_TOKEN_FILE"
  if [ -z "$CAPE_NODE_TOKEN" ]; then
    contributor_file="$(dirname "$CAPE_ADMIN_TOKEN_FILE")/contributor.json"
    cape_create_user contributor "$contributor_file"
    CAPE_NODE_TOKEN="$(cape_read_secret "$contributor_file")"
  fi
fi
[ -n "$CAPE_NODE_TOKEN" ] || { log "CAPE_NODE_TOKEN is required"; exit 2; }

runtime_filter_text="${CAPE_RUNTIME_ADAPTERS//,/ }"
read -r -a runtime_filters <<< "$runtime_filter_text"
runtime_args=()
for runtime_filter in "${runtime_filters[@]}"; do
  case "$runtime_filter" in
    local-process|apptainer|bubblewrap|podman) ;;
    *) log "invalid CAPE_RUNTIME_ADAPTERS entry: $runtime_filter"; exit 2 ;;
  esac
  runtime_args+=(--runtime-adapter "$runtime_filter")
done
runtime_args+=(
  --runtime-cache "$RUNTIME_CACHE/$CAPE_NODE_ID"
  --runtime-tmp "$RUNTIME_TMP/$CAPE_NODE_ID"
  --runtime-dir "$RUNTIME_DIR/$CAPE_NODE_ID"
  --apptainer-probe-image "$CAPE_IMAGE"
  --podman-probe-image "$CAPE_IMAGE"
)
[ -z "$APPTAINER_BIN" ] || runtime_args+=(--apptainer-binary "$APPTAINER_BIN")
[ -z "$BUBBLEWRAP_BIN" ] || runtime_args+=(--bubblewrap-binary "$BUBBLEWRAP_BIN")
[ -z "$PODMAN_BIN" ] || runtime_args+=(--podman-binary "$PODMAN_BIN")
[ "${APPTAINER_UNSQUASH:-off}" != on ] || runtime_args+=(--apptainer-unsquash)

log "joining allocation=$CAPE_ALLOCATION_ID cpu=$CAPE_NODE_CPU_CORES memory_gb=$CAPE_NODE_MEMORY_GB gpus=${gpu_args[*]} runtimes=${CAPE_RUNTIME_ADAPTERS:-auto}"
mkdir -p "$POOL_STATE_DIR/secrets"
chmod 700 "$POOL_STATE_DIR/secrets"
CAPE_TOKEN="$CAPE_NODE_TOKEN" "$CAPE_BIN" node enroll \
  --controller-url "$CAPE_URL" --node-id "$CAPE_NODE_ID" --pool "$CAPE_POOL" \
  --allocation-id "$CAPE_ALLOCATION_ID" \
  --cpu-cores "$CAPE_NODE_CPU_CORES" --memory-gb "$CAPE_NODE_MEMORY_GB" \
  "${runtime_args[@]}" \
  --log-dir "$POOL_STATE_DIR/agent-logs" --state-path "$CAPE_AGENT_STATE_PATH" \
  --heartbeat-interval "$CAPE_HEARTBEAT_INTERVAL" \
  --node-token-out "$POOL_STATE_DIR/secrets/enrollment.json" \
  "${gpu_args[@]}" > "$POOL_STATE_DIR/agent.log" 2>&1 &
agent_pid="$!"

log "ready; waiting for $STOP_FLAG"
while [ ! -f "$STOP_FLAG" ]; do
  sleep 1
done
