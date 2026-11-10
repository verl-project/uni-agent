#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# pool.sh — configure the CAPE hardware pool and mutate it at run time.
#
# Mimics real dynamic capacity: a controller plus one or more node agents that join GPUs into the
# pool, with the ability to INJECT new capacity and DRAIN/REMOVE capacity mid-run (allocations come
# and go). On a single physical box we model "multiple nodes/allocations" by running several agents,
# each owning a DISJOINT subset of the box's GPU UUIDs.
#
#   pool.sh up <G>                       start controller + join the first <G> GPUs as node-0/alloc-0
#   pool.sh inject <node_id> <alloc_id> <n>   join the next <n> unused GPUs as a new node/allocation
#   pool.sh drain <alloc_id>             drain allocation (no new placement; finish in-flight)
#   pool.sh remove <alloc_id>            remove allocation (capacity disappears)
#   pool.sh down                         remove all allocations, stop agents + controller
#
# Env: CAPE_URL CAPE_POOL CAPE_RUNTIME_ADAPTERS APPTAINER_BIN BUBBLEWRAP_BIN PODMAN_BIN RUNTIME_CACHE RUNTIME_TMP POOL_STATE_DIR CAPE_STATE_DIR CAPE_ADMIN_TOKEN_FILE CAPE_CONTRIBUTOR_TOKEN_FILE
set -euo pipefail
source "$(dirname "$0")/identity.sh"
umask 077

: "${CAPE_URL:=http://127.0.0.1:8765}"
: "${CAPE_HOST:=127.0.0.1}"
: "${CAPE_PORT:=8765}"
: "${CAPE_POOL:=coding-agent-gpu}"
: "${CAPE_RUNTIME_ADAPTERS:=}"
: "${APPTAINER_BIN:=$(command -v apptainer 2>/dev/null || true)}"
: "${BUBBLEWRAP_BIN:=$(command -v bwrap 2>/dev/null || true)}"
: "${PODMAN_BIN:=$(command -v podman 2>/dev/null || true)}"
: "${APPTAINER_UNSQUASH:=off}"
: "${CAPE_APPTAINER_PROBE_IMAGE:=${CAPE_IMAGE:-}}"
: "${CAPE_PODMAN_PROBE_IMAGE:=${CAPE_IMAGE:-}}"
: "${RUNTIME_CACHE:=/tmp/cape-apptainer-cache-$$}"
: "${RUNTIME_TMP:=/tmp/cape-apptainer-tmp-$$}"
: "${POOL_STATE_DIR:=$PWD/.cape-pool-state}"
: "${CAPE_STATE_DIR:=$POOL_STATE_DIR/controller-state}"
: "${CAPE_ADMIN_TOKEN_FILE:=$POOL_STATE_DIR/secrets/admin.json}"
: "${CAPE_CONTRIBUTOR_TOKEN_FILE:=$POOL_STATE_DIR/secrets/contributor.json}"
: "${CAPE_BIN:=rscape}"
case "$CAPE_BIN" in
  */*) [ -x "$CAPE_BIN" ] || { echo "pool.sh: CAPE_BIN is not executable: $CAPE_BIN" >&2; exit 2; } ;;
  *) CAPE_BIN="$(command -v "$CAPE_BIN")" || { echo "pool.sh: CAPE_BIN is not on PATH" >&2; exit 2; } ;;
esac

mkdir -p "$POOL_STATE_DIR" "$RUNTIME_CACHE" "$RUNTIME_TMP" "$POOL_STATE_DIR/secrets"
chmod 700 "$POOL_STATE_DIR/secrets"
USED_GPUS="$POOL_STATE_DIR/used_gpus.txt"; touch "$USED_GPUS"
AGENT_PIDS="$POOL_STATE_DIR/agents.pids"
ALLOCS="$POOL_STATE_DIR/allocations.txt"
CTRL_PID="$POOL_STATE_DIR/controller.pid"

node_for_allocation() {
  awk -F, -v id="$1" '$1==id {print $2; found=1} END {if(!found) exit 1}' "$POOL_STATE_DIR/node-allocations.csv"
}

all_gpu_uuids() {
  if [ -n "${CAPE_GPU_UUIDS:-}" ]; then
    printf '%s\n' $CAPE_GPU_UUIDS
  else
    nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | sed 's/^ *//;s/ *$//'
  fi
}

# echo the next <n> GPU UUIDs not already recorded in USED_GPUS, and mark them used.
take_gpus() {
  local n="$1" taken=0 u
  for u in $(all_gpu_uuids); do
    grep -qxF "$u" "$USED_GPUS" && continue
    echo "$u"; echo "$u" >> "$USED_GPUS"
    taken=$((taken+1)); [ "$taken" -ge "$n" ] && break
  done
  [ "$taken" -lt "$n" ] && echo "pool.sh: only $taken free GPUs (< $n requested)" >&2
}

join_node() {
  # join_node <node_id> <alloc_id> <gpu_uuid...>
  local nid="$1" aid="$2"; shift 2
  local gpu_args=(); local u; for u in "$@"; do gpu_args+=(--gpu-uuid "$u"); done
  local runtime_filter_text="${CAPE_RUNTIME_ADAPTERS//,/ }"
  local runtime_filter
  local -a runtime_filters=()
  local -a runtime_args=(
    --runtime-cache "$RUNTIME_CACHE/$nid"
    --runtime-tmp "$RUNTIME_TMP/$nid"
  )
  read -r -a runtime_filters <<< "$runtime_filter_text"
  for runtime_filter in "${runtime_filters[@]}"; do
    case "$runtime_filter" in
      local-process|apptainer|bubblewrap|podman) ;;
      *) echo "pool.sh: invalid CAPE_RUNTIME_ADAPTERS entry: $runtime_filter" >&2; return 2 ;;
    esac
    runtime_args+=(--runtime-adapter "$runtime_filter")
  done
  [ -z "$APPTAINER_BIN" ] || runtime_args+=(--apptainer-binary "$APPTAINER_BIN")
  [ -z "$BUBBLEWRAP_BIN" ] || runtime_args+=(--bubblewrap-binary "$BUBBLEWRAP_BIN")
  [ -z "$PODMAN_BIN" ] || runtime_args+=(--podman-binary "$PODMAN_BIN")
  [ -z "$CAPE_APPTAINER_PROBE_IMAGE" ] || runtime_args+=(--apptainer-probe-image "$CAPE_APPTAINER_PROBE_IMAGE")
  [ -z "$CAPE_PODMAN_PROBE_IMAGE" ] || runtime_args+=(--podman-probe-image "$CAPE_PODMAN_PROBE_IMAGE")
  [ "$APPTAINER_UNSQUASH" != on ] || runtime_args+=(--apptainer-unsquash)
  echo "[pool] join node=$nid alloc=$aid gpus=$* "
  # The node agent runs a long-lived heartbeat/command-pull loop -> background it.
  "$CAPE_BIN" node enroll \
    --controller-url "$CAPE_URL" --node-id "$nid" --pool "$CAPE_POOL" \
    --allocation-id "$aid" \
    "${runtime_args[@]}" \
    --log-dir "$POOL_STATE_DIR/agent-logs/$nid" --state-path "$POOL_STATE_DIR/agent-state-$nid.json" \
    --token-file "$CAPE_CONTRIBUTOR_TOKEN_FILE" --node-token-out "$POOL_STATE_DIR/secrets/agent-$nid.json" "${gpu_args[@]}" &
  echo "$!" >> "$AGENT_PIDS"
  printf '%s,%s\n' "$aid" "$nid" >> "$POOL_STATE_DIR/node-allocations.csv"
  grep -qxF "$aid" "$ALLOCS" 2>/dev/null || echo "$aid" >> "$ALLOCS"
}

wait_healthy() {
  local i
  for i in $(seq 1 50); do
    curl -sf "$CAPE_URL/health" >/dev/null 2>&1 && return 0
    sleep 0.2
  done
  echo "pool.sh: controller did not become healthy at $CAPE_URL" >&2; return 1
}

cmd="${1:?usage: pool.sh up|inject|drain|remove|down ...}"; shift || true
case "$cmd" in
  up)
    G="${1:?usage: pool.sh up <G>}"
    echo "[pool] starting controller on $CAPE_HOST:$CAPE_PORT"
    cape_init_instance "$CAPE_STATE_DIR" "$CAPE_ADMIN_TOKEN_FILE"
    "$CAPE_BIN" controller start --host "$CAPE_HOST" --port "$CAPE_PORT" --state-dir "$CAPE_STATE_DIR" --token-file "$CAPE_ADMIN_TOKEN_FILE" &
    echo "$!" > "$CTRL_PID"
    wait_healthy
    cape_ensure_pool "$CAPE_POOL" "$CAPE_ADMIN_TOKEN_FILE"
    cape_create_user contributor "$CAPE_CONTRIBUTOR_TOKEN_FILE"
    # shellcheck disable=SC2046
    join_node "node-0" "alloc-0" $(take_gpus "$G")
    ;;
  inject)
    nid="${1:?node_id}"; aid="${2:?alloc_id}"; n="${3:?n}"
    # shellcheck disable=SC2046
    join_node "$nid" "$aid" $(take_gpus "$n")
    ;;
  drain)
    aid="${1:?alloc_id}"
    echo "[pool] draining allocation $aid"
    "$CAPE_BIN" node drain "$(node_for_allocation "$aid")" --controller-url "$CAPE_URL" --token-file "$CAPE_ADMIN_TOKEN_FILE"
    ;;
  remove)
    aid="${1:?alloc_id}"
    echo "[pool] removing allocation $aid"
    "$CAPE_BIN" node exit "$(node_for_allocation "$aid")" --controller-url "$CAPE_URL" --token-file "$CAPE_ADMIN_TOKEN_FILE"
    ;;
  down)
    echo "[pool] tearing down"
    if [ -f "$ALLOCS" ]; then
      while read -r aid; do
        "$CAPE_BIN" node drain "$(node_for_allocation "$aid")" --controller-url "$CAPE_URL" --token-file "$CAPE_ADMIN_TOKEN_FILE" || true
        "$CAPE_BIN" node exit "$(node_for_allocation "$aid")" --controller-url "$CAPE_URL" --token-file "$CAPE_ADMIN_TOKEN_FILE" || true
      done < "$ALLOCS"
    fi
    # Agents treat SIGTERM as a graceful-drain request and must contact the
    # controller to finish it.  Keep the controller alive while they observe
    # the authoritative allocation removal and exit; killing both sides back
    # to back strands an agent in its heartbeat retry loop.
    if [ -f "$AGENT_PIDS" ]; then
      while read -r p; do kill "$p" 2>/dev/null || true; done < "$AGENT_PIDS"
      for _ in $(seq 1 50); do
        alive=0
        while read -r p; do kill -0 "$p" 2>/dev/null && alive=1; done < "$AGENT_PIDS"
        [ "$alive" -eq 0 ] && break
        sleep 0.1
      done
      while read -r p; do
        if kill -0 "$p" 2>/dev/null; then
          echo "[pool] agent pid=$p exceeded shutdown grace; forcing exit" >&2
          kill -KILL "$p" 2>/dev/null || true
        fi
      done < "$AGENT_PIDS"
    fi
    if [ -f "$CTRL_PID" ]; then
      controller_pid="$(cat "$CTRL_PID")"
      kill "$controller_pid" 2>/dev/null || true
      for _ in $(seq 1 50); do
        kill -0 "$controller_pid" 2>/dev/null || break
        sleep 0.1
      done
      if kill -0 "$controller_pid" 2>/dev/null; then
        echo "[pool] controller pid=$controller_pid exceeded shutdown grace; forcing exit" >&2
        kill -KILL "$controller_pid" 2>/dev/null || true
      fi
    fi
    rm -f "$AGENT_PIDS" "$CTRL_PID" "$ALLOCS" "$USED_GPUS"
    ;;
  *)
    echo "pool.sh: unknown command '$cmd'" >&2; exit 2;;
esac
