#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
log_dir=$SCRIPT_DIR/logs
timeout=${CAPE_SHUTDOWN_TIMEOUT:-120}
settle=${CAPE_SHUTDOWN_SETTLE:-15}
force=${CAPE_SHUTDOWN_FORCE:-0}
strict_host_audit=${CAPE_SHUTDOWN_STRICT_HOST_AUDIT:-0}

usage() {
  cat <<'EOF'
Usage: shutdown.sh [--log-dir PATH] [--timeout SEC] [--settle SEC] [--force] [--strict-host-audit]

Order: drain/retire nodes through the controller, wait for agent-owned cleanup,
then stop only recorded role sessions. Unrecorded CAPE processes are reported
but never signalled automatically. --strict-host-audit treats any other CAPE
process owned by this OS user as an error; default cleanup is run-scoped.
EOF
}

while (($#)); do
  case "$1" in
    --log-dir) log_dir=${2:?}; shift 2 ;;
    --timeout) timeout=${2:?}; shift 2 ;;
    --settle) settle=${2:?}; shift 2 ;;
    --force) force=1; shift ;;
    --strict-host-audit) strict_host_audit=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "shutdown.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
for value in "$timeout" "$settle"; do
  [[ $value =~ ^[0-9]+$ ]] || { echo "shutdown.sh: bounds must be nonnegative integers" >&2; exit 2; }
done

self_pid=$$
read -r self_pgid self_sid < <(python3 -c 'import os; print(os.getpgrp(), os.getsid(0))')
declare -A owned_pids=()
declare -A ports=()

read_record() {
  local file=$1
  local pid sid label extra
  [[ -s $file ]] || return 1
  read -r pid sid label extra < "$file" || return 1
  [[ $pid =~ ^[0-9]+$ && $sid =~ ^[0-9]+$ && -n $label && -z ${extra:-} ]] || return 1
  [[ $pid == "$sid" ]] || return 1
  printf '%s %s %s\n' "$pid" "$sid" "$label"
}

record_is_live() {
  local pid=$1
  kill -0 "$pid" 2>/dev/null
}

echo "[shutdown] 1/3 drain and retire through the controller"
# Only retire memberships launched by this local workflow, never every pool node.
for role in exclusive shared; do
  record=$(read_record "$log_dir/${role}_node.pid" 2>/dev/null || true)
  [[ -n $record ]] || continue
  membership=$(python3 "$SCRIPT_DIR/role_record.py" read \
    --pid-file "$log_dir/${role}_node.pid" --controller-url "${CAPE_URL:?Set CAPE_URL}") || exit 1
  mapfile -t fields <<< "$membership"
  [[ ${#fields[@]} == 2 ]] || exit 1
  args=(--controller-url "$CAPE_URL" --pool "${fields[0]}" --node-id "${fields[1]}" --timeout "$timeout")
  [[ $force != 1 ]] || args+=(--force)
  bash "$SCRIPT_DIR/retire_node.sh" --token-file "${CAPE_USER_TOKEN_FILE:?Set CAPE_USER_TOKEN_FILE}" \
    "${args[@]}" || exit 1
done

for file in "$log_dir/controller.pid" "$log_dir/exclusive_node.pid" "$log_dir/shared_node.pid"; do
  record=$(read_record "$file" 2>/dev/null || true)
  [[ -n $record ]] || continue
  read -r pid sid label <<< "$record"
  owned_pids["$pid"]=1
done

stop_recorded_role() {
  local file=$1 role=$2 grace=$3
  local record pid sid label pgid
  record=$(read_record "$file" 2>/dev/null || true)
  if [[ -z $record ]]; then
    [[ -e $file ]] && echo "[shutdown] ignoring invalid $role pid record: $file"
    rm -f "$file"
    return 0
  fi
  read -r pid sid label <<< "$record"
  if ! record_is_live "$pid"; then
    echo "[shutdown] $role already exited or pid was reused"
    rm -f "$file"
    return 0
  fi
  pgid=$pid
  if [[ $pid == "$self_pid" || $pgid == "$self_pgid" || $sid == "$self_sid" ]]; then
    echo "[shutdown] REFUSING to signal own process/session for $role pid=$pid pgid=$pgid sid=$sid" >&2
    return 1
  fi
  if [[ $pgid != "$pid" || $sid != "$pid" ]]; then
    echo "[shutdown] REFUSING unowned process shape for $role pid=$pid pgid=$pgid sid=$sid" >&2
    return 1
  fi
  echo "[shutdown] SIGTERM $role session $sid"
  kill -TERM -- "-$pgid" 2>/dev/null || true
  for ((second=0; second<grace; second++)); do
    record_is_live "$pid" || break
    sleep 1
  done
  if record_is_live "$pid"; then
    echo "[shutdown] SIGKILL $role session $sid after ${grace}s"
    kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
  rm -f "$file"
}

echo "[shutdown] 2/3 wait for retired agents, then stop recorded sessions"
for ((second=0; second<settle; second++)); do
  live=0
  for file in "$log_dir/exclusive_node.pid" "$log_dir/shared_node.pid"; do
    record=$(read_record "$file" 2>/dev/null || true)
    [[ -n $record ]] || continue
    read -r pid _ <<< "$record"
    record_is_live "$pid" && live=1
  done
  [[ $live == 0 ]] && break
  sleep 1
done
stop_recorded_role "$log_dir/exclusive_node.pid" exclusive-node 15
stop_recorded_role "$log_dir/shared_node.pid" shared-node 15
stop_recorded_role "$log_dir/controller.pid" controller 15

echo "[shutdown] 3/3 audit ports and unrecorded CAPE roles"
failed=0
for port in "${!ports[@]}"; do
  if command -v ss >/dev/null 2>&1; then
    listeners=$(ss -lntp 2>/dev/null | awk -v needle=":$port" '$4 ~ needle {print}' || true)
  elif python3 - "$port" <<'PY'
import socket, sys
sock = socket.socket()
sock.settimeout(.2)
try:
    listening = sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0
finally:
    sock.close()
raise SystemExit(0 if listening else 1)
PY
  then
    listeners="TCP connection accepted on 127.0.0.1:$port (ss unavailable)"
  else
    listeners=
  fi
  if [[ -n $listeners ]]; then
    echo "[shutdown] listener remains on CAPE port $port: $listeners" >&2
    failed=1
  fi
done

while read -r pid pgid sid command_line; do
  [[ -n ${pid:-} ]] || continue
  [[ $pid == "$self_pid" || $pgid == "$self_pgid" || $sid == "$self_sid" ]] && continue
  [[ -n ${owned_pids[$pid]:-} ]] && continue
  case "$command_line" in
    *rscape*" controller start"*|*pycape*" controller start"*|*rscape*" node enroll"*|*pycape*" node enroll"*)
    echo "[shutdown] unmanaged CAPE role remains: pid=$pid pgid=$pgid sid=$sid command=$command_line" >&2
    [[ $strict_host_audit == 1 ]] || continue
    if [[ $pgid == "$sid" && $pgid != "$self_pgid" ]]; then
      echo "[shutdown] stop explicitly with: kill -TERM -- -$pgid" >&2
    else
      echo "[shutdown] stop explicitly with: kill -TERM $pid" >&2
    fi
      failed=1
      ;;
  esac
done < <(ps -u "$(id -u)" -o pid=,pgid=,sid=,args=)

if [[ $failed == 0 ]]; then
  echo "[shutdown] complete; owned roles stopped; unrelated roles were not signalled"
else
  echo "[shutdown] incomplete; review exact diagnostics above" >&2
  exit 1
fi
