#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: launch_role.sh --pid-file PATH --log-file PATH --label NAME [--timeout SEC] -- COMMAND [ARG...]

Launch one foreground CAPE role in its own session. The wrapper publishes its
pid when ready (the controller wrapper supervises its service child; the node
wrapper execs its agent). The record contains: pid execution-session-id label.
EOF
}

pid_file=
log_file=
label=
timeout=${CAPE_ROLE_START_TIMEOUT:-600}
while (($#)); do
  case "$1" in
    --pid-file) pid_file=${2:?}; shift 2 ;;
    --log-file) log_file=${2:?}; shift 2 ;;
    --label) label=${2:?}; shift 2 ;;
    --timeout) timeout=${2:?}; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "launch_role.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n $pid_file && -n $log_file && -n $label && $# -gt 0 ]] || {
  usage >&2
  exit 2
}
[[ $timeout =~ ^[0-9]+$ && $timeout -gt 0 ]] || {
  echo "launch_role.sh: timeout must be a positive integer" >&2
  exit 2
}

mkdir -p "$(dirname "$pid_file")" "$(dirname "$log_file")"
if [[ -s $pid_file ]]; then
  read -r old_pid _ < "$pid_file" || true
  if [[ ${old_pid:-} =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "launch_role.sh: $label is already recorded as live pid $old_pid" >&2
    exit 1
  fi
  rm -f "$pid_file"
fi

CAPE_ROLE_PID_FILE="$pid_file" CAPE_ROLE_LABEL="$label" \
"${CAPE_LAUNCH_PYTHON:-python3}" -c \
  'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
  "$@" >> "$log_file" 2>&1 < /dev/null &
pid=$!
published=0
cleanup() {
  if [[ $published != 1 ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT HUP INT TERM

deadline=$((SECONDS + timeout))
while ((SECONDS < deadline)); do
  if ! kill -0 "$pid" 2>/dev/null; then
    wait "$pid" 2>/dev/null || true
    echo "launch_role.sh: $label exited before becoming its CAPE service" >&2
    tail -40 "$log_file" >&2 || true
    exit 1
  fi
  if [[ -s $pid_file ]]; then
    read -r recorded_pid execution_session_id recorded_label extra < "$pid_file" || true
    if [[ $recorded_pid == "$pid" && $execution_session_id == "$pid" && $recorded_label == "$label" && -z ${extra:-} ]]; then
      sleep .1
      kill -0 "$pid" 2>/dev/null || {
        echo "launch_role.sh: $label failed during final exec" >&2
        rm -f "$pid_file"
        exit 1
      }
      published=1
      trap - EXIT HUP INT TERM
      printf '[launch] %s pid=%s session=%s\n' "$label" "$pid" "$execution_session_id"
      exit 0
    fi
  fi
  sleep .1
done

echo "launch_role.sh: timed out waiting for $label wrapper pid $pid to publish readiness" >&2
tail -40 "$log_file" >&2 || true
exit 1
