#!/usr/bin/env bash
set -uo pipefail

if (( EUID != 0 )); then
  exec /usr/bin/sudo -n -H /opt/triton-agent-tools/cleanup_task_processes.sh
fi

state_dir="${PWD}/.triton_verify_processes"
shopt -s nullglob

signal_registered() {
  local signal=$1 state wrapper kind child
  for state in "${state_dir}"/*.state; do
    read -r wrapper kind child <"${state}" || continue
    [[ "${wrapper}" =~ ^[0-9]+$ && "${wrapper}" -gt 1 ]] && kill "-${signal}" "${wrapper}" 2>/dev/null || true
    [[ "${kind}" == pgid && "${child}" =~ ^[0-9]+$ && "${child}" -gt 1 ]] \
      && kill "-${signal}" -- "-${child}" 2>/dev/null || true
  done
}

registered_alive() {
  local state wrapper kind child
  for state in "${state_dir}"/*.state; do
    read -r wrapper kind child <"${state}" || continue
    [[ "${wrapper}" =~ ^[0-9]+$ ]] && kill -0 "${wrapper}" 2>/dev/null && return 0
    [[ "${kind}" == pgid && "${child}" =~ ^[0-9]+$ ]] \
      && kill -0 -- "-${child}" 2>/dev/null && return 0
  done
  return 1
}

signal_registered TERM
for _ in {1..50}; do
  registered_alive || break
  sleep 0.1
done
if registered_alive; then
  signal_registered KILL
  for _ in {1..20}; do
    registered_alive || break
    sleep 0.1
  done
fi

registered_alive && exit 1

rm -f "${state_dir}"/*.state
rmdir "${state_dir}" 2>/dev/null || true
