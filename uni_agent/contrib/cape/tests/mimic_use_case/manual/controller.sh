#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Install CAPE into a Python 3.12 virtual environment, prepare the authenticated
# manual-reproduction namespace, and run its controller in the foreground.
# Configure the run with environment variables; use --help for the supported
# surface.
set -euo pipefail
source "$(dirname "$0")/../identity.sh"

usage() {
  cat <<'EOF'
Usage: controller.sh

Install CAPE, prepare authentication, and start an authenticated CAPE
controller. The controller remains in the foreground until interrupted.

Common overrides:
  RUN_ID                  Fresh run id (default: manual-<UTC timestamp>)
  CASE                    Case name (default: manual)
  CONTROLLER_HOST         Routable hostname advertised to other nodes
  CAPE_PORT               Controller port (default: 18892)
  CAPE_POOL               Primary pool name (default: exclusive-pool)
  CAPE_ADDITIONAL_POOLS   Newline-separated additional pool names
                          (default: shared-pool; empty disables additional pools).
                          Pool names only label routing queues; a pool's
                          exclusive/shared mode is administrator-owned. Default shared-pool
                          is created shared; other pools default exclusive.
  CAPE_GPU_CONTENTION_POLICY
                          barrier-ratio|ratio-only|exclusive-first
                          (default: barrier-ratio)
  CAPE_GPU_SHARED_RATIO   Shared admissions per exclusive turn (default: 4)
  CAPE_GPU_SHARE_OVERRIDES
                          Newline-separated GPU_UUID=POLICY[:RATIO]
                          per-physical-GPU overrides
  CAPE_PROFILES_FILE      Optional controller-local JSON profile registry
  CAPE_MAX_ACTIVE_GPU_SESSIONS_PER_USER
  CAPE_MAX_QUEUED_REQUESTS_PER_USER
  CAPE_MAX_ACTIVE_SESSIONS_PER_POOL
                          Optional non-negative quota limits
  CAPE_IDLE_SESSION_TIMEOUT_SECONDS
                          Cached-session idle timeout (default: 3600)
  CAPE_RECONCILE_MISSING_GRACE_SECONDS
                          Restart missing-session grace (default: 5)
  CAPE_RECONCILE_MISSING_HEARTBEATS
                          Consecutive authoritative misses (default: 2)
  BUNDLE                  Evidence directory
  REPO_ROOT               Combined CAPE repository root
  CAPE_PROJECT            Python project directory (default: REPO_ROOT/pycape)
  CAPE_VENV_ROOT          Role-local venv root (default: ./venv)
  CAPE_VENV_HOST_ID       Host component of the default venv identity
  CAPE_VENV               Exact venv path
                          (default: VENV_ROOT/controller-<host>-<arch>-python3.12)
  CAPE_PYTHON_312         Optional explicit Python 3.12 executable
  CAPE_PIXI_BIN           Optional Pixi executable for rootless Python bootstrap
  CAPE_BIN                CAPE implementation selector (default: rscape).
                          Set pycape to use the role-local Python venv.
  CAPE_FORCE_INSTALL      Set to 1 to reinstall CAPE (default: 0)
  CAPE_MAX_NODES          Per-pool enrollment limit (default: 4)
  CAPE_STATE_DIR          Existing/new persistent controller directory
  CAPE_ADMIN_TOKEN_FILE   Private bootstrap credential file
  CAPE_SHARED_GPU_SLOTS   Slots for the default shared-pool (default: 4)
  CAPE_REPLACE_EXISTING   Set to 1 to replace this namespace's owned files

Only the bootstrap admin credential is created here, in CAPE_ADMIN_TOKEN_FILE.
Create regular credentials separately with user.sh. No handoff files are emitted.
EOF
}

die() {
  printf 'controller.sh: %s\n' "$*" >&2
  exit 2
}

python312_has_ensurepip_wheel() {
  "$@" -c '
import ensurepip
import pathlib
import sys
import venv

assert sys.version_info[:2] == (3, 12)
roots = [pathlib.Path(ensurepip.__file__).resolve().parent / "_bundled"]
wheel_dir = getattr(ensurepip, "_WHEEL_PKG_DIR", "")
if wheel_dir:
    roots.append(pathlib.Path(wheel_dir))
assert any(root.is_dir() and any(root.glob("pip-*.whl")) for root in roots)
' >/dev/null 2>&1
}

python312_bootstrap_mode() {
  if python312_has_ensurepip_wheel "$@"; then
    printf 'ensurepip\n'
  elif "$@" -m pip --help 2>/dev/null | grep -q -- '--python'; then
    printf 'external-pip\n'
  else
    return 1
  fi
}

python312_can_bootstrap_venv() {
  python312_bootstrap_mode "$@" >/dev/null
}

create_python312_venv() {
  local mode="$1"
  local target="$2"
  local clear_existing="$3"
  shift 3
  local -a venv_args=(-m venv --copies)
  [ "$clear_existing" != 1 ] || venv_args+=(--clear)
  [ "$mode" != external-pip ] || venv_args+=(--without-pip)
  venv_args+=("$target")
  "$@" "${venv_args[@]}" || return
  if [ "$mode" = external-pip ]; then
    "$@" -m pip --python "$target/bin/python" install \
      --disable-pip-version-check pip
  fi
}

if [ "$#" -gt 0 ]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unexpected argument: $1"
      ;;
  esac
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../project_root.sh"
REPO_ROOT="${REPO_ROOT:-$(cape_project_root "$SCRIPT_DIR")}"
CAPE_PROJECT="${CAPE_PROJECT:-$REPO_ROOT/pycape}"

RUN_ID="${RUN_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}"
CASE="${CASE:-manual}"
# Provision two pools by default; regular users register separately.
CAPE_POOL="${CAPE_POOL:-exclusive-pool}"
CAPE_ADDITIONAL_POOLS="${CAPE_ADDITIONAL_POOLS-shared-pool}"
CAPE_PORT="${CAPE_PORT:-18892}"
CAPE_BIND_HOST="${CAPE_BIND_HOST:-0.0.0.0}"
CAPE_SCHEDULE_INTERVAL="${CAPE_SCHEDULE_INTERVAL:-0.05}"
CAPE_GPU_CONTENTION_POLICY="${CAPE_GPU_CONTENTION_POLICY:-barrier-ratio}"
CAPE_GPU_SHARED_RATIO="${CAPE_GPU_SHARED_RATIO:-4}"
CAPE_GPU_SHARE_OVERRIDES="${CAPE_GPU_SHARE_OVERRIDES:-}"
CAPE_PROFILES_FILE="${CAPE_PROFILES_FILE:-}"
CAPE_MAX_ACTIVE_GPU_SESSIONS_PER_USER="${CAPE_MAX_ACTIVE_GPU_SESSIONS_PER_USER:-}"
CAPE_MAX_QUEUED_REQUESTS_PER_USER="${CAPE_MAX_QUEUED_REQUESTS_PER_USER:-}"
CAPE_MAX_ACTIVE_SESSIONS_PER_POOL="${CAPE_MAX_ACTIVE_SESSIONS_PER_POOL:-}"
CAPE_IDLE_SESSION_TIMEOUT_SECONDS="${CAPE_IDLE_SESSION_TIMEOUT_SECONDS:-3600}"
CAPE_RECONCILE_MISSING_GRACE_SECONDS="${CAPE_RECONCILE_MISSING_GRACE_SECONDS:-5}"
CAPE_RECONCILE_MISSING_HEARTBEATS="${CAPE_RECONCILE_MISSING_HEARTBEATS:-2}"
CAPE_MAX_NODES="${CAPE_MAX_NODES:-4}"
if [ -n "${CAPE_JOIN_ALLOWED_NODE_PREFIXES:-}${CAPE_JOIN_ALLOWED_RUNTIME_CONFIGS:-}" ]; then
  die "Join Token restrictions are removed; configure pool membership limits with admin pool"
fi
CAPE_REPLACE_EXISTING="${CAPE_REPLACE_EXISTING:-0}"
CAPE_VENV_ROOT="${CAPE_VENV_ROOT:-$REPO_ROOT/venv}"
CAPE_VENV_HOST_ID="${CAPE_VENV_HOST_ID:-$(hostname -s 2>/dev/null || hostname)}"
CAPE_VENV_HOST_ID="${CAPE_VENV_HOST_ID//[^A-Za-z0-9_.-]/_}"
CAPE_VENV="${CAPE_VENV:-${CAPE_VENV_ROOT}/controller-${CAPE_VENV_HOST_ID}-$(uname -m)-python3.12}"
CAPE_PYTHON_312="${CAPE_PYTHON_312:-}"
CAPE_PIXI_BIN="${CAPE_PIXI_BIN:-}"
CAPE_FORCE_INSTALL="${CAPE_FORCE_INSTALL:-0}"
case "$CAPE_VENV" in
  /*) ;;
  *) CAPE_VENV="$PWD/${CAPE_VENV#./}" ;;
esac
CAPE_BIN_SELECTION="${CAPE_BIN:-rscape}"
if [ "$CAPE_BIN_SELECTION" = pycape ]; then
  CAPE_BIN_OVERRIDE=""
  CAPE_BIN="$CAPE_VENV/bin/pycape"
else
  CAPE_BIN_OVERRIDE="$CAPE_BIN_SELECTION"
  CAPE_BIN="$CAPE_BIN_SELECTION"
fi

if [ -z "${CONTROLLER_HOST:-}" ]; then
  CONTROLLER_HOST="$(hostname -f 2>/dev/null || hostname)"
fi
CAPE_URL="${CAPE_URL:-http://$CONTROLLER_HOST:$CAPE_PORT}"
CAPE_CONTROLLER_URL="$CAPE_URL"

BUNDLE="${BUNDLE:-$REPO_ROOT/evidence/cape-use-case/$CASE-$RUN_ID}"
CAPE_STATE_DIR="${CAPE_STATE_DIR:-$BUNDLE/pool/controller-state}"
CAPE_ADMIN_TOKEN_FILE="${CAPE_ADMIN_TOKEN_FILE:-$BUNDLE/pool/secrets/admin.json}"
CAPE_CONTROLLER_LOG="${CAPE_CONTROLLER_LOG:-$BUNDLE/pool/controller/controller.log}"


[ -n "$CAPE_POOL" ] || die "CAPE_POOL must not be empty"
pool_names=("$CAPE_POOL")
while IFS= read -r additional_pool; do
  additional_pool="${additional_pool%$'\r'}"
  [ -n "$additional_pool" ] || continue
  for existing_pool in "${pool_names[@]}"; do
    [ "$additional_pool" != "$existing_pool" ] || \
      die "CAPE_ADDITIONAL_POOLS contains duplicate pool: $additional_pool"
  done
  pool_names+=("$additional_pool")
done <<< "$CAPE_ADDITIONAL_POOLS"

case "$CAPE_PORT" in
  ''|*[!0-9]*) die "CAPE_PORT must be an integer" ;;
esac
[ "$CAPE_PORT" -ge 1 ] && [ "$CAPE_PORT" -le 65535 ] || \
  die "CAPE_PORT must be between 1 and 65535"

case "$CAPE_MAX_NODES" in
  ''|*[!0-9]*) die "CAPE_MAX_NODES must be a positive integer" ;;
esac
[ "$CAPE_MAX_NODES" -ge 1 ] || die "CAPE_MAX_NODES must be at least 1"

[[ "$CAPE_SCHEDULE_INTERVAL" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
  die "CAPE_SCHEDULE_INTERVAL must be numeric"
case "$CAPE_GPU_CONTENTION_POLICY" in
  barrier-ratio|ratio-only|exclusive-first) ;;
  *) die "CAPE_GPU_CONTENTION_POLICY must be barrier-ratio, ratio-only, or exclusive-first" ;;
esac
case "$CAPE_GPU_SHARED_RATIO" in
  ''|*[!0-9]*) die "CAPE_GPU_SHARED_RATIO must be a positive integer" ;;
esac
[ "$CAPE_GPU_SHARED_RATIO" -ge 1 ] || \
  die "CAPE_GPU_SHARED_RATIO must be at least 1"
for quota_name in \
  CAPE_MAX_ACTIVE_GPU_SESSIONS_PER_USER \
  CAPE_MAX_QUEUED_REQUESTS_PER_USER \
  CAPE_MAX_ACTIVE_SESSIONS_PER_POOL; do
  quota_value="${!quota_name}"
  case "$quota_value" in
    ''|*[!0-9]*) [ -z "$quota_value" ] || die "$quota_name must be a non-negative integer" ;;
  esac
done
for timeout_name in CAPE_IDLE_SESSION_TIMEOUT_SECONDS CAPE_RECONCILE_MISSING_GRACE_SECONDS; do
  timeout_value="${!timeout_name}"
  [[ "$timeout_value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
    die "$timeout_name must be a non-negative number"
done
case "$CAPE_RECONCILE_MISSING_HEARTBEATS" in
  ''|*[!0-9]*) die "CAPE_RECONCILE_MISSING_HEARTBEATS must be a positive integer" ;;
esac
[ "$CAPE_RECONCILE_MISSING_HEARTBEATS" -ge 1 ] || \
  die "CAPE_RECONCILE_MISSING_HEARTBEATS must be at least 1"
[ -z "$CAPE_PROFILES_FILE" ] || [ -r "$CAPE_PROFILES_FILE" ] || \
  die "CAPE_PROFILES_FILE is not readable: $CAPE_PROFILES_FILE"
gpu_share_override_args=()
while IFS= read -r gpu_share_override; do
  gpu_share_override="${gpu_share_override%$'\r'}"
  [ -n "$gpu_share_override" ] || continue
  override_uuid="${gpu_share_override%%=*}"
  override_policy_ratio="${gpu_share_override#*=}"
  [ "$override_uuid" != "$gpu_share_override" ] && \
    [ -n "$override_uuid" ] && [ -n "$override_policy_ratio" ] || \
    die "CAPE_GPU_SHARE_OVERRIDES entries must be GPU_UUID=POLICY[:RATIO]"
  case "$override_uuid" in
    *[[:space:]]*) die "GPU override UUID must not contain whitespace" ;;
  esac
  override_policy="${override_policy_ratio%%:*}"
  case "$override_policy" in
    barrier-ratio|ratio-only|exclusive-first) ;;
    *) die "GPU override policy must be barrier-ratio, ratio-only, or exclusive-first" ;;
  esac
  if [[ "$override_policy_ratio" == *:* ]]; then
    override_ratio="${override_policy_ratio#*:}"
    case "$override_ratio" in
      ''|*[!0-9]*) die "GPU override ratio must be a positive integer" ;;
    esac
    [ "$override_ratio" -ge 1 ] || \
      die "GPU override ratio must be at least 1"
  fi
  gpu_share_override_args+=(--gpu-share-override "$gpu_share_override")
done <<< "$CAPE_GPU_SHARE_OVERRIDES"
case "$CAPE_REPLACE_EXISTING" in
  0|1) ;;
  *) die "CAPE_REPLACE_EXISTING must be 0 or 1" ;;
esac
case "$CAPE_FORCE_INSTALL" in
  0|1) ;;
  *) die "CAPE_FORCE_INSTALL must be 0 or 1" ;;
esac
case "$CAPE_URL" in
  http://*) ;;
  *) die "CAPE_URL must use http:// because the CAPE controller does not terminate TLS" ;;
esac

if [ -n "$CAPE_BIN_OVERRIDE" ]; then
  printf '[controller] using CAPE executable selection: %s\n' "$CAPE_BIN_SELECTION"
elif [ -e "$CAPE_VENV" ] && [ ! -d "$CAPE_VENV" ]; then
  die "CAPE_VENV exists but is not a directory: $CAPE_VENV"
fi

if [ -z "$CAPE_BIN_OVERRIDE" ]; then
mkdir -p "$CAPE_VENV_ROOT"
install_lock="$CAPE_VENV_ROOT/.install-controller-${CAPE_VENV_HOST_ID}-$(uname -m)-python3.12.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$install_lock"
  flock 9
fi

VENV_PYTHON="$CAPE_VENV/bin/python"
venv_ready=0
if [ -x "$VENV_PYTHON" ]; then
  venv_version="$($VENV_PYTHON -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
  if [ "$venv_version" = 3.12 ] && \
     "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
    venv_ready=1
  fi
fi

if [ "$venv_ready" != 1 ]; then
  python312_cmd=()
  python312_description=""

  if [ -n "$CAPE_PYTHON_312" ]; then
    if [[ "$CAPE_PYTHON_312" == */* ]]; then
      [ -x "$CAPE_PYTHON_312" ] || \
        die "CAPE_PYTHON_312 is not executable: $CAPE_PYTHON_312"
    else
      command -v "$CAPE_PYTHON_312" >/dev/null 2>&1 || \
        die "CAPE_PYTHON_312 was not found: $CAPE_PYTHON_312"
    fi
    python312_cmd=("$CAPE_PYTHON_312")
    python312_description="$CAPE_PYTHON_312"
    python312_can_bootstrap_venv "${python312_cmd[@]}" || \
      die "$CAPE_PYTHON_312 cannot bootstrap a pip-enabled venv"
  elif command -v python3.12 >/dev/null 2>&1 && \
       python312_can_bootstrap_venv python3.12; then
    python312_cmd=(python3.12)
    python312_description="$(command -v python3.12)"
  else
    if [ -z "$CAPE_PIXI_BIN" ]; then
      if command -v pixi >/dev/null 2>&1; then
        CAPE_PIXI_BIN="$(command -v pixi)"
      fi
    fi
    if [ -n "$CAPE_PIXI_BIN" ] && [[ "$CAPE_PIXI_BIN" != */* ]]; then
      resolved_pixi="$(command -v "$CAPE_PIXI_BIN" 2>/dev/null || true)"
      [ -z "$resolved_pixi" ] || CAPE_PIXI_BIN="$resolved_pixi"
    fi
    [ -n "$CAPE_PIXI_BIN" ] && [ -x "$CAPE_PIXI_BIN" ] || \
      die "a pip-enabled Python 3.12 bootstrap and Pixi were not found; set CAPE_PYTHON_312 or CAPE_PIXI_BIN"
    python312_cmd=("$CAPE_PIXI_BIN" exec --no-progress --spec "python=3.12" python)
    python312_description="Pixi ($CAPE_PIXI_BIN)"
    printf '[controller] usable Python 3.12 venv bootstrap not found; bootstrapping it with %s\n' "$python312_description"
  fi

  bootstrap_version="$("${python312_cmd[@]}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" || \
    die "failed to start Python through $python312_description"
  [ "$bootstrap_version" = 3.12 ] || \
    die "Python bootstrap must provide 3.12, got $bootstrap_version from $python312_description"
  python312_bootstrap_mode="$(python312_bootstrap_mode "${python312_cmd[@]}")" || \
    die "$python312_description cannot bootstrap a pip-enabled venv"

  if [ -d "$CAPE_VENV" ] && [ ! -f "$CAPE_VENV/pyvenv.cfg" ]; then
    die "CAPE_VENV exists but is not a Python virtual environment: $CAPE_VENV"
  fi

  mkdir -p "$(dirname "$CAPE_VENV")"
  if [ -f "$CAPE_VENV/pyvenv.cfg" ]; then
    printf '[controller] rebuilding incompatible Python 3.12 virtual environment: %s\n' "$CAPE_VENV"
    create_python312_venv "$python312_bootstrap_mode" "$CAPE_VENV" 1 \
      "${python312_cmd[@]}" || \
      die "failed to rebuild the Python 3.12 virtual environment"
  else
    printf '[controller] creating Python 3.12 virtual environment: %s\n' "$CAPE_VENV"
    create_python312_venv "$python312_bootstrap_mode" "$CAPE_VENV" 0 \
      "${python312_cmd[@]}" || \
      die "failed to create the Python 3.12 virtual environment"
  fi
fi

[ -x "$VENV_PYTHON" ] || die "virtual-environment Python is missing: $VENV_PYTHON"
venv_version="$($VENV_PYTHON -c 'import sys; print("%d.%d" % sys.version_info[:2])')" || \
  die "failed to query virtual-environment Python"
[ "$venv_version" = 3.12 ] || \
  die "CAPE_VENV must use Python 3.12, got $venv_version"

[ -f "$CAPE_PROJECT/pyproject.toml" ] || \
  die "CAPE project metadata was not found: $CAPE_PROJECT/pyproject.toml"

install_marker="$CAPE_VENV/.cape-install-fingerprint"
install_fingerprint="$($VENV_PYTHON -c '
import hashlib, os, sys
project, metadata = sys.argv[1:]
h = hashlib.sha256()
h.update(os.path.realpath(project).encode())
with open(metadata, "rb") as stream:
    h.update(stream.read())
print(h.hexdigest())
' "$CAPE_PROJECT" "$CAPE_PROJECT/pyproject.toml")"

install_ready=0
if [ "$CAPE_FORCE_INSTALL" = 0 ] && [ -x "$CAPE_BIN" ] && [ -r "$install_marker" ]; then
  recorded_fingerprint="$(sed -n '1p' "$install_marker")"
  if [ "$recorded_fingerprint" = "$install_fingerprint" ] && \
     "$VENV_PYTHON" -c 'import aiohttp, pycape' >/dev/null 2>&1; then
    install_ready=1
  fi
fi

if [ "$install_ready" = 1 ]; then
  printf '[controller] reusing CAPE installation in %s\n' "$CAPE_VENV"
else
  printf '[controller] installing CAPE from %s into %s\n' "$CAPE_PROJECT" "$CAPE_VENV"
  "$VENV_PYTHON" -m pip install \
    --disable-pip-version-check \
    --editable "$CAPE_PROJECT" || die "CAPE installation failed"

  "$VENV_PYTHON" -c \
    'import sys, aiohttp, pycape; assert sys.version_info[:2] == (3, 12)' || \
    die "installed CAPE or aiohttp could not be imported with Python 3.12"
  marker_tmp="$install_marker.tmp.$$"
  printf '%s\n' "$install_fingerprint" > "$marker_tmp"
  mv -- "$marker_tmp" "$install_marker"
fi

if command -v flock >/dev/null 2>&1; then
  flock -u 9
  exec 9>&-
fi
fi

CAPE_BIN="$(cape_resolve_executable "$CAPE_BIN" CAPE_BIN)" || exit 2
cape_inspect_executable "$CAPE_BIN_SELECTION" "$CAPE_BIN" || exit 2
command -v tee >/dev/null 2>&1 || die "tee is required"

owned_files=(
  "$CAPE_CONTROLLER_LOG"
)
existing_files=()
for path in "${owned_files[@]}"; do
  [ ! -e "$path" ] || existing_files+=("$path")
done

if [ "${#existing_files[@]}" -gt 0 ]; then
  if [ "$CAPE_REPLACE_EXISTING" != 1 ]; then
    printf 'controller.sh: refusing to reuse an existing case namespace:\n' >&2
    printf '  %s\n' "${existing_files[@]}" >&2
    die "choose a fresh RUN_ID/BUNDLE or set CAPE_REPLACE_EXISTING=1"
  fi
  rm -f -- "${owned_files[@]}"
fi

umask 077
mkdir -p \
  "$BUNDLE/pool/controller" \
  "$BUNDLE/users" \
  "$(dirname "$CAPE_ADMIN_TOKEN_FILE")" \
  "$(dirname "$CAPE_STATE_DIR")" \
  "$(dirname "$CAPE_CONTROLLER_LOG")"
cape_record_executable controller "$CAPE_BIN" \
  "$BUNDLE/pool/controller/implementation.txt"

chmod 700 "$(dirname "$CAPE_ADMIN_TOKEN_FILE")"
cape_init_instance "$CAPE_STATE_DIR" "$CAPE_ADMIN_TOKEN_FILE"
controller_args=(
  controller start
  --host "$CAPE_BIND_HOST"
  --port "$CAPE_PORT"
  --schedule-interval "$CAPE_SCHEDULE_INTERVAL"
  --state-dir "$CAPE_STATE_DIR"
  --token-file "$CAPE_ADMIN_TOKEN_FILE"
  --gpu-contention-policy "$CAPE_GPU_CONTENTION_POLICY"
  --gpu-shared-ratio "$CAPE_GPU_SHARED_RATIO"
  --idle-session-timeout-seconds "$CAPE_IDLE_SESSION_TIMEOUT_SECONDS"
  --reconcile-missing-grace-seconds "$CAPE_RECONCILE_MISSING_GRACE_SECONDS"
  --reconcile-missing-heartbeats "$CAPE_RECONCILE_MISSING_HEARTBEATS"
)
controller_args+=("${gpu_share_override_args[@]}")
[ -z "$CAPE_PROFILES_FILE" ] || controller_args+=(--profiles-file "$CAPE_PROFILES_FILE")
[ -z "$CAPE_MAX_ACTIVE_GPU_SESSIONS_PER_USER" ] || \
  controller_args+=(--max-active-gpu-sessions-per-user "$CAPE_MAX_ACTIVE_GPU_SESSIONS_PER_USER")
[ -z "$CAPE_MAX_QUEUED_REQUESTS_PER_USER" ] || \
  controller_args+=(--max-queued-requests-per-user "$CAPE_MAX_QUEUED_REQUESTS_PER_USER")
[ -z "$CAPE_MAX_ACTIVE_SESSIONS_PER_POOL" ] || \
  controller_args+=(--max-active-sessions-per-pool "$CAPE_MAX_ACTIVE_SESSIONS_PER_POOL")
controller_pid=""
cleanup_controller() {
  local code=$?
  trap - EXIT INT TERM
  if [ -n "$controller_pid" ]; then
    kill "$controller_pid" 2>/dev/null || true
    wait "$controller_pid" 2>/dev/null || true
  fi
  exit "$code"
}
trap cleanup_controller EXIT INT TERM
"$CAPE_BIN" "${controller_args[@]}" > >(tee "$CAPE_CONTROLLER_LOG") 2>&1 &
controller_pid=$!
ready=0
for _ in $(seq 1 100); do
  kill -0 "$controller_pid" 2>/dev/null || die "controller exited before bootstrap completed"
  if curl -fsS "http://127.0.0.1:$CAPE_PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 0.1
done
[ "$ready" = 1 ] || die "controller health timeout"

for ((pool_index = 0; pool_index < ${#pool_names[@]}; pool_index++)); do
  pool_name="${pool_names[$pool_index]}"
  mode=exclusive; slots=1
  if [ "$pool_name" = shared-pool ]; then mode=shared; slots="${CAPE_SHARED_GPU_SLOTS:-4}"; fi
  if [ "$pool_index" = 0 ]; then mode="${CAPE_POOL_MODE:-$mode}"; slots="${CAPE_GPU_SLOTS:-$slots}"; fi
  cape_ensure_pool "$pool_name" "$CAPE_ADMIN_TOKEN_FILE" "$mode" "$slots"
  "$CAPE_BIN" admin pool edit "$pool_name" --max-nodes "$CAPE_MAX_NODES" \
    --controller-url "$CAPE_URL" --token-file "$CAPE_ADMIN_TOKEN_FILE" >/dev/null
  printf '[controller] pool=%s ready\n' "$pool_name"
done
printf '[controller] url=%s state=%s\n' "$CAPE_URL" "$CAPE_STATE_DIR"
printf '[controller] admin credential file=%s (secret not printed)\n' "$CAPE_ADMIN_TOKEN_FILE"
cape_publish_role_pid
unset CAPE_ROLE_PID_FILE CAPE_ROLE_LABEL
wait "$controller_pid"
controller_pid=""
