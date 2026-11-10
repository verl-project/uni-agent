#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Join one GPU node to a controller prepared by manual/controller.sh.
# The node agent stays in the foreground until interrupted or authoritatively
# removed; backgrounding is the caller's choice.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/credentials.sh"
cape_manual_credentials "$@"
set -- "${CAPE_MANUAL_ARGS[@]}"

usage() {
  cat <<'EOF'
Credential input: --token-file FILE (private personal credential JSON).
Use an explicit controller URL and pool; controller.sh no longer generates handoffs.

Usage:
  register_node.sh /path/to/controller.env
  CAPE_CONTROLLER_ENV=/path/to/controller.env register_node.sh
  source /path/to/controller.env && register_node.sh

The controller environment supplies only the URL, pool, run identity, and
authentication tokens. Node-local paths and executables are configured with:

  CAPE_NODE_REPO_ROOT       Node host's combined CAPE repository root
  CAPE_NODE_PROJECT         Node host's Python project directory
                            (default: NODE_REPO_ROOT/pycape)
  CAPE_NODE_BUNDLE          Node evidence directory (default: manual/evidence beside this script)
  CAPE_NODE_ID             Unique node id
                           (default: <pool>-<short-hostname>)
  CAPE_POOL_MODE           exclusive|shared (default: exclusive)
  CAPE_GPU_SLOTS           Slots per GPU. Required explicitly for shared
                           pools; exclusive defaults to and requires 1.
  CAPE_POOL_PLACEMENT      pack|spread shared-slot placement
                           (default: pack)
  CAPE_ALLOCATION_ID       Explicit allocation id override. By default a fresh
                           CASE-RUN_ID-NODE_ID-TIMESTAMP-PID id is generated
                           once per invocation. ALLOCATION_ID is the lower-
                           precedence compatibility override.
  CAPE_GPU_UUIDS           Explicit comma/space-separated GPU UUIDs
  CAPE_GPU_COUNT           Otherwise select the first N visible GPUs
  CAPE_NODE_CPU_CORES      Registered CPUs (default: nproc)
  CAPE_NODE_MEMORY_GB      Registered memory GiB (default: /proc/meminfo)
  CAPE_RUNTIME_ADAPTERS    Optional comma/space-separated runtime probe filter.
                           Default: auto-probe apptainer, bubblewrap and podman.
                           local-process requires explicit selection.
  CAPE_APPTAINER_BIN       Node-local Apptainer executable
  CAPE_BUBBLEWRAP_BIN      Node-local Bubblewrap executable
  CAPE_BUBBLEWRAP_SYSTEM_PATHS
                           Optional comma/newline-separated node-local
                           system/toolchain directories exposed by Bubblewrap
                           host policies (default: /usr, /etc, /opt). Host
                           root and /proc, /sys, /dev, /run are forbidden.
  CAPE_BUBBLEWRAP_READ_ONLY_PATHS
                           Optional comma/newline-separated virtual-kernel
                           subpaths, always read-only; default empty.
  CAPE_PODMAN_BIN          Node-local Podman executable
  CAPE_APPTAINER_PROBE_IMAGE
                           SIF/OCI image used to prove Apptainer policies
  CAPE_PODMAN_PROBE_IMAGE  OCI image used to prove Podman policies
  CAPE_RUNTIME_ROOT        Node-local runtime root (default: /tmp/cape-RUN-NODE)
  CAPE_EXECUTION_STATE_DIR Retained layer root (default: CAPE_RUNTIME_ROOT/retained); use a durable volume for reboot retention.
  CAPE_NODE_RUNTIME_CACHE_CONFIG
                            Join-token/cache-locality identity (default: runtime cache path)
  CAPE_NODE_CACHED_IMAGES  Newline-separated image identities already cached on this node
  CAPE_NODE_LOG_REDACT_ENV_KEYS
                            Comma/space-separated extra environment keys to mask in runtime logs
  CAPE_NODE_DRAIN_BEFORE_SEC
                            Drain this many seconds before the allocation deadline
  CAPE_NODE_CLEANUP_BUFFER_SEC
                            Reserve this many deadline seconds for runtime cleanup
  CAPE_ALLOCATION_DEADLINE Absolute Unix allocation deadline, if bounded
  CAPE_NO_WARM_SESSIONS    Set to 1 to disable stateful writable-layer capability
  CAPE_STAY_AFTER_REMOVAL  Set to 1 to keep heartbeating after authoritative
                           removal instead of cleaning sessions and exiting 0
  CAPE_NODE_VENV_ROOT      Role-local venv root (default: NODE_REPO_ROOT/venv)
  CAPE_NODE_VENV_HOST_ID   Host component of the default venv identity
  CAPE_NODE_VENV           Exact node venv path
  CAPE_NODE_TOKEN          Contributor personal credential (normally in the handoff)
  CAPE_NODE_RESUME         Set to 1 with the same node/state directory to resume a non-retired enrollment
  CAPE_NODE_BIN            Implementation selector (default: rscape).
                           Set pycape to use the role-local Python venv.
  CAPE_NODE_PYTHON_312     Optional explicit Python 3.12 executable
  CAPE_NODE_PIXI_BIN       Optional Pixi executable for Python bootstrap
  CAPE_FORCE_INSTALL       Set to 1 to reinstall CAPE (default: 0)

If neither CAPE_GPU_UUIDS nor CAPE_GPU_COUNT is set, every GPU reported by
nvidia-smi is registered. Do that only when the allocation owns all of them.
By default, after authoritative controller removal this foreground process
cleans its local runtime sessions and exits 0. CAPE_STAY_AFTER_REMOVAL=1 is the
explicit opt-out for diagnostics or externally managed legacy behavior.

Pool mode/slots/placement belongs to the administrator. Enrollment reads that
policy from the scheduler; it never creates or reconfigures a pool. Once the pool
is reconfigurable-empty (no members, non-terminal requests or live/cached sessions),
an administrator can change it with `pycape admin pool edit` or `rscape admin pool edit`.
EOF
}

die() {
  printf 'register_node.sh: %s\n' "$*" >&2
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

log() {
  printf '[node %s] %s\n' "${CAPE_NODE_ID:-unconfigured}" "$*"
}

load_controller_environment() {
  local path="$1"
  local assignments
  local local_name
  local -a portable_names=(
    CAPE_CONTROLLER_ENV_VERSION RUN_ID CASE CAPE_POOL CAPE_PORT
    CONTROLLER_HOST CAPE_URL CAPE_CONTROLLER_URL CAPE_ADMIN_TOKEN
    CAPE_NODE_TOKEN CAPE_USER_TOKENS_INLINE CAPE_POOL_MODE CAPE_GPU_SLOTS
  )
  assignments="$({
    set -euo pipefail
    for local_name in "${portable_names[@]}"; do
      case "$local_name" in CAPE_POOL_MODE|CAPE_GPU_SLOTS) continue;; esac
      unset "$local_name"
    done
    # shellcheck disable=SC1090
    source "$path" >/dev/null
    for local_name in "${portable_names[@]}"; do
      if [[ -v "$local_name" ]]; then
        printf 'export %s=%q\n' "$local_name" "${!local_name}"
      fi
    done
  })" || die "failed to load controller environment: $path"
  for local_name in "${portable_names[@]}"; do unset "$local_name"; done
  eval "$assignments"
  case "${CAPE_CONTROLLER_ENV_VERSION:-missing}" in
    3) ;;
    *) die "unsupported CAPE_CONTROLLER_ENV_VERSION: ${CAPE_CONTROLLER_ENV_VERSION:-missing}; regenerate the handoff with current controller.sh" ;;
  esac
}

controller_env="${CAPE_CONTROLLER_ENV:-}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --controller-url) CAPE_URL=${2:?}; shift 2 ;;
    --pool) CAPE_POOL=${2:?}; shift 2 ;;
    *)
      [ "$#" -eq 1 ] || die "only one controller.env path is accepted"
      controller_env="$1"
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "$controller_env" ]; then
  [ -r "$controller_env" ] || die "controller environment is not readable: $controller_env"
  # Evaluate only a trusted handoff, then import its portable allowlist. Paths,
  # executables, runtime settings, and role-local overrides remain node-local.
  load_controller_environment "$controller_env"
fi

source "$SCRIPT_DIR/../project_root.sh"
SCRIPT_REPO_ROOT="$(cape_project_root "$SCRIPT_DIR")"
REPO_ROOT="${CAPE_NODE_REPO_ROOT:-$SCRIPT_REPO_ROOT}"
CAPE_PROJECT="${CAPE_NODE_PROJECT:-$REPO_ROOT/pycape}"
CAPE_URL="${CAPE_URL:-${CAPE_CONTROLLER_URL:-}}"
CAPE_POOL="${CAPE_POOL:-exclusive-pool}"
CAPE_POOL_MODE="${CAPE_POOL_MODE:-exclusive}"
CAPE_GPU_SLOTS_EXPLICIT=0
if [[ -v CAPE_GPU_SLOTS ]] && [ -n "$CAPE_GPU_SLOTS" ]; then
  CAPE_GPU_SLOTS_EXPLICIT=1
fi
CAPE_GPU_SLOTS="${CAPE_GPU_SLOTS:-1}"
CAPE_POOL_PLACEMENT="${CAPE_POOL_PLACEMENT:-pack}"
CASE="${CASE:-manual}"

[ -n "$CAPE_URL" ] || die "CAPE_URL is required"
RUN_ID="${RUN_ID:-manual}"
CAPE_NODE_TOKEN="${CAPE_NODE_TOKEN:-${CAPE_TOKEN:-}}"
[ -n "$CAPE_NODE_TOKEN" ] || die "CAPE_NODE_TOKEN (contributor personal credential) is required"

for variable_name in REPO_ROOT CAPE_PROJECT; do
  value="${!variable_name}"
  case "$value" in
    /*) ;;
    *) printf -v "$variable_name" '%s/%s' "$PWD" "${value#./}" ;;
  esac
done
CAPE_NODE_BUNDLE="${CAPE_NODE_BUNDLE:-$SCRIPT_DIR/evidence}"
case "$CAPE_NODE_BUNDLE" in
  /*) ;;
  *) CAPE_NODE_BUNDLE="$PWD/${CAPE_NODE_BUNDLE#./}" ;;
esac
CAPE_NODE_BIN_SELECTION="${CAPE_NODE_BIN:-rscape}"
if [ "$CAPE_NODE_BIN_SELECTION" = pycape ]; then
  [ -f "$CAPE_PROJECT/pyproject.toml" ] || \
    die "CAPE project metadata was not found: $CAPE_PROJECT/pyproject.toml"
fi

if [ -z "${CAPE_NODE_ID:-${NODE_ID:-}}" ]; then
  node_prefix="$CAPE_POOL-$(hostname -s 2>/dev/null || hostname)"
  CAPE_NODE_ID="${node_prefix:0:31}-$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
else
  CAPE_NODE_ID="${CAPE_NODE_ID:-$NODE_ID}"
fi
if [ -n "${CAPE_ALLOCATION_ID:-}" ]; then
  : # The CAPE-specific operator override wins verbatim.
elif [ -n "${ALLOCATION_ID:-}" ]; then
  CAPE_ALLOCATION_ID="$ALLOCATION_ID"
else
  allocation_prefix="$CASE-$RUN_ID-$CAPE_NODE_ID"
  CAPE_ALLOCATION_ID="${allocation_prefix:0:31}-$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
fi
[[ $CAPE_NODE_ID =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]] || die "CAPE_NODE_ID must contain 1–64 ASCII name characters"
[[ $CAPE_ALLOCATION_ID =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$ ]] || die "CAPE_ALLOCATION_ID must contain 1–64 ASCII name characters"
CAPE_NODE_CPU_CORES="${CAPE_NODE_CPU_CORES:-$(nproc 2>/dev/null || true)}"
CAPE_NODE_MEMORY_GB="${CAPE_NODE_MEMORY_GB:-$(awk '/MemTotal/ {printf "%d", $2/1024/1024; found=1} END {if (!found) exit 1}' /proc/meminfo 2>/dev/null || true)}"
if [ -n "${CAPE_RUNTIME_ADAPTER:-}" ]; then
  die "CAPE_RUNTIME_ADAPTER was removed; use optional CAPE_RUNTIME_ADAPTERS as a probe filter"
fi
CAPE_RUNTIME_ADAPTERS="${CAPE_RUNTIME_ADAPTERS:-}"
CAPE_BUBBLEWRAP_SYSTEM_PATHS="${CAPE_BUBBLEWRAP_SYSTEM_PATHS:-}"
CAPE_BUBBLEWRAP_READ_ONLY_PATHS="${CAPE_BUBBLEWRAP_READ_ONLY_PATHS:-}"
CAPE_APPTAINER_PROBE_IMAGE="${CAPE_APPTAINER_PROBE_IMAGE:-${CAPE_RUNTIME_PROBE_IMAGE:-docker://docker.io/library/ubuntu:24.04}}"
CAPE_PODMAN_PROBE_IMAGE="${CAPE_PODMAN_PROBE_IMAGE:-docker://docker.io/library/ubuntu:24.04}"
CAPE_HEARTBEAT_INTERVAL="${CAPE_HEARTBEAT_INTERVAL:-1}"
CAPE_NODE_DRAIN_BEFORE_SEC="${CAPE_NODE_DRAIN_BEFORE_SEC:-0}"
CAPE_NODE_CLEANUP_BUFFER_SEC="${CAPE_NODE_CLEANUP_BUFFER_SEC:-0}"
CAPE_NODE_CACHED_IMAGES="${CAPE_NODE_CACHED_IMAGES:-}"
CAPE_NODE_LOG_REDACT_ENV_KEYS="${CAPE_NODE_LOG_REDACT_ENV_KEYS:-}"
CAPE_NO_WARM_SESSIONS="${CAPE_NO_WARM_SESSIONS:-0}"
CAPE_STAY_AFTER_REMOVAL="${CAPE_STAY_AFTER_REMOVAL:-0}"
CAPE_FORCE_INSTALL="${CAPE_FORCE_INSTALL:-0}"
CAPE_NODE_VENV_ROOT="${CAPE_NODE_VENV_ROOT:-$REPO_ROOT/venv}"
CAPE_NODE_VENV_HOST_ID="${CAPE_NODE_VENV_HOST_ID:-$(hostname -s 2>/dev/null || hostname)}"
CAPE_NODE_VENV_HOST_ID="${CAPE_NODE_VENV_HOST_ID//[^A-Za-z0-9_.-]/_}"
CAPE_NODE_VENV="${CAPE_NODE_VENV:-${CAPE_NODE_VENV_ROOT}/node-${CAPE_NODE_VENV_HOST_ID}-$(uname -m)-python3.12}"
CAPE_NODE_PYTHON_312="${CAPE_NODE_PYTHON_312:-}"
CAPE_NODE_PIXI_BIN="${CAPE_NODE_PIXI_BIN:-}"
if [ "$CAPE_NODE_BIN_SELECTION" = pycape ]; then
  CAPE_NODE_BIN_OVERRIDE=""
else
  CAPE_NODE_BIN_OVERRIDE="$CAPE_NODE_BIN_SELECTION"
fi
CAPE_NVIDIA_SMI="${CAPE_NVIDIA_SMI:-nvidia-smi}"
CAPE_RUNTIME_ROOT="${CAPE_RUNTIME_ROOT:-/tmp/cape-$RUN_ID-$CAPE_NODE_ID}"
CAPE_EXECUTION_STATE_DIR="${CAPE_EXECUTION_STATE_DIR:-$CAPE_RUNTIME_ROOT/retained}"
CAPE_RUNTIME_CACHE="${CAPE_RUNTIME_CACHE:-$CAPE_RUNTIME_ROOT/cache}"
CAPE_RUNTIME_TMP="${CAPE_RUNTIME_TMP:-$CAPE_RUNTIME_ROOT/tmp}"
CAPE_RUNTIME_DIR="${CAPE_RUNTIME_DIR:-$CAPE_RUNTIME_ROOT/runtime}"
CAPE_APPTAINER_CACHE="${CAPE_APPTAINER_CACHE:-$CAPE_RUNTIME_ROOT/apptainer/cache}"
CAPE_APPTAINER_TMP="${CAPE_APPTAINER_TMP:-$CAPE_RUNTIME_ROOT/apptainer/tmp}"
CAPE_APPTAINER_RUNTIME_DIR="${CAPE_APPTAINER_RUNTIME_DIR:-$CAPE_RUNTIME_ROOT/apptainer/runtime}"
CAPE_BUBBLEWRAP_CACHE="${CAPE_BUBBLEWRAP_CACHE:-$CAPE_RUNTIME_ROOT/bubblewrap/cache}"
CAPE_BUBBLEWRAP_TMP="${CAPE_BUBBLEWRAP_TMP:-$CAPE_RUNTIME_ROOT/bubblewrap/tmp}"
CAPE_BUBBLEWRAP_RUNTIME_DIR="${CAPE_BUBBLEWRAP_RUNTIME_DIR:-$CAPE_RUNTIME_ROOT/bubblewrap/runtime}"
CAPE_PODMAN_GRAPHROOT="${CAPE_PODMAN_GRAPHROOT:-$CAPE_RUNTIME_ROOT/podman/graphroot}"
CAPE_PODMAN_TMPDIR="${CAPE_PODMAN_TMPDIR:-$CAPE_RUNTIME_ROOT/podman/tmp}"
CAPE_PODMAN_RUNROOT="${CAPE_PODMAN_RUNROOT:-$CAPE_RUNTIME_ROOT/podman/runroot}"
CAPE_NODE_RUNTIME_CACHE_CONFIG="${CAPE_NODE_RUNTIME_CACHE_CONFIG:-$CAPE_RUNTIME_CACHE}"
CAPE_NODE_STATE_DIR="${CAPE_NODE_STATE_DIR:-$CAPE_NODE_BUNDLE/pool/$CAPE_NODE_ID}"
CAPE_AGENT_LOG="${CAPE_AGENT_LOG:-$CAPE_NODE_STATE_DIR/agent.log}"
CAPE_AGENT_LOG_DIR="${CAPE_AGENT_LOG_DIR:-$CAPE_NODE_STATE_DIR/agent-logs}"
CAPE_AGENT_STATE_PATH="${CAPE_AGENT_STATE_PATH:-$CAPE_NODE_STATE_DIR/agent-state.json}"

case "$CAPE_NODE_ID" in
  ''|*[!A-Za-z0-9_.-]*) die "CAPE_NODE_ID may contain only letters, digits, dot, underscore, and dash" ;;
esac
case "$CAPE_POOL_MODE" in
  exclusive|shared) ;;
  *) die "CAPE_POOL_MODE must be exclusive or shared" ;;
esac
if [ "$CAPE_POOL_MODE" = shared ] && \
   [ "$CAPE_GPU_SLOTS_EXPLICIT" != 1 ]; then
  die "shared pools require explicit CAPE_GPU_SLOTS"
fi
case "$CAPE_GPU_SLOTS" in
  ''|*[!0-9]*) die "CAPE_GPU_SLOTS must be a positive integer" ;;
esac
[ "$CAPE_GPU_SLOTS" -ge 1 ] || die "CAPE_GPU_SLOTS must be at least 1"
if [ "$CAPE_POOL_MODE" = exclusive ] && [ "$CAPE_GPU_SLOTS" -ne 1 ]; then
  die "exclusive pools require CAPE_GPU_SLOTS=1"
fi
case "$CAPE_POOL_PLACEMENT" in
  pack|spread) ;;
  *) die "CAPE_POOL_PLACEMENT must be pack or spread" ;;
esac
case "$CAPE_ALLOCATION_ID" in
  ''|*[!A-Za-z0-9_.-]*) die "CAPE_ALLOCATION_ID may contain only letters, digits, dot, underscore, and dash" ;;
esac
case "$CAPE_NODE_CPU_CORES" in
  ''|*[!0-9]*) die "CAPE_NODE_CPU_CORES must be a positive integer" ;;
esac
[ "$CAPE_NODE_CPU_CORES" -ge 1 ] || die "CAPE_NODE_CPU_CORES must be at least 1"
case "$CAPE_NODE_MEMORY_GB" in
  ''|*[!0-9]*) die "CAPE_NODE_MEMORY_GB must be a positive integer" ;;
esac
[ "$CAPE_NODE_MEMORY_GB" -ge 1 ] || die "CAPE_NODE_MEMORY_GB must be at least 1"
case "$CAPE_FORCE_INSTALL" in
  0|1) ;;
  *) die "CAPE_FORCE_INSTALL must be 0 or 1" ;;
esac
case "$CAPE_NO_WARM_SESSIONS" in
  0|1) ;;
  *) die "CAPE_NO_WARM_SESSIONS must be 0 or 1" ;;
esac
case "$CAPE_STAY_AFTER_REMOVAL" in
  0|1) ;;
  *) die "CAPE_STAY_AFTER_REMOVAL must be 0 or 1" ;;
esac
case "${CAPE_APPTAINER_UNSQUASH:-0}" in
  0|1) ;;
  *) die "CAPE_APPTAINER_UNSQUASH must be 0 or 1" ;;
esac
case "${CAPE_NO_CONTAIN:-0}" in
  0|1) ;;
  *) die "CAPE_NO_CONTAIN must be 0 or 1" ;;
esac
runtime_filter_text="${CAPE_RUNTIME_ADAPTERS//,/ }"
runtime_filters=()
raw_runtime_filters=()
declare -A runtime_filter_seen=()
read -r -a raw_runtime_filters <<< "$runtime_filter_text"
for runtime_filter in "${raw_runtime_filters[@]}"; do
  case "$runtime_filter" in
    apptainer|bubblewrap|podman|local-process) ;;
    *) die "CAPE_RUNTIME_ADAPTERS entries must be apptainer, bubblewrap, podman, or local-process" ;;
  esac
  if [ -z "${runtime_filter_seen[$runtime_filter]:-}" ]; then
    runtime_filters+=("$runtime_filter")
    runtime_filter_seen["$runtime_filter"]=1
  fi
done
bubblewrap_system_paths=()
declare -A bubblewrap_system_path_seen=()
bubblewrap_system_path_text="${CAPE_BUBBLEWRAP_SYSTEM_PATHS//,/$'\n'}"
command -v readlink >/dev/null 2>&1 || die "readlink is required"
while IFS= read -r bubblewrap_system_path; do
  bubblewrap_system_path="${bubblewrap_system_path%$'\r'}"
  bubblewrap_system_path="${bubblewrap_system_path#"${bubblewrap_system_path%%[![:space:]]*}"}"
  bubblewrap_system_path="${bubblewrap_system_path%"${bubblewrap_system_path##*[![:space:]]}"}"
  [ -n "$bubblewrap_system_path" ] || continue
  case "$bubblewrap_system_path" in
    /*) ;;
    *) die "CAPE_BUBBLEWRAP_SYSTEM_PATHS entries must be absolute paths" ;;
  esac
  [ -d "$bubblewrap_system_path" ] || \
    die "Bubblewrap system path is not a directory: $bubblewrap_system_path"
  bubblewrap_system_path_input="$bubblewrap_system_path"
  bubblewrap_system_path="$(readlink -f -- "$bubblewrap_system_path")" || \
    die "failed to resolve Bubblewrap system path: $bubblewrap_system_path_input"
  case "$bubblewrap_system_path" in
    /|/proc|/proc/*|/sys|/sys/*|/dev|/dev/*|/run|/run/*)
      die "CAPE_BUBBLEWRAP_SYSTEM_PATHS must not expose host root or virtual kernel trees: $bubblewrap_system_path_input"
      ;;
  esac
  if [ -z "${bubblewrap_system_path_seen[$bubblewrap_system_path]:-}" ]; then
    bubblewrap_system_paths+=("$bubblewrap_system_path")
    bubblewrap_system_path_seen["$bubblewrap_system_path"]=1
  fi
done <<< "$bubblewrap_system_path_text"
redact_key_text="${CAPE_NODE_LOG_REDACT_ENV_KEYS//,/ }"
redact_keys=()
raw_redact_keys=()
declare -A redact_key_seen=()
read -r -a raw_redact_keys <<< "$redact_key_text"
for redact_key in "${raw_redact_keys[@]}"; do
  [[ "$redact_key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || \
    die "CAPE_NODE_LOG_REDACT_ENV_KEYS entries must be environment-variable names"
  if [ -z "${redact_key_seen[$redact_key]:-}" ]; then
    redact_keys+=("$redact_key")
    redact_key_seen["$redact_key"]=1
  fi
done
[[ "$CAPE_HEARTBEAT_INTERVAL" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
  die "CAPE_HEARTBEAT_INTERVAL must be numeric"
awk -v value="$CAPE_HEARTBEAT_INTERVAL" 'BEGIN {exit !(value > 0)}' || \
  die "CAPE_HEARTBEAT_INTERVAL must be greater than zero"
for timing_name in CAPE_NODE_DRAIN_BEFORE_SEC CAPE_NODE_CLEANUP_BUFFER_SEC; do
  timing_value="${!timing_name}"
  [[ "$timing_value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
    die "$timing_name must be a non-negative number"
done
if [ -n "${CAPE_ALLOCATION_DEADLINE:-}" ]; then
  [[ "$CAPE_ALLOCATION_DEADLINE" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
    die "CAPE_ALLOCATION_DEADLINE must be a positive Unix timestamp"
  awk -v value="$CAPE_ALLOCATION_DEADLINE" 'BEGIN {exit !(value > 0)}' || \
    die "CAPE_ALLOCATION_DEADLINE must be greater than zero"
fi
[ -n "$CAPE_NODE_RUNTIME_CACHE_CONFIG" ] || \
  die "CAPE_NODE_RUNTIME_CACHE_CONFIG must not be empty"

case "$CAPE_NODE_VENV" in
  /*) ;;
  *) CAPE_NODE_VENV="$PWD/${CAPE_NODE_VENV#./}" ;;
esac
CAPE_NODE_BIN="${CAPE_NODE_BIN_OVERRIDE:-$CAPE_NODE_VENV/bin/pycape}"
CAPE_NODE_BIN="${CAPE_NODE_BIN/#\~/$HOME}"
command -v tee >/dev/null 2>&1 || die "tee is required"

if [ -n "$CAPE_NODE_BIN_OVERRIDE" ]; then
  CAPE_NODE_BIN="$(cape_resolve_executable "$CAPE_NODE_BIN" CAPE_NODE_BIN)" || exit 2
  if [ -x "$CAPE_NODE_VENV/bin/python" ]; then
    VENV_PYTHON="$CAPE_NODE_VENV/bin/python"
  else
    VENV_PYTHON="$(command -v python3 2>/dev/null || true)"
  fi
  [ -n "$VENV_PYTHON" ] || die "python3 is required for the controller health probe"
  log "using CAPE executable selection: $CAPE_NODE_BIN_SELECTION"
else
mkdir -p "$CAPE_NODE_VENV_ROOT"
install_lock="$CAPE_NODE_VENV_ROOT/.install-node-${CAPE_NODE_VENV_HOST_ID}-$(uname -m)-python3.12.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$install_lock"
  flock 9
fi

VENV_PYTHON="$CAPE_NODE_VENV/bin/python"
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

  if [ -n "$CAPE_NODE_PYTHON_312" ]; then
    if [[ "$CAPE_NODE_PYTHON_312" == */* ]]; then
      [ -x "$CAPE_NODE_PYTHON_312" ] || \
        die "CAPE_NODE_PYTHON_312 is not executable: $CAPE_NODE_PYTHON_312"
    else
      command -v "$CAPE_NODE_PYTHON_312" >/dev/null 2>&1 || \
        die "CAPE_NODE_PYTHON_312 was not found: $CAPE_NODE_PYTHON_312"
    fi
    python312_cmd=("$CAPE_NODE_PYTHON_312")
    python312_description="$CAPE_NODE_PYTHON_312"
    python312_can_bootstrap_venv "${python312_cmd[@]}" || \
      die "$CAPE_NODE_PYTHON_312 cannot bootstrap a pip-enabled venv"
  elif command -v python3.12 >/dev/null 2>&1 && \
       python312_can_bootstrap_venv python3.12; then
    python312_cmd=(python3.12)
    python312_description="$(command -v python3.12)"
  else
    if [ -z "$CAPE_NODE_PIXI_BIN" ]; then
      if command -v pixi >/dev/null 2>&1; then
        CAPE_NODE_PIXI_BIN="$(command -v pixi)"
      fi
    fi
    if [ -n "$CAPE_NODE_PIXI_BIN" ] && [[ "$CAPE_NODE_PIXI_BIN" != */* ]]; then
      resolved_pixi="$(command -v "$CAPE_NODE_PIXI_BIN" 2>/dev/null || true)"
      [ -z "$resolved_pixi" ] || CAPE_NODE_PIXI_BIN="$resolved_pixi"
    fi
    [ -n "$CAPE_NODE_PIXI_BIN" ] && [ -x "$CAPE_NODE_PIXI_BIN" ] || \
      die "a pip-enabled Python 3.12 bootstrap and Pixi were not found; set CAPE_NODE_PYTHON_312 or CAPE_NODE_PIXI_BIN"
    python312_cmd=("$CAPE_NODE_PIXI_BIN" exec --no-progress --spec "python=3.12" python)
    python312_description="Pixi ($CAPE_NODE_PIXI_BIN)"
    log "usable Python 3.12 venv bootstrap not found; bootstrapping it with $python312_description"
  fi

  bootstrap_version="$("${python312_cmd[@]}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" || \
    die "failed to start Python through $python312_description"
  [ "$bootstrap_version" = 3.12 ] || \
    die "Python bootstrap must provide 3.12, got $bootstrap_version"
  python312_bootstrap_mode="$(python312_bootstrap_mode "${python312_cmd[@]}")" || \
    die "$python312_description cannot bootstrap a pip-enabled venv"

  if [ -d "$CAPE_NODE_VENV" ] && [ ! -f "$CAPE_NODE_VENV/pyvenv.cfg" ]; then
    die "CAPE_NODE_VENV exists but is not a Python virtual environment: $CAPE_NODE_VENV"
  fi
  mkdir -p "$(dirname "$CAPE_NODE_VENV")"
  if [ -f "$CAPE_NODE_VENV/pyvenv.cfg" ]; then
    log "rebuilding incompatible Python 3.12 venv: $CAPE_NODE_VENV"
    create_python312_venv "$python312_bootstrap_mode" "$CAPE_NODE_VENV" 1 \
      "${python312_cmd[@]}" || \
      die "failed to rebuild the node Python 3.12 venv"
  else
    log "creating Python 3.12 venv: $CAPE_NODE_VENV"
    create_python312_venv "$python312_bootstrap_mode" "$CAPE_NODE_VENV" 0 \
      "${python312_cmd[@]}" || \
      die "failed to create the node Python 3.12 venv"
  fi
fi

[ -x "$VENV_PYTHON" ] || die "node venv Python is missing: $VENV_PYTHON"
[ "$($VENV_PYTHON -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = 3.12 ] || \
  die "node venv does not use Python 3.12"

install_marker="$CAPE_NODE_VENV/.cape-install-fingerprint"
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
if [ "$CAPE_FORCE_INSTALL" = 0 ] && [ -x "$CAPE_NODE_BIN" ] && [ -r "$install_marker" ]; then
  recorded_fingerprint="$(sed -n '1p' "$install_marker")"
  if [ "$recorded_fingerprint" = "$install_fingerprint" ] && \
     "$VENV_PYTHON" -c 'import aiohttp, pycape' >/dev/null 2>&1; then
    install_ready=1
  fi
fi

if [ "$install_ready" = 1 ]; then
  log "reusing CAPE installation in $CAPE_NODE_VENV"
else
  log "installing CAPE from $CAPE_PROJECT into $CAPE_NODE_VENV"
  "$VENV_PYTHON" -m pip install \
    --disable-pip-version-check \
    --editable "$CAPE_PROJECT" || die "CAPE installation failed"
  "$VENV_PYTHON" -c \
    'import sys, aiohttp, pycape; assert sys.version_info[:2] == (3, 12)' || \
    die "installed CAPE or aiohttp could not be imported"
  [ -x "$CAPE_NODE_BIN" ] || die "CAPE executable was not installed: $CAPE_NODE_BIN"
  marker_tmp="$install_marker.tmp.$$"
  printf '%s\n' "$install_fingerprint" > "$marker_tmp"
  mv -- "$marker_tmp" "$install_marker"
fi

if command -v flock >/dev/null 2>&1; then
  flock -u 9
  exec 9>&-
fi
fi

CAPE_NODE_BIN="$(cape_resolve_executable "$CAPE_NODE_BIN" CAPE_NODE_BIN)" || exit 2
cape_inspect_executable "$CAPE_NODE_BIN_SELECTION" "$CAPE_NODE_BIN" || exit 2

health_url="${CAPE_URL%/}/health"
if ! "$VENV_PYTHON" -c '
import json, sys
from urllib.request import urlopen
with urlopen(sys.argv[1], timeout=5) as response:
    body = json.load(response)
    if response.status != 200 or body.get("ok") is not True:
        raise SystemExit(1)
' "$health_url"; then
  die "controller health check failed: $health_url"
fi
log "controller reachable at $CAPE_URL"

if [[ "$CAPE_NVIDIA_SMI" == */* ]]; then
  [ -x "$CAPE_NVIDIA_SMI" ] || die "nvidia-smi is not executable: $CAPE_NVIDIA_SMI"
else
  command -v "$CAPE_NVIDIA_SMI" >/dev/null 2>&1 || die "nvidia-smi was not found"
fi

visible_output="$("$CAPE_NVIDIA_SMI" --query-gpu=uuid --format=csv,noheader)" || \
  die "nvidia-smi could not enumerate visible GPUs"
visible_gpu_uuids=()
while IFS= read -r uuid; do
  uuid="$(printf '%s' "$uuid" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [ -z "$uuid" ] || visible_gpu_uuids+=("$uuid")
done <<< "$visible_output"
[ "${#visible_gpu_uuids[@]}" -gt 0 ] || die "no GPUs are visible on this node"

selected_gpu_uuids=()
if [ -n "${CAPE_GPU_UUIDS:-}" ]; then
  requested_gpu_text="${CAPE_GPU_UUIDS//,/ }"
  read -r -a selected_gpu_uuids <<< "$requested_gpu_text"
elif [ -n "${CAPE_GPU_COUNT:-}" ]; then
  case "$CAPE_GPU_COUNT" in
    ''|*[!0-9]*) die "CAPE_GPU_COUNT must be a positive integer" ;;
  esac
  [ "$CAPE_GPU_COUNT" -ge 1 ] || die "CAPE_GPU_COUNT must be at least 1"
  [ "$CAPE_GPU_COUNT" -le "${#visible_gpu_uuids[@]}" ] || \
    die "CAPE_GPU_COUNT=$CAPE_GPU_COUNT exceeds ${#visible_gpu_uuids[@]} visible GPUs"
  selected_gpu_uuids=("${visible_gpu_uuids[@]:0:CAPE_GPU_COUNT}")
else
  selected_gpu_uuids=("${visible_gpu_uuids[@]}")
fi
[ "${#selected_gpu_uuids[@]}" -gt 0 ] || die "no GPU UUIDs were selected"

declare -A visible_gpu_set=()
for uuid in "${visible_gpu_uuids[@]}"; do
  visible_gpu_set["$uuid"]=1
done
declare -A selected_gpu_set=()
gpu_args=()
for uuid in "${selected_gpu_uuids[@]}"; do
  case "$uuid" in
    GPU-*) ;;
    *) die "expected a whole-GPU UUID beginning with GPU-, got: $uuid" ;;
  esac
  [ -n "${visible_gpu_set[$uuid]:-}" ] || die "selected GPU is not visible on this node: $uuid"
  [ -z "${selected_gpu_set[$uuid]:-}" ] || die "GPU UUID was selected more than once: $uuid"
  selected_gpu_set["$uuid"]=1
  gpu_args+=(--gpu-uuid "$uuid")
done

[ "$CAPE_NODE_CPU_CORES" -ge "${#selected_gpu_uuids[@]}" ] || \
  die "registered CPU cores must be at least the selected GPU count"
[ "$CAPE_NODE_MEMORY_GB" -ge "${#selected_gpu_uuids[@]}" ] || \
  die "registered memory GiB must be at least the selected GPU count"

umask 077
mkdir -p \
  "$CAPE_NODE_STATE_DIR" \
  "$CAPE_AGENT_LOG_DIR" \
  "$CAPE_RUNTIME_CACHE" \
  "$CAPE_RUNTIME_TMP" \
  "$CAPE_RUNTIME_DIR" \
  "$CAPE_APPTAINER_CACHE" \
  "$CAPE_APPTAINER_TMP" \
  "$CAPE_APPTAINER_RUNTIME_DIR" \
  "$CAPE_BUBBLEWRAP_CACHE" \
  "$CAPE_BUBBLEWRAP_TMP" \
  "$CAPE_BUBBLEWRAP_RUNTIME_DIR" \
  "$CAPE_PODMAN_GRAPHROOT" \
  "$CAPE_PODMAN_TMPDIR" \
  "$CAPE_PODMAN_RUNROOT"
cape_record_executable node "$CAPE_NODE_BIN" \
  "$CAPE_NODE_STATE_DIR/implementation.txt"

mkdir -p "$CAPE_NODE_STATE_DIR/credentials"
chmod 700 "$CAPE_NODE_STATE_DIR/credentials"
join_args=(
  node enroll
  --node-token-out "$CAPE_NODE_STATE_DIR/credentials/enrollment.json"
  --controller-url "$CAPE_URL"
  --node-id "$CAPE_NODE_ID"
  --pool "$CAPE_POOL"
  --pool-mode "$CAPE_POOL_MODE"
  --gpu-slots "$CAPE_GPU_SLOTS"
  --pool-placement "$CAPE_POOL_PLACEMENT"
  --allocation-id "$CAPE_ALLOCATION_ID"
  --cpu-cores "$CAPE_NODE_CPU_CORES"
  --memory-gb "$CAPE_NODE_MEMORY_GB"
  --log-dir "$CAPE_AGENT_LOG_DIR"
  --state-path "$CAPE_AGENT_STATE_PATH"
  --execution-state-dir "$CAPE_EXECUTION_STATE_DIR"
  --heartbeat-interval "$CAPE_HEARTBEAT_INTERVAL"
  --drain-before-sec "$CAPE_NODE_DRAIN_BEFORE_SEC"
  --cleanup-buffer-sec "$CAPE_NODE_CLEANUP_BUFFER_SEC"
  --runtime-cache-config "$CAPE_NODE_RUNTIME_CACHE_CONFIG"
)
if [ "${CAPE_NODE_RESUME:-0}" = 1 ]; then join_args+=(--resume); fi
join_args+=("${gpu_args[@]}")
for runtime_filter in "${runtime_filters[@]}"; do
  join_args+=(--runtime-adapter "$runtime_filter")
done
for bubblewrap_system_path in "${bubblewrap_system_paths[@]}"; do
  join_args+=(--bubblewrap-system-path "$bubblewrap_system_path")
done

while IFS= read -r cached_image; do
  cached_image="${cached_image%$'\r'}"
  [ -z "$cached_image" ] || join_args+=(--cached-image "$cached_image")
done <<< "$CAPE_NODE_CACHED_IMAGES"
[ "$CAPE_NO_WARM_SESSIONS" != 1 ] || join_args+=(--no-warm-sessions)
[ "$CAPE_STAY_AFTER_REMOVAL" != 1 ] || join_args+=(--stay-after-removal)
for redact_key in "${redact_keys[@]}"; do
  join_args+=(--log-redact-env-key "$redact_key")
done

resolve_runtime_binary() {
  local configured="$1"
  local command_name="$2"
  local label="$3"
  if [ -n "$configured" ]; then
    if [[ "$configured" == */* ]]; then
      configured="${configured/#\~/$HOME}"
      [ -x "$configured" ] || die "$label is not executable: $configured"
      printf '%s\n' "$configured"
    else
      command -v "$configured" 2>/dev/null || die "$label was not found: $configured"
    fi
  else
    command -v "$command_name" 2>/dev/null || true
  fi
}

APPTAINER_BIN="$(resolve_runtime_binary "${CAPE_APPTAINER_BIN:-}" apptainer CAPE_APPTAINER_BIN)"
BUBBLEWRAP_BIN="$(resolve_runtime_binary "${CAPE_BUBBLEWRAP_BIN:-}" bwrap CAPE_BUBBLEWRAP_BIN)"
PODMAN_BIN="$(resolve_runtime_binary "${CAPE_PODMAN_BIN:-}" podman CAPE_PODMAN_BIN)"

join_args+=(
  --apptainer-cache "$CAPE_APPTAINER_CACHE"
  --apptainer-tmp "$CAPE_APPTAINER_TMP"
  --apptainer-runtime-dir "$CAPE_APPTAINER_RUNTIME_DIR"
  --bubblewrap-cache "$CAPE_BUBBLEWRAP_CACHE"
  --bubblewrap-tmp "$CAPE_BUBBLEWRAP_TMP"
  --bubblewrap-runtime-dir "$CAPE_BUBBLEWRAP_RUNTIME_DIR"
  --podman-graphroot "$CAPE_PODMAN_GRAPHROOT"
  --podman-tmpdir "$CAPE_PODMAN_TMPDIR"
  --podman-runroot "$CAPE_PODMAN_RUNROOT"
)
[ -z "$APPTAINER_BIN" ] || join_args+=(--apptainer-binary "$APPTAINER_BIN")
[ -z "$BUBBLEWRAP_BIN" ] || join_args+=(--bubblewrap-binary "$BUBBLEWRAP_BIN")
while IFS= read -r read_only_path; do
  [ -z "$read_only_path" ] || join_args+=(--bubblewrap-read-only-path "$read_only_path")
done <<< "${CAPE_BUBBLEWRAP_READ_ONLY_PATHS//,/$'\n'}"
[ -z "$PODMAN_BIN" ] || join_args+=(--podman-binary "$PODMAN_BIN")
[ -z "$CAPE_APPTAINER_PROBE_IMAGE" ] || \
  join_args+=(--apptainer-probe-image "$CAPE_APPTAINER_PROBE_IMAGE")
[ -z "$CAPE_PODMAN_PROBE_IMAGE" ] || \
  join_args+=(--podman-probe-image "$CAPE_PODMAN_PROBE_IMAGE")
[ "${CAPE_APPTAINER_UNSQUASH:-0}" != 1 ] || join_args+=(--apptainer-unsquash)
[ "${CAPE_NO_CONTAIN:-0}" != 1 ] || join_args+=(--no-contain)

if [ -n "${CAPE_ALLOCATION_DEADLINE:-}" ]; then
  join_args+=(--allocation-deadline "$CAPE_ALLOCATION_DEADLINE")
fi

if [ "${#runtime_filters[@]}" -eq 0 ]; then
  runtime_filter_log=auto:apptainer,bubblewrap,podman
else
  runtime_filter_log="${runtime_filters[*]}"
fi
{
  log "joining controller=$CAPE_URL allocation=$CAPE_ALLOCATION_ID runtime_filter=$runtime_filter_log"
  log "resources cpu=$CAPE_NODE_CPU_CORES memory_gb=$CAPE_NODE_MEMORY_GB gpus=${selected_gpu_uuids[*]}"
  log "CAPE will announce REGISTERING before runtime probes and register ACTIVE after they pass"
  if [ "$CAPE_STAY_AFTER_REMOVAL" = 1 ]; then
    log "agent remains in the foreground after removal (explicit opt-out); log=$CAPE_AGENT_LOG"
  else
    log "agent remains in the foreground and exits 0 after authoritative removal; log=$CAPE_AGENT_LOG"
  fi
} | tee "$CAPE_AGENT_LOG"

if [ -n "${CAPE_ROLE_PID_FILE:-}" ]; then
  python3 "$SCRIPT_DIR/role_record.py" write --pid-file "$CAPE_ROLE_PID_FILE" \
    --controller-url "$CAPE_URL" --node-id "$CAPE_NODE_ID" --pool "$CAPE_POOL" \
    --allocation-id "$CAPE_ALLOCATION_ID"
fi
cape_publish_role_pid
unset CAPE_ROLE_PID_FILE CAPE_ROLE_LABEL
exec > >(tee -a "$CAPE_AGENT_LOG") 2>&1
# Credentials travel in the environment, which only this user can read, not on the world-readable command line;
# both implementations read them there and drop them before starting workloads.
export CAPE_TOKEN="$CAPE_NODE_TOKEN"
unset CAPE_NODE_TOKEN CAPE_ADMIN_TOKEN CAPE_JOIN_TOKEN
exec "$CAPE_NODE_BIN" "${join_args[@]}"
