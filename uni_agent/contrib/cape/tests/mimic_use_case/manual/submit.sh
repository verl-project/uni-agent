#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Install/reuse a local CAPE client and submit exactly one asynchronous workload
# to an existing controller. Querying and waiting belong in manual/status.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/credentials.sh"
cape_manual_credentials "$@"
set -- "${CAPE_MANUAL_ARGS[@]}"

submit_phase_entry="$(date +%s.%N)"
submit_phase_handoff_loaded="$submit_phase_entry"
submit_phase_client_ready="$submit_phase_entry"
submit_phase_pre_binary="$submit_phase_entry"
submit_phase_binary_return="$submit_phase_entry"
submit_timing_file="${CAPE_SUBMIT_TIMING_FILE:-}"

write_submit_timing() {
  local rc="$1"
  [ -n "$submit_timing_file" ] || return 0
  local exit_stamp target temporary
  exit_stamp="$(date +%s.%N)"
  target="$submit_timing_file"
  case "$target" in
    /*) ;;
    *) target="$PWD/${target#./}" ;;
  esac
  umask 077
  mkdir -p "$(dirname "$target")" || return 0
  temporary="$target.tmp.$$"
  if printf '%s=%s\n' \
      entry "$submit_phase_entry" \
      handoff_loaded "$submit_phase_handoff_loaded" \
      client_ready "$submit_phase_client_ready" \
      pre_binary "$submit_phase_pre_binary" \
      binary_return "$submit_phase_binary_return" \
      exit "$exit_stamp" \
      exit_code "$rc" > "$temporary"; then
    mv -- "$temporary" "$target" || true
  else
    rm -f "$temporary"
  fi
}

submit_timing_on_exit() {
  local rc="$?"
  trap - EXIT
  write_submit_timing "$rc"
  exit "$rc"
}
trap submit_timing_on_exit EXIT

usage() {
  cat <<'EOF'
Credential input: --token-file FILE (private personal credential JSON).
Use an explicit controller URL and pool; controller.sh no longer generates handoffs.

Usage:
  submit.sh [OPTIONS] -- COMMAND [ARG ...]
  submit.sh [OPTIONS] --mimic A|B

submit.sh submits exactly one asynchronous CAPE run request and prints only
its req-... identifier on stdout. Use status.sh to inspect, follow, or wait for
the request.

Connection and identity:
  --controller-env FILE       Optional controller.env from controller.sh
  --controller-url URL        Override the controller URL
  --pool POOL                 Override the pool
  --user USER                 Required request owner
  --token TOKEN               Explicit submitter/admin token
  --no-token                  Omit credentials for a negative authentication probe
  --user-tokens FILE          Submit-host-local CSV mapping USER,TOKEN
  --session-id ID                Logical ownership/reuse identity (default: USER-task)
                              Both values participate in reuse matching.
  --suspend-on-failure
                              Opt out of the default that keeps a reusable
                              session warm after a nonzero workload exit.

Image, resources, and placement:
  --image IMAGE               Container image
  --profile PROFILE           Controller-side profile
  --runtime-adapter ADAPTER   apptainer|bubblewrap|podman|local-process
  --gpus N                    Requested GPUs (default: 1)
  --cpu-cores N               Requested CPU cores (default: 1)
  --memory-gb N               Requested memory GiB (default: 1)
  --max-duration-seconds N    Workload deadline
  --command-timeout-seconds N Per-command admission bound; defaults to workload deadline
  --hard-gpu-enforcement      Require a hard GPU isolation runtime
  --port N                    Requested port; repeatable
  --locality VALUE            Placement locality constraint
  --isolation-policy POLICY   Isolation policy (default: default)

The selected pool's join-declared mode determines exclusive versus shared GPU
admission. There is no request-side --gpu-mode option.

Execution surface:
  --bind SRC:DST[:ro|rw]      Explicit mount; repeatable
  --env KEY=VALUE             Workload environment entry; repeatable
  --cwd PATH                  Working directory inside the container
  --workspace-id ID           Reuse an authorized retained managed image
  --workspace-target PATH     Managed image bind target (mimic default: /cape-workspace)
  --log-path PATH             Runtime log path
  --artifact-path PATH        Artifact path; repeatable
  --mimic A|B                 Submit one built-in mimic workload
  --user-root DIR             Mimic user's host workspace/output root

Optional evidence:
  --record FILE               Append analyzer-compatible submission CSV
  --label LABEL               CSV request label (default: workload or A/B)
  CAPE_SUBMIT_TIMING_FILE     Optional per-call phase file containing entry,
                              handoff-loaded, client-ready, pre-binary,
                              binary-return, exit, and exit-code fields.
                              Values and credentials are never recorded.

Client preparation:
  --resolve-client            Install or reuse the submitter CAPE client, print
                              its resolved CAPE executable path on stdout, and exit
                              without submitting. Callers that submit many
                              requests (for example concurrency.sh) run this
                              once and export the printed path as
                              CAPE_SUBMIT_BIN so later submissions skip
                              re-validating the client environment. No
                              controller, token, or workload arguments are
                              needed in this mode.

Local client overrides:
  CAPE_SUBMIT_BIN             Implementation selector (default: rscape).
                              Set pycape to use the role-local Python venv;
                              an explicit path uses that prepared executable.
  CAPE_SUBMIT_REPO_ROOT       Submit host's combined CAPE repository root
  CAPE_SUBMIT_PROJECT         Submit host's Python project directory
                              (default: SUBMIT_REPO_ROOT/pycape)
  CAPE_SUBMIT_BUNDLE          Submit-host/node-visible base for mimic user roots
  CAPE_SUBMIT_VENV_ROOT       Role-local venv root (default: ./venv)
  CAPE_SUBMIT_VENV_HOST_ID    Host component of the default venv identity
  CAPE_SUBMIT_VENV            Exact Python 3.12 venv path
  CAPE_SUBMIT_PYTHON_312      Explicit Python 3.12 executable
  CAPE_SUBMIT_PIXI_BIN        Explicit Pixi executable
  CAPE_FORCE_INSTALL=1        Force a CAPE reinstall

Most command-line options also have CAPE_SUBMIT_* environment equivalents,
for example CAPE_SUBMIT_USER, CAPE_SUBMIT_TOKEN, CAPE_SUBMIT_GPUS, and
CAPE_SUBMIT_RECORD. CAPE_SUBMIT_RUN_TAG overrides the automatic mimic run tag.
Options must precede the `--` command separator.

controller.env is executable shell syntax and contains credentials. Its inline
token map removes any dependency on the controller host's token-file path.
Source or pass only a trusted file produced by the matching controller.sh run.

Mimic A runs gemm_silu.py. Mimic B installs CUTLASS DSL when needed and runs
cutedsl_add.py. Both create and bind USER_ROOT/{workspace,output}, mount the
local mimic source read-only, request one GPU in the default image policy, and
reuse USER/task unless overridden. The controller prefers Apptainer when both
Apptainer and Podman can serve that policy. Every bind source must be visible
at the same absolute path from the GPU node.
EOF
}

die() {
  printf 'submit.sh: %s\n' "$*" >&2
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
  printf '[submit] %s\n' "$*" >&2
}

require_value() {
  [ "$#" -ge 2 ] && [ -n "$2" ] || die "$1 requires a value"
}

load_controller_environment() {
  local path="$1"
  local assignments
  local local_name
  local -a portable_names=(
    CAPE_CONTROLLER_ENV_VERSION RUN_ID CASE CAPE_POOL CAPE_PORT
    CONTROLLER_HOST CAPE_URL CAPE_CONTROLLER_URL CAPE_ADMIN_TOKEN
    CAPE_JOIN_TOKEN CAPE_PROBE_NODE_TOKEN CAPE_USER_TOKENS_INLINE
  )
  assignments="$({
    set -euo pipefail
    for local_name in "${portable_names[@]}"; do unset "$local_name"; done
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
controller_url_override=""
pool_override=""
user_override=""
token_override=""
token_override_set=0
no_token=0
user_tokens_override=""
user_tokens_override_set=0
session_id_override=""
suspend_on_failure=0
image_override=""
profile_override=""
runtime_adapter_override=""
gpus_override=""
cpu_cores_override=""
memory_gb_override=""
max_duration_override=""
command_timeout_override=""
hard_gpu_enforcement=0
locality_override=""
isolation_policy_override=""
cwd_override=""
workspace_id_override=""
workspace_target_override=""
log_path_override=""
mimic_kind=""
user_root_override=""
record_override=""
label_override=""
resolve_client=0
declare -a bind_values=()
declare -a env_values=()
declare -a port_values=()
declare -a artifact_values=()
declare -a workload_command=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --controller-env)
      require_value "$@"
      controller_env="$2"
      shift 2
      ;;
    --controller-env=*)
      controller_env="${1#*=}"
      [ -n "$controller_env" ] || die "--controller-env requires a value"
      shift
      ;;
    --controller-url)
      require_value "$@"
      controller_url_override="$2"
      shift 2
      ;;
    --controller-url=*)
      controller_url_override="${1#*=}"
      [ -n "$controller_url_override" ] || die "--controller-url requires a value"
      shift
      ;;
    --pool)
      require_value "$@"
      pool_override="$2"
      shift 2
      ;;
    --pool=*)
      pool_override="${1#*=}"
      [ -n "$pool_override" ] || die "--pool requires a value"
      shift
      ;;
    --user)
      require_value "$@"
      user_override="$2"
      shift 2
      ;;
    --user=*)
      user_override="${1#*=}"
      [ -n "$user_override" ] || die "--user requires a value"
      shift
      ;;
    --token)
      require_value "$@"
      token_override="$2"
      token_override_set=1
      shift 2
      ;;
    --token=*)
      token_override="${1#*=}"
      [ -n "$token_override" ] || die "--token requires a value"
      token_override_set=1
      shift
      ;;
    --no-token)
      no_token=1
      shift
      ;;
    --user-tokens)
      require_value "$@"
      user_tokens_override="$2"
      user_tokens_override_set=1
      shift 2
      ;;
    --user-tokens=*)
      user_tokens_override="${1#*=}"
      [ -n "$user_tokens_override" ] || die "--user-tokens requires a value"
      user_tokens_override_set=1
      shift
      ;;
    --session-id)
      require_value "$@"
      session_id_override="$2"
      shift 2
      ;;
    --session-id=*)
      session_id_override="${1#*=}"
      [ -n "$session_id_override" ] || die "--session-id requires a value"
      shift
      ;;
    --workspace|--workspace=*)
      die "--workspace was removed; use --session-id for logical identity and --bind for paths"
      ;;
    --suspend-on-failure)
      suspend_on_failure=1
      shift
      ;;
    --image)
      require_value "$@"
      image_override="$2"
      shift 2
      ;;
    --image=*)
      image_override="${1#*=}"
      [ -n "$image_override" ] || die "--image requires a value"
      shift
      ;;
    --profile)
      require_value "$@"
      profile_override="$2"
      shift 2
      ;;
    --profile=*)
      profile_override="${1#*=}"
      [ -n "$profile_override" ] || die "--profile requires a value"
      shift
      ;;
    --runtime-adapter)
      require_value "$@"
      runtime_adapter_override="$2"
      shift 2
      ;;
    --runtime-adapter=*)
      runtime_adapter_override="${1#*=}"
      [ -n "$runtime_adapter_override" ] || die "--runtime-adapter requires a value"
      shift
      ;;
    --gpus)
      require_value "$@"
      gpus_override="$2"
      shift 2
      ;;
    --gpus=*)
      gpus_override="${1#*=}"
      [ -n "$gpus_override" ] || die "--gpus requires a value"
      shift
      ;;
    --cpu-cores)
      require_value "$@"
      cpu_cores_override="$2"
      shift 2
      ;;
    --cpu-cores=*)
      cpu_cores_override="${1#*=}"
      [ -n "$cpu_cores_override" ] || die "--cpu-cores requires a value"
      shift
      ;;
    --memory-gb)
      require_value "$@"
      memory_gb_override="$2"
      shift 2
      ;;
    --memory-gb=*)
      memory_gb_override="${1#*=}"
      [ -n "$memory_gb_override" ] || die "--memory-gb requires a value"
      shift
      ;;
    --max-duration-seconds)
      require_value "$@"
      max_duration_override="$2"
      shift 2
      ;;
    --max-duration-seconds=*)
      max_duration_override="${1#*=}"
      [ -n "$max_duration_override" ] || die "--max-duration-seconds requires a value"
      shift
      ;;
    --command-timeout-seconds)
      require_value "$@"
      command_timeout_override="$2"
      shift 2
      ;;
    --command-timeout-seconds=*)
      command_timeout_override="${1#*=}"
      [ -n "$command_timeout_override" ] || die "--command-timeout-seconds requires a value"
      shift
      ;;
    --hard-gpu-enforcement)
      hard_gpu_enforcement=1
      shift
      ;;
    --port)
      require_value "$@"
      port_values+=("$2")
      shift 2
      ;;
    --port=*)
      port_values+=("${1#*=}")
      shift
      ;;
    --locality)
      require_value "$@"
      locality_override="$2"
      shift 2
      ;;
    --locality=*)
      locality_override="${1#*=}"
      [ -n "$locality_override" ] || die "--locality requires a value"
      shift
      ;;
    --isolation-policy)
      require_value "$@"
      isolation_policy_override="$2"
      shift 2
      ;;
    --isolation-policy=*)
      isolation_policy_override="${1#*=}"
      [ -n "$isolation_policy_override" ] || die "--isolation-policy requires a value"
      shift
      ;;
    --bind)
      require_value "$@"
      bind_values+=("$2")
      shift 2
      ;;
    --bind=*)
      bind_values+=("${1#*=}")
      shift
      ;;
    --env)
      require_value "$@"
      env_values+=("$2")
      shift 2
      ;;
    --env=*)
      env_values+=("${1#*=}")
      shift
      ;;
    --workspace-id)
      require_value "$@"
      workspace_id_override="$2"
      shift 2
      ;;
    --workspace-id=*)
      workspace_id_override="${1#*=}"
      [ -n "$workspace_id_override" ] || die "--workspace-id requires a value"
      shift
      ;;
    --workspace-target)
      require_value "$@"
      workspace_target_override="$2"
      shift 2
      ;;
    --workspace-target=*)
      workspace_target_override="${1#*=}"
      [ -n "$workspace_target_override" ] || die "--workspace-target requires a value"
      shift
      ;;
    --cwd)
      require_value "$@"
      cwd_override="$2"
      shift 2
      ;;
    --cwd=*)
      cwd_override="${1#*=}"
      [ -n "$cwd_override" ] || die "--cwd requires a value"
      shift
      ;;
    --log-path)
      require_value "$@"
      log_path_override="$2"
      shift 2
      ;;
    --log-path=*)
      log_path_override="${1#*=}"
      [ -n "$log_path_override" ] || die "--log-path requires a value"
      shift
      ;;
    --artifact-path)
      require_value "$@"
      artifact_values+=("$2")
      shift 2
      ;;
    --artifact-path=*)
      artifact_values+=("${1#*=}")
      shift
      ;;
    --mimic)
      require_value "$@"
      mimic_kind="$2"
      shift 2
      ;;
    --mimic=*)
      mimic_kind="${1#*=}"
      [ -n "$mimic_kind" ] || die "--mimic requires a value"
      shift
      ;;
    --user-root)
      require_value "$@"
      user_root_override="$2"
      shift 2
      ;;
    --user-root=*)
      user_root_override="${1#*=}"
      [ -n "$user_root_override" ] || die "--user-root requires a value"
      shift
      ;;
    --record)
      require_value "$@"
      record_override="$2"
      shift 2
      ;;
    --record=*)
      record_override="${1#*=}"
      [ -n "$record_override" ] || die "--record requires a value"
      shift
      ;;
    --label)
      require_value "$@"
      label_override="$2"
      shift 2
      ;;
    --label=*)
      label_override="${1#*=}"
      [ -n "$label_override" ] || die "--label requires a value"
      shift
      ;;
    --resolve-client)
      resolve_client=1
      shift
      ;;
    --wait|--timeout|--timeout=*)
      die "$1 is not supported; submit.sh is asynchronous, use status.sh wait"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      workload_command=("$@")
      set --
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      die "workload arguments must follow the -- separator: $1"
      ;;
  esac
done

[ "$no_token" != 1 ] || [ "$token_override_set" != 1 ] || \
  die "--token and --no-token are mutually exclusive"
[ -z "$mimic_kind" ] || [ "${#workload_command[@]}" -eq 0 ] || \
  die "--mimic and an explicit workload command are mutually exclusive"
if [ "$resolve_client" != 1 ] && [ -z "$mimic_kind" ] && \
   [ "${#workload_command[@]}" -eq 0 ]; then
  usage >&2
  die "provide --mimic A|B or a workload command after --"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "$controller_env" ]; then
  case "$controller_env" in
    /*) ;;
    *) controller_env="$PWD/${controller_env#./}" ;;
  esac
  [ -r "$controller_env" ] || \
    die "controller environment is not readable: $controller_env"
  # Evaluate only a trusted handoff, then import its portable allowlist. The
  # submitter's paths, executable, venv, image, and evidence remain local.
  load_controller_environment "$controller_env"
fi
submit_phase_handoff_loaded="$(date +%s.%N)"

source "$SCRIPT_DIR/../project_root.sh"
SCRIPT_REPO_ROOT="$(cape_project_root "$SCRIPT_DIR")"
REPO_ROOT="${CAPE_SUBMIT_REPO_ROOT:-$SCRIPT_REPO_ROOT}"
CAPE_PROJECT="${CAPE_SUBMIT_PROJECT:-$REPO_ROOT/pycape}"
CAPE_SUBMIT_URL="${controller_url_override:-${CAPE_SUBMIT_URL:-${CAPE_URL:-${CAPE_CONTROLLER_URL:-}}}}"
CAPE_SUBMIT_POOL="${pool_override:-${CAPE_SUBMIT_POOL:-${CAPE_POOL:-}}}"
CAPE_SUBMIT_USER="${user_override:-${CAPE_SUBMIT_USER:-}}"
if [ -n "${CAPE_SUBMIT_WORKSPACE:-}" ]; then
  die "CAPE_SUBMIT_WORKSPACE was removed; use CAPE_SUBMIT_SESSION_ID"
fi
CAPE_SUBMIT_SESSION_ID="${session_id_override:-${CAPE_SUBMIT_SESSION_ID:-}}"
if [ -n "${CAPE_SUBMIT_TASK_ID:-}${CAPE_SUBMIT_SESSION_KEY:-}" ]; then
  die "CAPE_SUBMIT_TASK_ID/CAPE_SUBMIT_SESSION_KEY were removed; use CAPE_SUBMIT_SESSION_ID"
fi
CAPE_SUBMIT_IMAGE="${image_override:-${CAPE_SUBMIT_IMAGE:-}}"
CAPE_SUBMIT_PROFILE="${profile_override:-${CAPE_SUBMIT_PROFILE:-}}"
CAPE_SUBMIT_RUNTIME_ADAPTER="${runtime_adapter_override:-${CAPE_SUBMIT_RUNTIME_ADAPTER:-}}"
CAPE_SUBMIT_GPUS="${gpus_override:-${CAPE_SUBMIT_GPUS:-1}}"
CAPE_SUBMIT_CPU_CORES="${cpu_cores_override:-${CAPE_SUBMIT_CPU_CORES:-1}}"
CAPE_SUBMIT_MEMORY_GB="${memory_gb_override:-${CAPE_SUBMIT_MEMORY_GB:-1}}"
CAPE_SUBMIT_MAX_DURATION="${max_duration_override:-${CAPE_SUBMIT_MAX_DURATION_SECONDS:-}}"
CAPE_SUBMIT_COMMAND_TIMEOUT="${command_timeout_override:-${CAPE_SUBMIT_COMMAND_TIMEOUT_SECONDS:-}}"
CAPE_SUBMIT_LOCALITY="${locality_override:-${CAPE_SUBMIT_LOCALITY:-}}"
CAPE_SUBMIT_ISOLATION_POLICY="${isolation_policy_override:-${CAPE_SUBMIT_ISOLATION_POLICY:-default}}"
CAPE_SUBMIT_CWD="${cwd_override:-${CAPE_SUBMIT_CWD:-}}"
CAPE_SUBMIT_WORKSPACE_ID="${workspace_id_override:-${CAPE_SUBMIT_WORKSPACE_ID:-}}"
CAPE_SUBMIT_WORKSPACE_TARGET="${workspace_target_override:-${CAPE_SUBMIT_WORKSPACE_TARGET:-}}"
CAPE_SUBMIT_LOG_PATH="${log_path_override:-${CAPE_SUBMIT_LOG_PATH:-}}"
CAPE_SUBMIT_RECORD="${record_override:-${CAPE_SUBMIT_RECORD:-}}"
CAPE_SUBMIT_LABEL="${label_override:-${CAPE_SUBMIT_LABEL:-workload}}"
CAPE_SUBMIT_USER_ROOT="${user_root_override:-${CAPE_SUBMIT_USER_ROOT:-}}"
CAPE_SUBMIT_USER_TOKENS="${user_tokens_override:-${CAPE_SUBMIT_USER_TOKENS:-}}"
CAPE_SUBMIT_USER_TOKENS_INLINE="${CAPE_SUBMIT_USER_TOKENS_INLINE:-${CAPE_USER_TOKENS_INLINE:-}}"
CAPE_SUBMIT_BUNDLE="${CAPE_SUBMIT_BUNDLE:-}"
CAPE_FORCE_INSTALL="${CAPE_FORCE_INSTALL:-0}"
CAPE_SUBMIT_HARD_GPU_ENFORCEMENT="${CAPE_SUBMIT_HARD_GPU_ENFORCEMENT:-$hard_gpu_enforcement}"
CAPE_SUBMIT_VENV_ROOT="${CAPE_SUBMIT_VENV_ROOT:-$REPO_ROOT/venv}"
CAPE_SUBMIT_VENV_HOST_ID="${CAPE_SUBMIT_VENV_HOST_ID:-$(hostname -s 2>/dev/null || hostname)}"
CAPE_SUBMIT_VENV_HOST_ID="${CAPE_SUBMIT_VENV_HOST_ID//[^A-Za-z0-9_.-]/_}"
CAPE_SUBMIT_VENV="${CAPE_SUBMIT_VENV:-${CAPE_SUBMIT_VENV_ROOT}/submit-${CAPE_SUBMIT_VENV_HOST_ID}-$(uname -m)-python3.12}"
CAPE_SUBMIT_BIN_SELECTION="${CAPE_SUBMIT_BIN:-rscape}"
CAPE_SUBMIT_PYTHON_312="${CAPE_SUBMIT_PYTHON_312:-}"
CAPE_SUBMIT_PIXI_BIN="${CAPE_SUBMIT_PIXI_BIN:-}"

if [[ -v CAPE_SUBMIT_GPU_MODE ]]; then
  die "CAPE_SUBMIT_GPU_MODE was removed; select an exclusive or shared pool"
fi

if [ -n "$CAPE_SUBMIT_BUNDLE" ]; then
  case "$CAPE_SUBMIT_BUNDLE" in
    /*) ;;
    *) CAPE_SUBMIT_BUNDLE="$PWD/${CAPE_SUBMIT_BUNDLE#./}" ;;
  esac
fi

if [ "$resolve_client" != 1 ]; then
  [ -n "$CAPE_SUBMIT_URL" ] || \
    die "controller URL is required; pass --controller-env or --controller-url"
  case "$CAPE_SUBMIT_URL" in
    http://*|https://*) ;;
    *) die "controller URL must begin with http:// or https://: $CAPE_SUBMIT_URL" ;;
  esac
  CAPE_SUBMIT_URL="${CAPE_SUBMIT_URL%/}"
  [ -n "$CAPE_SUBMIT_POOL" ] || die "pool is required; pass --controller-env or --pool"
  [ -n "$CAPE_SUBMIT_USER" ] || die "request owner is required; pass --user or CAPE_SUBMIT_USER"
fi

case "$CAPE_FORCE_INSTALL" in
  0|1) ;;
  *) die "CAPE_FORCE_INSTALL must be 0 or 1" ;;
esac
case "$CAPE_SUBMIT_HARD_GPU_ENFORCEMENT" in
  0|1) hard_gpu_enforcement="$CAPE_SUBMIT_HARD_GPU_ENFORCEMENT" ;;
  *) die "CAPE_SUBMIT_HARD_GPU_ENFORCEMENT must be 0 or 1" ;;
esac
case "$CAPE_SUBMIT_GPUS" in
  ''|*[!0-9]*) die "--gpus must be a non-negative integer" ;;
esac
case "$CAPE_SUBMIT_CPU_CORES" in
  ''|*[!0-9]*) die "--cpu-cores must be a positive integer" ;;
esac
[ "$CAPE_SUBMIT_CPU_CORES" -ge 1 ] || die "--cpu-cores must be at least 1"
case "$CAPE_SUBMIT_MEMORY_GB" in
  ''|*[!0-9]*) die "--memory-gb must be a positive integer" ;;
esac
[ "$CAPE_SUBMIT_MEMORY_GB" -ge 1 ] || die "--memory-gb must be at least 1"
if [ -n "$CAPE_SUBMIT_MAX_DURATION" ]; then
  [[ "$CAPE_SUBMIT_MAX_DURATION" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
    die "--max-duration-seconds must be numeric"
  awk -v value="$CAPE_SUBMIT_MAX_DURATION" 'BEGIN {exit !(value > 0)}' || \
    die "--max-duration-seconds must be greater than zero"
fi
if [ -n "$CAPE_SUBMIT_COMMAND_TIMEOUT" ]; then
  [[ "$CAPE_SUBMIT_COMMAND_TIMEOUT" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
    die "--command-timeout-seconds must be numeric"
  awk -v value="$CAPE_SUBMIT_COMMAND_TIMEOUT" 'BEGIN {exit !(value > 0)}' || \
    die "--command-timeout-seconds must be greater than zero"
fi
if [ -n "$CAPE_SUBMIT_RUNTIME_ADAPTER" ]; then
  case "$CAPE_SUBMIT_RUNTIME_ADAPTER" in
    apptainer|bubblewrap|podman|local-process) ;;
    *) die "--runtime-adapter must be apptainer, bubblewrap, podman, or local-process" ;;
  esac
fi
case "$CAPE_SUBMIT_ISOLATION_POLICY" in
  default|none|host-stateless|host-stateful|image-stateless|image-stateful) ;;
  *) die "--isolation-policy must be default, none, host-stateless, host-stateful, image-stateless, or image-stateful" ;;
esac
for port in "${port_values[@]}"; do
  case "$port" in
    ''|*[!0-9]*) die "--port must be an integer: $port" ;;
  esac
  [ "$port" -ge 1 ] && [ "$port" -le 65535 ] || \
    die "--port must be between 1 and 65535: $port"
done
for entry in "${env_values[@]}"; do
  case "$entry" in
    ?*=*) ;;
    *) die "--env must be KEY=VALUE: $entry" ;;
  esac
done

case "$mimic_kind" in
  A|a|gemm|gemm-silu|gemm_silu)
    mimic_kind=A
    ;;
  B|b|add|cutedsl|cutedsl-add|cutedsl_add)
    mimic_kind=B
    ;;
  '')
    ;;
  *)
    die "--mimic must be A or B"
    ;;
esac

if [ -n "$mimic_kind" ]; then
  MIMIC_LOCAL_DIR="${CAPE_SUBMIT_MIMIC_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
  case "$MIMIC_LOCAL_DIR" in
    /*) ;;
    *) MIMIC_LOCAL_DIR="$PWD/${MIMIC_LOCAL_DIR#./}" ;;
  esac
  [ -r "$MIMIC_LOCAL_DIR/gemm_silu.py" ] || \
    die "mimic source is unavailable: $MIMIC_LOCAL_DIR/gemm_silu.py"
  [ -r "$MIMIC_LOCAL_DIR/cutedsl_add.py" ] || \
    die "mimic source is unavailable: $MIMIC_LOCAL_DIR/cutedsl_add.py"

  if [ -z "$CAPE_SUBMIT_USER_ROOT" ]; then
    [ -n "$CAPE_SUBMIT_BUNDLE" ] || \
      die "mimic mode needs --user-root or role-local CAPE_SUBMIT_BUNDLE"
    CAPE_SUBMIT_USER_ROOT="$CAPE_SUBMIT_BUNDLE/users/$CAPE_SUBMIT_USER"
  fi
  case "$CAPE_SUBMIT_USER_ROOT" in
    /*) ;;
    *) CAPE_SUBMIT_USER_ROOT="$PWD/${CAPE_SUBMIT_USER_ROOT#./}" ;;
  esac
  for path in "$CAPE_SUBMIT_USER_ROOT" "$MIMIC_LOCAL_DIR"; do
    case "$path" in
      *:*) die "mimic host paths may not contain a colon: $path" ;;
    esac
  done

  for bind_spec in "${bind_values[@]}"; do
    bind_tail="${bind_spec#*:}"
    [ "$bind_tail" != "$bind_spec" ] || die "invalid --bind value: $bind_spec"
    bind_target="${bind_tail%%:*}"
    case "$bind_target" in
      /workspace|/output|/cape-mimic)
        die "mimic mode already owns bind target $bind_target"
        ;;
    esac
  done

  [ -n "$CAPE_SUBMIT_SESSION_ID" ] || CAPE_SUBMIT_SESSION_ID="$CAPE_SUBMIT_USER-task"
  user_workspace="$CAPE_SUBMIT_USER_ROOT/workspace"
  user_output="$CAPE_SUBMIT_USER_ROOT/output"
  umask 077
  mkdir -p "$user_workspace" "$user_output"
  bind_values+=(
    "$user_workspace:/workspace:rw"
    "$user_output:/output:rw"
    "$MIMIC_LOCAL_DIR:/cape-mimic:ro"
  )
  env_values+=(
    "CAPE_RUN_TAG=${CAPE_SUBMIT_RUN_TAG:-$CAPE_SUBMIT_USER@${RUN_ID:-manual}}"
    "CAPE_OUTPUT=/output"
  )
  [ -n "$CAPE_SUBMIT_CWD" ] || CAPE_SUBMIT_CWD=/workspace
  [ -n "$CAPE_SUBMIT_WORKSPACE_TARGET" ] || CAPE_SUBMIT_WORKSPACE_TARGET=/cape-workspace
  [ -n "$CAPE_SUBMIT_IMAGE" ] || [ -n "$CAPE_SUBMIT_PROFILE" ] || \
    CAPE_SUBMIT_IMAGE=docker://nvcr.io/nvidia/pytorch:26.06-py3
  [ "$CAPE_SUBMIT_LABEL" != workload ] || CAPE_SUBMIT_LABEL="$mimic_kind"

  if [ "$mimic_kind" = A ]; then
    workload_command=(python /cape-mimic/gemm_silu.py)
  else
    workload_command=(
      bash -lc
      'set -e; python -c "import cutlass" >/dev/null 2>&1 || pip install --quiet nvidia-cutlass-dsl; exec python /cape-mimic/cutedsl_add.py'
    )
  fi
fi

[ -z "$CAPE_SUBMIT_IMAGE" ] || [ -z "$CAPE_SUBMIT_PROFILE" ] || \
  die "--image and --profile are mutually exclusive"

if [ -n "$CAPE_SUBMIT_RUNTIME_ADAPTER" ]; then
  case "$CAPE_SUBMIT_RUNTIME_ADAPTER:$CAPE_SUBMIT_ISOLATION_POLICY" in
    apptainer:default|apptainer:image-stateless|apptainer:image-stateful|\
    bubblewrap:default|bubblewrap:host-stateless|bubblewrap:host-stateful|\
    podman:default|podman:image-stateless|podman:image-stateful|\
    local-process:default|local-process:none)
      ;;
    *)
      die "$CAPE_SUBMIT_RUNTIME_ADAPTER cannot satisfy isolation policy $CAPE_SUBMIT_ISOLATION_POLICY"
      ;;
  esac
fi

if [ "$no_token" = 1 ]; then
  CAPE_SUBMIT_TOKEN=""
elif [ "$token_override_set" = 1 ]; then
  CAPE_SUBMIT_TOKEN="$token_override"
elif [ -n "${CAPE_SUBMIT_TOKEN:-}" ]; then
  CAPE_SUBMIT_TOKEN="$CAPE_SUBMIT_TOKEN"
elif [ -n "${CAPE_TOKEN:-}" ]; then
  CAPE_SUBMIT_TOKEN="$CAPE_TOKEN"
else
  CAPE_SUBMIT_TOKEN=""
  if [ -n "$CAPE_SUBMIT_USER_TOKENS" ]; then
    if [ -r "$CAPE_SUBMIT_USER_TOKENS" ]; then
      CAPE_SUBMIT_TOKEN="$(awk -F, -v user="$CAPE_SUBMIT_USER" '
$1 == user {print $2; found=1; exit}
END {if (!found) exit 1}
' "$CAPE_SUBMIT_USER_TOKENS" 2>/dev/null || true)"
      CAPE_SUBMIT_TOKEN="${CAPE_SUBMIT_TOKEN%$'\r'}"
      if [ -z "$CAPE_SUBMIT_TOKEN" ] && [ "$user_tokens_override_set" = 1 ]; then
        die "no token for $CAPE_SUBMIT_USER in $CAPE_SUBMIT_USER_TOKENS"
      fi
    elif [ "$user_tokens_override_set" = 1 ]; then
      die "user-token CSV is not readable: $CAPE_SUBMIT_USER_TOKENS"
    fi
  fi
  if [ -z "$CAPE_SUBMIT_TOKEN" ] && [ -n "$CAPE_SUBMIT_USER_TOKENS_INLINE" ]; then
    while IFS= read -r token_record; do
      token_user="${token_record%%,*}"
      [ "$token_user" != "$token_record" ] || continue
      [ "$token_user" = "$CAPE_SUBMIT_USER" ] || continue
      CAPE_SUBMIT_TOKEN="${token_record#*,}"
      CAPE_SUBMIT_TOKEN="${CAPE_SUBMIT_TOKEN%$'\r'}"
      break
    done <<< "$CAPE_SUBMIT_USER_TOKENS_INLINE"
  fi
  if [ -z "$CAPE_SUBMIT_TOKEN" ] && [ -n "${CAPE_ADMIN_TOKEN:-}" ]; then
    CAPE_SUBMIT_TOKEN="$CAPE_ADMIN_TOKEN"
    log "user token unavailable; using the controller admin token for $CAPE_SUBMIT_USER"
  fi
fi

case "$CAPE_SUBMIT_VENV_ROOT" in
  /*) ;;
  *) CAPE_SUBMIT_VENV_ROOT="$PWD/${CAPE_SUBMIT_VENV_ROOT#./}" ;;
esac
case "$CAPE_SUBMIT_VENV" in
  /*) ;;
  *) CAPE_SUBMIT_VENV="$PWD/${CAPE_SUBMIT_VENV#./}" ;;
esac

resolve_executable() {
  local candidate="$1"
  local label="$2"
  if [[ "$candidate" == */* ]]; then
    candidate="${candidate/#\~/$HOME}"
    [ -x "$candidate" ] || die "$label is not executable: $candidate"
    printf '%s\n' "$candidate"
  else
    command -v "$candidate" 2>/dev/null || die "$label was not found: $candidate"
  fi
}

ensure_cape_client() {
  local evidence_file=""
  [ -z "$CAPE_SUBMIT_BUNDLE" ] || \
    evidence_file="$CAPE_SUBMIT_BUNDLE/submit-client-implementation.txt"
  if [ "$CAPE_SUBMIT_BIN_SELECTION" != pycape ]; then
    CAPE_SUBMIT_BIN="$(cape_resolve_executable "$CAPE_SUBMIT_BIN_SELECTION" CAPE_SUBMIT_BIN)" || exit 2
    cape_inspect_executable "$CAPE_SUBMIT_BIN_SELECTION" "$CAPE_SUBMIT_BIN" || exit 2
    cape_record_executable submit "$CAPE_SUBMIT_BIN" "$evidence_file"
    return
  fi

  CAPE_SUBMIT_BIN="$CAPE_SUBMIT_VENV/bin/pycape"
  VENV_PYTHON="$CAPE_SUBMIT_VENV/bin/python"
  [ -f "$CAPE_PROJECT/pyproject.toml" ] || \
    die "CAPE source is required to install the client: $CAPE_PROJECT/pyproject.toml"

  mkdir -p "$CAPE_SUBMIT_VENV_ROOT"
  local install_lock="$CAPE_SUBMIT_VENV_ROOT/.install-submit-${CAPE_SUBMIT_VENV_HOST_ID}-$(uname -m)-python3.12.lock"
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$install_lock"
    flock 9
  fi

  local venv_ready=0
  if [ -x "$VENV_PYTHON" ]; then
    local venv_version
    venv_version="$($VENV_PYTHON -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
    if [ "$venv_version" = 3.12 ] && \
       "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
      venv_ready=1
    fi
  fi

  if [ "$venv_ready" != 1 ]; then
    local -a python312_cmd=()
    local python312_description=""
    if [ -n "$CAPE_SUBMIT_PYTHON_312" ]; then
      local resolved_python
      resolved_python="$(resolve_executable "$CAPE_SUBMIT_PYTHON_312" CAPE_SUBMIT_PYTHON_312)"
      python312_cmd=("$resolved_python")
      python312_description="$resolved_python"
      python312_can_bootstrap_venv "${python312_cmd[@]}" || \
        die "$CAPE_SUBMIT_PYTHON_312 cannot bootstrap a pip-enabled venv"
    elif command -v python3.12 >/dev/null 2>&1 && \
         python312_can_bootstrap_venv python3.12; then
      python312_cmd=("$(command -v python3.12)")
      python312_description="${python312_cmd[0]}"
    else
      if [ -z "$CAPE_SUBMIT_PIXI_BIN" ]; then
        if command -v pixi >/dev/null 2>&1; then
          CAPE_SUBMIT_PIXI_BIN="$(command -v pixi)"
        fi
      fi
      [ -n "$CAPE_SUBMIT_PIXI_BIN" ] || \
        die "a pip-enabled Python 3.12 bootstrap and Pixi were not found; set CAPE_SUBMIT_PYTHON_312 or CAPE_SUBMIT_PIXI_BIN"
      CAPE_SUBMIT_PIXI_BIN="$(resolve_executable "$CAPE_SUBMIT_PIXI_BIN" CAPE_SUBMIT_PIXI_BIN)"
      python312_cmd=("$CAPE_SUBMIT_PIXI_BIN" exec --no-progress --spec "python=3.12" python)
      python312_description="Pixi ($CAPE_SUBMIT_PIXI_BIN)"
      log "usable Python 3.12 venv bootstrap not found; bootstrapping it with $python312_description"
    fi

    local bootstrap_version
    bootstrap_version="$("${python312_cmd[@]}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" || \
      die "failed to start Python through $python312_description"
    [ "$bootstrap_version" = 3.12 ] || \
      die "Python bootstrap must provide 3.12, got $bootstrap_version"
    local python312_bootstrap_mode
    python312_bootstrap_mode="$(python312_bootstrap_mode "${python312_cmd[@]}")" || \
      die "$python312_description cannot bootstrap a pip-enabled venv"

    if [ -d "$CAPE_SUBMIT_VENV" ] && [ ! -f "$CAPE_SUBMIT_VENV/pyvenv.cfg" ]; then
      die "CAPE_SUBMIT_VENV exists but is not a Python virtual environment: $CAPE_SUBMIT_VENV"
    fi
    mkdir -p "$(dirname "$CAPE_SUBMIT_VENV")"
    if [ -f "$CAPE_SUBMIT_VENV/pyvenv.cfg" ]; then
      log "rebuilding incompatible Python 3.12 venv: $CAPE_SUBMIT_VENV"
      create_python312_venv "$python312_bootstrap_mode" "$CAPE_SUBMIT_VENV" 1 \
        "${python312_cmd[@]}" >&2 || \
        die "failed to rebuild the submit-client Python 3.12 venv"
    else
      log "creating Python 3.12 venv: $CAPE_SUBMIT_VENV"
      create_python312_venv "$python312_bootstrap_mode" "$CAPE_SUBMIT_VENV" 0 \
        "${python312_cmd[@]}" >&2 || \
        die "failed to create the submit-client Python 3.12 venv"
    fi
  fi

  [ -x "$VENV_PYTHON" ] || die "submit-client venv Python is missing: $VENV_PYTHON"
  [ "$($VENV_PYTHON -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = 3.12 ] || \
    die "submit-client venv does not use Python 3.12"

  local install_marker="$CAPE_SUBMIT_VENV/.cape-install-fingerprint"
  local install_fingerprint
  install_fingerprint="$($VENV_PYTHON -c '
import hashlib, os, sys
project, metadata = sys.argv[1:]
digest = hashlib.sha256()
digest.update(os.path.realpath(project).encode())
with open(metadata, "rb") as stream:
    digest.update(stream.read())
print(digest.hexdigest())
' "$CAPE_PROJECT" "$CAPE_PROJECT/pyproject.toml")"

  local install_ready=0
  if [ "$CAPE_FORCE_INSTALL" = 0 ] && [ -x "$CAPE_SUBMIT_BIN" ] && [ -r "$install_marker" ]; then
    local recorded_fingerprint
    recorded_fingerprint="$(sed -n '1p' "$install_marker")"
    if [ "$recorded_fingerprint" = "$install_fingerprint" ] && \
       "$VENV_PYTHON" -c 'import aiohttp, pycape' >/dev/null 2>&1; then
      install_ready=1
    fi
  fi

  if [ "$install_ready" != 1 ]; then
    log "installing CAPE from $CAPE_PROJECT into $CAPE_SUBMIT_VENV"
    "$VENV_PYTHON" -m pip install \
      --disable-pip-version-check \
      --editable "$CAPE_PROJECT" >&2 || die "CAPE installation failed"
    "$VENV_PYTHON" -c \
      'import sys, aiohttp, pycape; assert sys.version_info[:2] == (3, 12)' || \
      die "installed CAPE or aiohttp could not be imported"
    [ -x "$CAPE_SUBMIT_BIN" ] || \
      die "CAPE executable was not installed: $CAPE_SUBMIT_BIN"
    local marker_tmp="$install_marker.tmp.$$"
    printf '%s\n' "$install_fingerprint" > "$marker_tmp"
    mv -- "$marker_tmp" "$install_marker"
  fi

  if command -v flock >/dev/null 2>&1; then
    flock -u 9
    exec 9>&-
  fi
  CAPE_SUBMIT_BIN="$(cape_resolve_executable "$CAPE_SUBMIT_BIN" CAPE_SUBMIT_BIN)" || exit 2
  cape_inspect_executable "$CAPE_SUBMIT_BIN_SELECTION" "$CAPE_SUBMIT_BIN" || exit 2
  cape_record_executable submit "$CAPE_SUBMIT_BIN" "$evidence_file"
}

record_file=""
prepare_record_file() {
  [ -n "$CAPE_SUBMIT_RECORD" ] || return 0
  record_file="$CAPE_SUBMIT_RECORD"
  case "$record_file" in
    /*) ;;
    *) record_file="$PWD/${record_file#./}" ;;
  esac
  for value in "$CAPE_SUBMIT_USER" "$CAPE_SUBMIT_LABEL" "$CAPE_SUBMIT_SESSION_ID"; do
    case "$value" in
      *','*|*$'\n'*|*$'\r'*) die "CSV identity values may not contain commas or newlines" ;;
    esac
  done
  umask 077
  mkdir -p "$(dirname "$record_file")"
  if command -v flock >/dev/null 2>&1; then
    exec 8>>"$record_file.lock"
    flock 8
  fi
  local expected_header='uid,request,jobid,t_start,t_return,latency,session_id'
  if [ ! -s "$record_file" ]; then
    printf '%s\n' "$expected_header" > "$record_file"
  else
    local existing_header
    existing_header="$(sed -n '1p' "$record_file")"
    [ "$existing_header" = "$expected_header" ] || \
      die "submission record has an unexpected header: $record_file"
  fi
  if command -v flock >/dev/null 2>&1; then
    flock -u 8
    exec 8>&-
  fi
}

append_record() {
  [ -n "$record_file" ] || return 0
  local request_id="$1"
  local t_start="$2"
  local t_return="$3"
  local latency="$4"
  local append_rc=0
  local locked=0
  if command -v flock >/dev/null 2>&1; then
    exec 8>>"$record_file.lock" || return $?
    if ! flock 8; then
      exec 8>&-
      return 1
    fi
    locked=1
  fi
  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "$CAPE_SUBMIT_USER" "$CAPE_SUBMIT_LABEL" "$request_id" \
    "$t_start" "$t_return" "$latency" "$CAPE_SUBMIT_SESSION_ID" \
    >> "$record_file" || append_rc=$?
  if [ "$locked" = 1 ]; then
    flock -u 8
    exec 8>&-
  fi
  return "$append_rc"
}

if [ "$resolve_client" = 1 ]; then
  ensure_cape_client
  printf '%s\n' "$CAPE_SUBMIT_BIN"
  exit 0
fi

prepare_record_file
ensure_cape_client
submit_phase_client_ready="$(date +%s.%N)"

run_args=(
  run
  --controller-url "$CAPE_SUBMIT_URL"
  --pool "$CAPE_SUBMIT_POOL"
  --user "$CAPE_SUBMIT_USER"
  --gpus "$CAPE_SUBMIT_GPUS"
  --cpu-cores "$CAPE_SUBMIT_CPU_CORES"
  --memory-gb "$CAPE_SUBMIT_MEMORY_GB"
  --isolation-policy "$CAPE_SUBMIT_ISOLATION_POLICY"
)
[ -z "$CAPE_SUBMIT_SESSION_ID" ] || run_args+=(--session-id "$CAPE_SUBMIT_SESSION_ID")
if [ -n "$CAPE_MANUAL_TOKEN_FILE" ]; then
  run_args+=(--token-file "$CAPE_MANUAL_TOKEN_FILE")
else
  [ -z "$CAPE_SUBMIT_TOKEN" ] || run_args+=(--token "$CAPE_SUBMIT_TOKEN")
fi
[ -z "$CAPE_SUBMIT_IMAGE" ] || run_args+=(--image "$CAPE_SUBMIT_IMAGE")
[ -z "$CAPE_SUBMIT_PROFILE" ] || run_args+=(--profile "$CAPE_SUBMIT_PROFILE")
[ -z "$CAPE_SUBMIT_RUNTIME_ADAPTER" ] || \
  run_args+=(--runtime-adapter "$CAPE_SUBMIT_RUNTIME_ADAPTER")
[ -z "$CAPE_SUBMIT_MAX_DURATION" ] || \
  run_args+=(--max-duration-seconds "$CAPE_SUBMIT_MAX_DURATION")
[ -z "$CAPE_SUBMIT_COMMAND_TIMEOUT" ] || \
  run_args+=(--command-timeout-seconds "$CAPE_SUBMIT_COMMAND_TIMEOUT")
[ "$hard_gpu_enforcement" != 1 ] || run_args+=(--hard-gpu-enforcement)
[ -z "$CAPE_SUBMIT_LOCALITY" ] || run_args+=(--locality "$CAPE_SUBMIT_LOCALITY")
[ -z "$CAPE_SUBMIT_CWD" ] || run_args+=(--cwd "$CAPE_SUBMIT_CWD")
[ -z "$CAPE_SUBMIT_WORKSPACE_ID" ] || run_args+=(--workspace-id "$CAPE_SUBMIT_WORKSPACE_ID")
[ -z "$CAPE_SUBMIT_WORKSPACE_TARGET" ] || run_args+=(--workspace-target "$CAPE_SUBMIT_WORKSPACE_TARGET")
[ -z "$CAPE_SUBMIT_LOG_PATH" ] || run_args+=(--log-path "$CAPE_SUBMIT_LOG_PATH")
[ "$suspend_on_failure" != 1 ] || \
  run_args+=(--suspend-on-failure)
for port in "${port_values[@]}"; do run_args+=(--port "$port"); done
for bind_spec in "${bind_values[@]}"; do run_args+=(--bind "$bind_spec"); done
for entry in "${env_values[@]}"; do run_args+=(--env "$entry"); done
for artifact in "${artifact_values[@]}"; do run_args+=(--artifact-path "$artifact"); done
run_args+=(-- "${workload_command[@]}")

submit_phase_pre_binary="$(date +%s.%N)"
t_start="$(date +%s.%N)"
if request_output="$("$CAPE_SUBMIT_BIN" "${run_args[@]}")"; then
  :
else
  submit_rc=$?
  submit_phase_binary_return="$(date +%s.%N)"
  exit "$submit_rc"
fi
submit_phase_binary_return="$(date +%s.%N)"
t_return="$(date +%s.%N)"
request_id="${request_output//$'\r'/}"
[[ "$request_id" =~ ^req-[A-Za-z0-9._-]+$ ]] || \
  die "unexpected CAPE run output: $request_output"
latency="$(awk -v start="$t_start" -v finish="$t_return" \
  'BEGIN {printf "%.6f", finish-start}')"

if ! append_record "$request_id" "$t_start" "$t_return" "$latency"; then
  log "warning: request was submitted, but evidence could not be appended to $record_file"
fi
log "user=$CAPE_SUBMIT_USER request=$request_id latency=${latency}s"
printf '%s\n' "$request_id"
