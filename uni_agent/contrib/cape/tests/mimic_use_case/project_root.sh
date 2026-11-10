#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cape_project_root() {
  local directory="${1:?starting directory is required}"
  while [[ $directory != / ]]; do
    if [[ -f $directory/pycape/pyproject.toml &&
          -f $directory/pycape/__init__.py &&
          -f $directory/rscape/Cargo.toml &&
          -d $directory/tests ]]; then
      printf '%s\n' "$directory"
      return 0
    fi
    directory="$(dirname "$directory")"
  done
  printf 'could not locate the CAPE project root\n' >&2
  return 1
}

pycape_project_root() {
  local repository
  repository="$(cape_project_root "${1:?starting directory is required}")" || return
  printf '%s/pycape\n' "$repository"
}

cape_publish_role_pid() {
  local pid_file=${CAPE_ROLE_PID_FILE:-}
  local label=${CAPE_ROLE_LABEL:-cape-role}
  [ -n "$pid_file" ] || return 0
  local pid=${BASHPID:-$$}
  local execution_session_id
  execution_session_id="$(python3 -c 'import os; print(os.getsid(0))')" || return
  [ "$execution_session_id" = "$pid" ] || {
    printf 'CAPE role must lead its own session before pid publication: pid=%s sid=%s\n' \
      "$pid" "$execution_session_id" >&2
    return 1
  }
  local temporary="$pid_file.tmp.$pid"
  mkdir -p "$(dirname "$pid_file")"
  printf '%s %s %s\n' "$pid" "$execution_session_id" "$label" > "$temporary"
  mv "$temporary" "$pid_file"
}

cape_resolve_executable() {
  local candidate="${1:?executable selection is required}"
  local label="${2:-CAPE executable}"
  if [[ $candidate == */* ]]; then
    candidate="${candidate/#\~/$HOME}"
    if [[ ! -x $candidate ]]; then
      printf '%s is not executable: %s\n' "$label" "$candidate" >&2
      return 2
    fi
    printf '%s\n' "$candidate"
  else
    command -v "$candidate" 2>/dev/null || {
      printf '%s was not found: %s\n' "$label" "$candidate" >&2
      return 2
    }
  fi
}

cape_inspect_executable() {
  local selection="${1:?executable selection is required}"
  local executable="${2:?resolved executable is required}"
  local name version_line
  name="$(basename "$executable")"
  case "$name" in
    rscape|pycape) CAPE_CLIENT_IMPLEMENTATION="$name" ;;
    *) CAPE_CLIENT_IMPLEMENTATION="$name" ;;
  esac
  version_line=""
  if [[ $selection == rscape || $selection == pycape ]]; then
    version_line="$({ "$executable" --version 2>&1 || true; } | sed -n '1p')"
    if [[ -z $version_line ]]; then
      printf '%s version probe failed: %s --version\n' "$selection" "$executable" >&2
      return 2
    fi
  fi
  if [[ $version_line == "$CAPE_CLIENT_IMPLEMENTATION "* ]]; then
    CAPE_CLIENT_VERSION="${version_line#"$CAPE_CLIENT_IMPLEMENTATION "}"
  else
    CAPE_CLIENT_VERSION="${version_line:-unreported}"
  fi
}

cape_record_executable() {
  local role="${1:?role is required}"
  local executable="${2:?resolved executable is required}"
  local evidence_file="${3:-}"
  printf '[%s] cape_implementation=%s cape_version=%s cape_bin=%s\n' \
    "$role" "$CAPE_CLIENT_IMPLEMENTATION" "$CAPE_CLIENT_VERSION" "$executable" >&2
  if [[ -n $evidence_file ]]; then
    mkdir -p "$(dirname "$evidence_file")"
    {
      printf 'implementation=%s\n' "$CAPE_CLIENT_IMPLEMENTATION"
      printf 'version=%s\n' "$CAPE_CLIENT_VERSION"
      printf 'executable=%s\n' "$executable"
    } > "$evidence_file"
  fi
}
