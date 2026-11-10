#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Shared explicit credential input for manual role wrappers.
source "$(dirname "${BASH_SOURCE[0]}")/../identity.sh"

cape_manual_credentials() {
  CAPE_MANUAL_ARGS=()
  CAPE_MANUAL_TOKEN_FILE=""
  local file="${CAPE_TOKEN_FILE:-}" plain=0
  while (($#)); do
    case "$1" in
      --) CAPE_MANUAL_ARGS+=("$@"); break ;;
      --token-file)
        [[ $# -ge 2 && -n $2 ]] || { echo '--token-file requires a path' >&2; return 2; }
        file=$2; shift 2 ;;
      --token-file=*) file=${1#*=}; [[ -n $file ]] || return 2; shift ;;
      --token|--token=*|--no-token) plain=1; CAPE_MANUAL_ARGS+=("$1"); shift ;;
      *) CAPE_MANUAL_ARGS+=("$1"); shift ;;
    esac
  done
  if [[ -n $file ]]; then
    [[ $plain == 0 ]] || { echo 'choose --token-file or --token/--no-token, not both' >&2; return 2; }
    CAPE_TOKEN=$(cape_read_secret "$file") || return
    CAPE_MANUAL_TOKEN_FILE=$(realpath "$file") || return
    # Wrappers may inspect the value, but the real CLI receives --token-file.
    # Clap treats exported CAPE_TOKEN as --token and rejects both sources.
    export -n CAPE_TOKEN
    unset CAPE_ADMIN_TOKEN CAPE_NODE_TOKEN CAPE_SUBMIT_TOKEN CAPE_STATUS_TOKEN CAPE_RETIRE_TOKEN
    unset CAPE_CONTROLLER_ENV CAPE_USER_TOKENS_INLINE CAPE_SUBMIT_USER_TOKENS_INLINE
  fi
}
