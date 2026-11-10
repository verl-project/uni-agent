#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/credentials.sh"
cape_manual_credentials "$@"
set -- "${CAPE_MANUAL_ARGS[@]}"

pool=${1:?usage: wait_pool_ready.sh --token-file FILE POOL [TIMEOUT_SECONDS] (set CAPE_URL)}
: "${CAPE_URL:?Set CAPE_URL}"
timeout=${2:-${CAPE_READY_TIMEOUT:-600}}
[[ $timeout =~ ^[0-9]+$ && $timeout -gt 0 ]] || {
  echo "wait_pool_ready.sh: timeout must be a positive integer" >&2
  exit 2
}

deadline=$((SECONDS + timeout))
last=
auth_args=()
[ -z "$CAPE_MANUAL_TOKEN_FILE" ] || auth_args=(--token-file "$CAPE_MANUAL_TOKEN_FILE")
while ((SECONDS < deadline)); do
  if output=$(bash "$(dirname "$0")/status.sh" "${auth_args[@]}" --controller-url "$CAPE_URL" --pool "$pool" utilization 2>/dev/null); then
    last=$output
    if python3 -c 'import json,sys; value=json.load(sys.stdin); raise SystemExit(0 if value.get("active_nodes",0)>0 and value.get("registering_nodes",0)==0 else 1)' <<< "$output"; then
      printf '%s\n' "$output"
      exit 0
    fi
  fi
  sleep .1
done
echo "wait_pool_ready.sh: pool did not become ACTIVE within ${timeout}s" >&2
[[ -z $last ]] || printf 'last utilization: %s\n' "$last" >&2
exit 1
