#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
source "$(dirname "$0")/../identity.sh"
credential=${1:?usage: credential_exec.sh TOKEN_FILE COMMAND [ARG...]}
shift
# Never let an inherited admin or test-handoff credential override this identity.
unset CAPE_ADMIN_TOKEN CAPE_NODE_TOKEN CAPE_SUBMIT_TOKEN CAPE_STATUS_TOKEN CAPE_RETIRE_TOKEN
unset CAPE_CONTROLLER_ENV CAPE_USER_TOKENS_INLINE CAPE_SUBMIT_USER_TOKENS_INLINE
CAPE_TOKEN="$(cape_read_secret "$credential")"
export CAPE_TOKEN
exec "$@"
