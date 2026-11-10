#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
source "$(dirname "$0")/../identity.sh"
unset CAPE_CONTROLLER_ENV CAPE_TOKEN CAPE_ADMIN_TOKEN
: "${CAPE_URL:?Set CAPE_URL to the scheduler URL}"
: "${CAPE_USER:?Set CAPE_USER to your regular identity}"
: "${CAPE_USER_TOKEN_FILE:?Set CAPE_USER_TOKEN_FILE to a private credential path}"
umask 077
mkdir -p "$(dirname "$CAPE_USER_TOKEN_FILE")"
CAPE_BIN=$(bash "$(dirname "$0")/submit.sh" --resolve-client)
cape_create_user "$CAPE_USER" "$CAPE_USER_TOKEN_FILE"
printf '[user] credential saved in %s (secret not printed)\n' "$CAPE_USER_TOKEN_FILE"
