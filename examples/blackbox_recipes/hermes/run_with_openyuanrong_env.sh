#!/usr/bin/env bash
# Load the protected OpenYuanRong environment inside a Ray job driver.
# Values never appear in the Ray job command or runtime-env JSON.

set -euo pipefail

ENV_FILE="${OPENYUANRONG_ENV_FILE:-/home/zxh/.config/uni-agent/openyuanrong.env}"
if [[ ! -r "${ENV_FILE}" ]]; then
    echo "OpenYuanRong environment file is not readable: ${ENV_FILE}" >&2
    exit 2
fi

set -a
# The file is a protected, operator-owned shell environment.  Keep only the
# OpenYuanRong settings needed by the provider; unrelated legacy AKERNEL
# credentials must not leak into the Ray job.
. "${ENV_FILE}"
set +a
unset AKERNEL_SERVER_ADDRESS AKERNEL_TOKEN

if [[ -z "${OPENYUANRONG_SERVER_ADDRESS:-}" || -z "${OPENYUANRONG_TOKEN:-}" ]]; then
    echo "OpenYuanRong environment file did not define the required credentials" >&2
    exit 2
fi

exec "$@"
