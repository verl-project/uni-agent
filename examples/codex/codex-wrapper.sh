#!/bin/sh
set -eu
TOOL_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
export CODEX_MANAGED_PACKAGE_ROOT="${CODEX_MANAGED_PACKAGE_ROOT:-${TOOL_ROOT}}"
exec "${TOOL_ROOT}/vendor/x86_64-unknown-linux-musl/bin/codex" "$@"
