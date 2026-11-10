#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Shared persistent-CLI bootstrap for the reproduction scripts (source, do not execute).

cape_read_secret() {
  python3 - "$1" <<'PY'
import json, os, stat, sys
path = sys.argv[1]
info = os.lstat(path)
if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
    raise SystemExit("credential file must be a private owner-only regular file")
with open(path) as stream:
    print(json.load(stream)["secret"])
PY
}

cape_init_instance() {
  local state="$1" credential="$2"
  if [ ! -e "$state" ]; then
    "$CAPE_BIN" controller init --state-dir "$state" --user cape-admin --token-out "$credential" >/dev/null
  fi
  [ -f "$credential" ] || { echo "missing existing instance admin credential: $credential" >&2; return 2; }
}

cape_create_user() {
  local name="$1" credential="$2"
  if [ ! -e "$credential" ]; then
    env -u CAPE_TOKEN -u CAPE_ADMIN_TOKEN "$CAPE_BIN" user token create \
      --controller-url "$CAPE_URL" --user "$name" --token-out "$credential" >/dev/null
  else
    "$CAPE_BIN" user token resume --token-out "$credential" >/dev/null
  fi
}

cape_ensure_pool() {
  local pool="$1" credential="$2" mode="${3:-exclusive}" slots="${4:-1}" placement="${5:-pack}"
  local inventory
  inventory="$("$CAPE_BIN" user pools --controller-url "$CAPE_URL" --token-file "$credential")" || return
  if ! printf '%s' "$inventory" | python3 -c 'import json,sys; raise SystemExit(0 if any(p["name"] == sys.argv[1] for p in json.load(sys.stdin)) else 1)' "$pool"; then
    "$CAPE_BIN" admin pool create "$pool" --controller-url "$CAPE_URL" --token-file "$credential" \
      --pool-mode "$mode" --gpu-slots "$slots" --placement "$placement" >/dev/null
  fi
}
