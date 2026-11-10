#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

compiler=$1
shift
static_binary=0
previous=
for argument in "$@"; do
  if [ "$previous" = "--crate-name" ] && [ "$argument" = "rscape" ]; then
    static_binary=1
  fi
  previous=$argument
done

if [ "$static_binary" -eq 1 ]; then
  exec "$compiler" "$@" -C target-feature=+crt-static
fi
exec "$compiler" "$@"
