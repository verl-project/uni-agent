#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

if [ "$#" -ne 3 ]; then
    echo "usage: $0 SOURCE_DIR BUILD_DIR PREFIX" >&2
    exit 2
fi

source_dir=$1
build_dir=$2
prefix=$3

test -r "$source_dir/meson.build"
test ! -e "$build_dir"
test ! -e "$prefix"

meson setup \
    "$build_dir" \
    "$source_dir" \
    --prefix="$prefix" \
    -Dselinux=disabled \
    -Dman=disabled \
    -Dbash_completion=disabled \
    -Dzsh_completion=disabled \
    -Dtests=false
meson compile -C "$build_dir"
meson install -C "$build_dir"

binary="$prefix/bin/bwrap"
test -x "$binary"
"$binary" --version
file "$binary"
"$binary" --help
