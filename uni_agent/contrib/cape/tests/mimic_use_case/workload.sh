#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# workload.sh — the per-user workload, run INSIDE the leased container.
#
# It is the command submitted through the selected CAPE executable.
# Responsibilities: install any required packages, then launch the real mimic workload and write
# the per-user output (with CAPE_RUN_TAG, for the §12 R2 isolation check).
#
#   kind     : A | gemm | gemm_silu   -> PyTorch BF16 silu(A@B), 8192^3   (torch ships in the image)
#              B | add | cutedsl       -> CUTLASS CuTe DSL BF16 element-wise add (pip-installs the DSL)
#   run_tag  : per-user unique marker (e.g. "<uid>-<nonce>"); written into the output file
#   out_dir  : in-container output dir (the per-user bind target, e.g. /output)
set -euo pipefail

KIND="${1:?usage: workload.sh <A|B> <run_tag> <out_dir>}"
RUN_TAG="${2:-}"
OUT_DIR="${3:-/output}"

HERE="$(cd "$(dirname "$0")" && pwd)"   # the bound mimic dir inside the container
export CAPE_RUN_TAG="$RUN_TAG"
export CAPE_OUTPUT="$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "[workload] kind=$KIND run_tag=$RUN_TAG out_dir=$OUT_DIR CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

case "$KIND" in
  A|gemm|gemm_silu)
    # torch is already present in nvcr.io/nvidia/pytorch:* — no install needed.
    python "$HERE/gemm_silu.py"
    ;;
  B|add|cutedsl|cutedsl_add)
    # Install the CuTe DSL once; on a reused warm session the second command finds it already present.
    python -c 'import cutlass' >/dev/null 2>&1 || pip install --quiet nvidia-cutlass-dsl
    python "$HERE/cutedsl_add.py"
    ;;
  *)
    echo "[workload] unknown kind: $KIND" >&2
    exit 2
    ;;
esac
