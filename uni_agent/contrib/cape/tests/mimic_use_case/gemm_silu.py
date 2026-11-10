#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CAPE use-case workload A: BF16 GEMM (8192^3) followed by SiLU, on CUDA.

This script is reused unchanged by every user/request in the CAPE use-case mimic
It is deliberately self-contained and
imports torch lazily so the file still byte-compiles on hosts without torch.

Environment (all optional):
  CAPE_OUTPUT   directory to write the result file (default: current dir)
  CAPE_M/N/K    GEMM problem size            (default: 8192 each)
  CAPE_SEED     RNG seed                     (default: 0)

Output contract: writes <CAPE_OUTPUT>/gemm_silu_result.txt containing status,
problem size, device, the leased CUDA_VISIBLE_DEVICES (for the hardware-isolation
probe), elapsed time, and a checksum. Exit code 0 on PASS, 1 on FAIL, 2 if no GPU.
"""
import os
import sys
import time


def main():
    import torch

    if not torch.cuda.is_available():
        sys.stderr.write("CUDA not available\n")
        return 2

    M = int(os.environ.get("CAPE_M", "8192"))
    N = int(os.environ.get("CAPE_N", "8192"))
    K = int(os.environ.get("CAPE_K", "8192"))
    seed = int(os.environ.get("CAPE_SEED", "0"))
    out_dir = os.environ.get("CAPE_OUTPUT", os.getcwd())
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda")
    torch.manual_seed(seed)
    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(K, N, dtype=torch.bfloat16, device=device)

    torch.cuda.synchronize()
    start = time.time()
    c = torch.nn.functional.silu(a @ b)  # GEMM (a @ b) followed by SiLU
    torch.cuda.synchronize()
    elapsed = time.time() - start

    c_fp32 = c.float()
    checksum = c_fp32.sum().item()
    mean = c_fp32.mean().item()
    finite = bool(torch.isfinite(c_fp32).all().item())
    status = "PASS" if finite else "FAIL"

    device_name = torch.cuda.get_device_name(device)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    run_tag = os.environ.get("CAPE_RUN_TAG", "")  # per-user marker for the isolation test

    out_path = os.path.join(out_dir, "gemm_silu_result.txt")
    with open(out_path, "w") as fh:
        fh.write("workload=gemm_silu\n")
        fh.write("status=%s\n" % status)
        fh.write("run_tag=%s\n" % run_tag)
        fh.write("M=%d N=%d K=%d dtype=bfloat16\n" % (M, N, K))
        fh.write("device=%s\n" % device_name)
        fh.write("cuda_visible_devices=%s\n" % cuda_visible)
        fh.write("elapsed_s=%.6f\n" % elapsed)
        fh.write("checksum=%.6f\n" % checksum)
        fh.write("mean=%.6f\n" % mean)

    sys.stdout.write(
        "gemm_silu %s run_tag=%s shape=%dx%dx%d device=%s CUDA_VISIBLE_DEVICES=%s "
        "elapsed=%.4fs checksum=%.4f\n"
        % (status, run_tag, M, N, K, device_name, cuda_visible, elapsed, checksum)
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
