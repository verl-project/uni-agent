# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# This file has been modified from the original CUTLASS example: the kernel and
# JIT launch path were retained; the tutorial docstring, [DSL INFO] prints, the
# benchmark path, dtype/dynamic-layout permutations, and argparse were removed.
"""CAPE mimic use-case workload B: BF16 element-wise add via the CUTLASS CuTe DSL.

Merged + trimmed from the upstream example (kernel + jit launch kept; tutorial
docstring, [DSL INFO] prints, benchmark path, dtype/dynamic-layout permutations,
and argparse removed). A minimal BF16 runner self-verifies `c == a + b` and writes
the use-case sentinel.

Upstream: NVIDIA/cutlass
  examples/python/CuTeDSL/cute/ampere/kernel/elementwise/elementwise_add.py

Env (optional): CAPE_OUTPUT (sentinel dir), CAPE_M/CAPE_N (default 8192),
CAPE_RUN_TAG (per-user isolation marker). Requires `pip install nvidia-cutlass-dsl`
and a CUDA GPU. Writes <CAPE_OUTPUT>/cutedsl_add_result.txt. Exit 0 PASS / 2 no-GPU.
"""
import os
import sys

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    cC: cute.Tensor,  # coordinate tensor
    shape: cute.Shape,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]  # (TileM,TileN)
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]
    blkCrd = cC[blk_coord]

    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)

    tiled_copy_A = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_B = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_C = cute.make_tiled_copy_tv(copy_atom_store, thr_layout, val_layout)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    thr_copy_C = tiled_copy_C.get_slice(tidx)

    thrA = thr_copy_A.partition_S(blkA)
    thrB = thr_copy_B.partition_S(blkB)
    thrC = thr_copy_C.partition_S(blkC)

    # gmem -> rmem fragments
    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)

    # predicate for tiles whose shape is not a multiple of the tile size (guards OOB)
    thrCrd = thr_copy_C.partition_S(blkCrd)
    frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)
    for i in range(0, cute.size(frgPred), 1):
        frgPred[i] = cute.elem_less(thrCrd[i], shape)

    cute.copy(copy_atom_load, thrA, frgA, pred=frgPred)
    cute.copy(copy_atom_load, thrB, frgB, pred=frgPred)

    result = frgA.load() + frgB.load()
    frgC.store(result)

    cute.copy(copy_atom_store, frgC, thrC, pred=frgPred)


@cute.jit
def elementwise_add(mA, mB, mC, copy_bits: cutlass.Constexpr = 128):
    dtype = mA.element_type
    vector_size = copy_bits // dtype.width

    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM,TileN),(RestM,RestN))
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)

    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)

    elementwise_add_kernel.set_name_prefix("cutlass_dsl_elementwise_add_kernel")
    elementwise_add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def main():
    import torch

    if not torch.cuda.is_available():
        sys.stderr.write("CUDA not available\n")
        return 2

    M = int(os.environ.get("CAPE_M", "8192"))
    N = int(os.environ.get("CAPE_N", "8192"))
    out_dir = os.environ.get("CAPE_OUTPUT", os.getcwd())
    run_tag = os.environ.get("CAPE_RUN_TAG", "")  # per-user marker for the isolation test
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.makedirs(out_dir, exist_ok=True)

    a = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    c = torch.zeros_like(a)

    a_t = from_dlpack(a).mark_layout_dynamic()
    b_t = from_dlpack(b).mark_layout_dynamic()
    c_t = from_dlpack(c).mark_layout_dynamic()

    compiled = cute.compile(elementwise_add, a_t, b_t, c_t)
    compiled(a_t, b_t, c_t)
    torch.testing.assert_close(a + b, c)  # raises on mismatch

    out_path = os.path.join(out_dir, "cutedsl_add_result.txt")
    with open(out_path, "w") as fh:
        fh.write("workload=cutedsl_add\n")
        fh.write("status=PASS\n")
        fh.write("run_tag=%s\n" % run_tag)
        fh.write("M=%d N=%d dtype=bfloat16\n" % (M, N))
        fh.write("cuda_visible_devices=%s\n" % cuda_visible)
    sys.stdout.write(
        "cutedsl_add PASS run_tag=%s shape=%dx%d dtype=bfloat16 CUDA_VISIBLE_DEVICES=%s\n"
        % (run_tag, M, N, cuda_visible)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
