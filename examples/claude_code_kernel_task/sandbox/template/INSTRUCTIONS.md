# KernelBench workflow

The user request names the current operator. Its reference implementation and
protected cases are staged by the Task in `src/`.

1. Read `src/<op_name>.py`.
2. Implement `ModelNew` in `src/<op_name>_triton_ascend_impl.py`.
3. Validate only with `bash tools/verify_once.sh <op_name>`.
4. Use `output/verify/verify_result_summary.json` to repair failures.
5. Preserve the best fully verified implementation and follow `CLAUDE.md`.

The verifier stages the candidate, acquires one host-shared NPU lease, runs
correctness, benchmarks fully correct candidates, and updates
`metrics_best.json` plus `src/<op_name>_triton_ascend_impl_best.py`.
