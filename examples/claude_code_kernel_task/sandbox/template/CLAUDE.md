# Triton Ascend KernelBench Rollout

This workspace already contains one prepared KernelBench operator task. Use
`INSTRUCTIONS.md` as the source of truth.

Read only the small working set by default:
- `INSTRUCTIONS.md`
- `src/<op_name>.py`
- `output/verify/verify_result_summary.json` after a verifier failure
- `tools/verify_once.sh` only as the validation wrapper

Do not explore the workspace. Avoid broad file discovery such as `.claude/**`,
`.claude/skills/**`, `.claude/refs/**`, `src/*`, or `**/*`. Do not read verifier
source, full `verify_result.json`, full `*.raw.log`, or full skill/reference docs
unless the compact summary is missing and the exact file is necessary.
Do not run `ls`, `find`, `grep`, `ps`, `pgrep`, cache deletion, environment
probes, process inspection, `sleep`, `tail -f`, `nohup`, background `&`, or
reads from `/tmp/claude*` / `*/tasks/*`.

Workflow:
1. Read `INSTRUCTIONS.md` and the reference file.
2. Read the target stub once to satisfy Claude Code's Write guard, then use
   `Write` to replace it with a complete `ModelNew`. Read the target file again
   only after a verifier failure or before a small exact `Edit`.
3. Run the synchronous verifier wrapper from `INSTRUCTIONS.md`; it performs
   AST precheck, staging, NPU verify, and a benchmark after full correctness.
4. Repair only from the compact verifier summary with the smallest targeted
   change.
5. Once `passed_cases == total_cases`, invoke the exact local
   `triton-latency-optimizer` Skill and make one small performance change.
6. Re-run the same verifier after each performance change. It automatically
   benchmarks every fully-correct candidate and keeps only a strictly better
   best snapshot.
7. If the verifier hook reports an early stop, stop immediately.

Hard constraints:
- Keep reasoning and prose short.
- Do not write markdown summaries, status files, or extra implementation notes.
- Do not write literal `<tool_call>` / `<function=...>` markup. Invoke the real
  Claude Code tool when a tool is needed.
- Keep generated Python source ASCII-only, including code comments and
  docstrings. Use `input-to-output`, `>=`, and `<=` instead of Unicode symbols.
- Do not invoke `triton-task-extractor`; task extraction is complete.
- Do not use task-management or planning tools.
- Do not run benchmark commands or ad hoc latency probes. Use only the exact
  `triton-latency-optimizer` Skill for optimization guidance; the verifier
  wrapper performs the benchmark automatically after full correctness.
- Do not edit, replace, wrap, or change permissions on `tools/verify_once.sh`
  or verifier files under `.claude/skills/`; they resolve to immutable
  image-owned tools.
- Do not run bare `python3 -c` probes that import `torch`, `torch_npu`,
  `triton`, or the implementation module. Use `bash tools/verify_once.sh <op>`.
- Treat `No module named torch` or `No module named triton` from bare
  `python3` as an invalid probe, not as a code failure.
- Use `Write` for the initial full target implementation or any large rewrite;
  use `Edit` only for small exact replacements.
- Preserve partial best attempts; avoid broad rewrites unless the structure is
  clearly impossible.
- If AST or compilation reports the same issue twice, simplify the kernel shape
  handling instead of adding another special case.
- Run verifier commands in the foreground. Do not rerun timed-out validation
  as a background task or poll background task files.
- When invoking Bash for `tools/verify_once.sh`, set `timeout=3600000` and
  `run_in_background=false`.

Common rejected patterns:
- Passing `.data_ptr()` to kernels.
- Using `tl.to(...)` instead of `value.to(tl.float32)`.
- Tuning Ascend-rejected kwargs such as `num_warps`, `num_ctas`, or `num_stages`.
- Host-side core computation in `ModelNew.forward()`.
- Runtime Python control flow inside `@triton.jit` kernels.

Validation output is authoritative. AST success is only a precheck; correctness
requires verifier `passed_cases == total_cases`.
