# OpenCode process cleanup: PR #292 + PR #234

## Finding and owning layer

PR #234 alone does not cover the active shell-tool groups created by OpenCode
1.19.3. The CLI and its shell tool have different PGIDs and SIDs. Killing the
CLI's group leaves the tool shell, worker, and worker child running. This was
reproduced for both a Sandbox execution timeout and Agent task cancellation,
using the real pinned CLI with local synthetic SSE and a dummy key.

The gap belongs to LocalSandbox's execution lifecycle. The Agent continues to
delegate process termination to the Sandbox and remove its private state. It
does not acquire a separate PID manager or alter the CLI's spawning semantics.

## Exact comparison

| Component | Revision |
| --- | --- |
| PR #292 head | `e6a53b8b19241b0d2825959edf0312f2986ddb62` |
| PR #234 head | `51ab29fe8b56d6c45dce7d4fc50c61f332374bf7` |
| Main observed for comparison | `d7415466c9b4b3ff7f59a3298ee3e0f5e1297a2f` |
| OpenCode `v1.19.3` source | `e3c4eb242585dcd77be1a867391b7c25fd47ce52` |

The comparison starts with #292 and cherry-picks #234. The relevant LocalSandbox
code on the observed main revision still has the direct-child-only timeout path;
neither PR had been merged at the time of this verification.

The native Linux x86_64 binary reported version `1.19.3`. Its official release
archive SHA256 was
`13a8fb2978302071b86f9d7e00da9c8895f9cd846978379c3bdc6a5333536c9b`;
the extracted executable SHA256 was
`179ca421d61eefeadfa0c50e084a6e79c1623734732f0f4593f81ccfd3ab7ee9`.
Python 3.12.14 was used. This is Linux execution evidence, not a macOS rerun.

## Regression and expected outcomes

Set an explicit executable; missing OPENCODE_BIN skips the real-CLI cases and
must not be reported as verification. A missing or wrong configured binary
fails the test through executable resolution or the adapter's version preflight.

```bash
OPENCODE_BIN=/absolute/path/to/opencode \
  python -m pytest \
  tests/uni_agent/agents/test_opencode_process_cleanup.py -q -s
```

| Scenario | #292 + original #234 | With the provider follow-up |
| --- | --- | --- |
| Timeout while waiting for SSE | Pass | Pass, zero rescue signals |
| Cancellation while waiting for SSE | Pass | Pass, zero rescue signals |
| Timeout after a tool starts | Fail: detached shell, worker and child survive | Pass, zero rescue signals |
| Cancellation after a tool starts | Fail: detached shell, worker and child survive | Pass, zero rescue signals |

The test waits for a real HTTP request or an atomic tool readiness record before
termination. It captures PID, PPID, PGID, SID, process status and start time,
checks ancestry (including CLI launcher wrappers), and verifies that the tool
group differs from the CLI group. After production timeout/cancellation, it
polls passively for up to three seconds. A live tracked process fails the test
before any rescue signal. Failed observation also fails the test; missing `ps`
output is not silently interpreted as process exit.

`opencode_process_observation` is emitted before teardown and
`opencode_process_cleanup` adds the emergency-cleanup record. A passing test also
requires `rescue_signals: []` and no teardown errors. The failure-only cleanup
uses exact fixture identities so that reproducing the leak does not intentionally
leave processes running. Zombie processes count as exited: these tests establish
that tracked processes cannot execute further work, not that every kernel process
table entry has already been reaped.

The original `test_opencode_cli_probe.py` remains a separate observational probe.
Its manual `_terminate_pid()` cleanup must not be used as evidence for this
regression's pass condition.
In particular, its `was_running` field begins with `kill(pid, 0)`, which can
succeed for a zombie, and its immediate `returncode is None` check can precede
asyncio's child-watcher update. Those legacy fields alone are not evidence of a
process still executing after the provider's asynchronous SIGKILL.

The new CPU provider regressions require no OpenCode executable:

```bash
python -m pytest \
  tests/uni_agent/sandbox/test_local_sandbox.py \
  tests/uni_agent/sandbox/test_local_descendant_groups.py -q
```

They cover timeout/cancellation across recursively detached groups, an unrelated
sibling surviving, cancellation during spawn, snapshot failure, a group leader
exiting while a same-group child holds pipes, signal permission failure, and
stale snapshot membership. The permission-failure case deliberately verifies a
survivor and a warning; its success is error-handling coverage, not containment.

The final focused selection also ran the existing model configuration, OpenCode,
OpenClaw, Claude Code and mini-SWE-agent contracts, the original CLI probe, the
real-CLI Task smoke tests, and both LocalSandbox regression modules: **154 passed
in 72.83 seconds**. The final four-case cross-PR file was independently rerun
against the unmodified #292 + #234 combination: **2 failed, 2 passed in 39.54
seconds**, with both failures at the pre-rescue live-tool assertion. The four
new integration cases passed with no rescue signals on the follow-up. Passing
legacy probes and error-injection cases are not additional containment evidence.
Ruff lint/format checks, changed-file compileall and `git diff --check` passed;
the repository-wide inference/training suite was not run.

## Minimal provider follow-up

The patch retains #234's independent session, DEVNULL stdin, shielded spawn,
spawn callback, `_kill_group()` and avoidance of unbounded `proc.wait()`.

Before killing an active leader, a bounded POSIX `ps` snapshot identifies
PPID-descendant PGIDs. The original group and discovered extra groups are sent
through the same `_kill_group()` helper. Extra groups require at least one
observed member still in that group before signaling; the original group remains
eligible after its leader exits. The caller's own group is excluded. Failed
snapshots fall back to the original group and warn; per-group signal failures
warn and allow remaining cleanup attempts without replacing CancelledError or
the timeout result.

This changes exceptional LocalSandbox cleanup only. There is no Agent/Sandbox
API, configuration, model protocol, or dependency migration. One old #234 test
now tolerates ProcessLookupError in its emergency cleanup because the provider
may already have reaped the detached process.

## Compatibility and limits

- **Still active ancestry only.** Reparented background jobs, double forks, and
  detached processes whose parent already exited can remain invisible. Even a
  normal shell command that backgrounds a job with redirected output can create
  this condition before a later model-request timeout.
- **Races remain.** The process-table snapshot is not atomic. New children,
  group changes and PID/PGID reuse are not fully fenced. Membership revalidation
  reduces stale signaling but is not stable process identity or security isolation.
- **Permissions and visibility matter.** Missing/incompatible `ps`, process
  visibility restrictions, or denied signals can leave extra groups alive.
  Warnings expose these errors; the public timeout/cancellation result is unchanged.
- **Async cost.** Each abnormal cleanup may synchronously launch `ps`. Its
  one-second timeout is not a strict wall-clock bound on process creation and
  cleanup. A cancellation burst can serialize this cost. The original PR's
  concurrency benchmarks were not rerun and do not apply to this extension.
- **Platforms.** Linux x86_64/Python 3.12.14 was exercised. The command syntax and
  APIs are intended for POSIX, but macOS was not executed here. Windows Job Objects,
  remote Sandbox providers, Docker, Ray, production Gateway parsers, real models,
  and RL training were not tested by this work.
- **Normal completion.** Successful commands retain existing behavior; redirected
  background work is not automatically terminated after a normal return.

The result supports cleanup of the tested active OpenCode tool tree. It must not
be used to mark LocalSandbox as a reliable isolation boundary for every process
ever launched by the harness. Lifecycle-wide guarantees require a stronger
provider ownership primitive and separate validation.

## Primary sources

- [Uni-Agent PR #292](https://github.com/verl-project/uni-agent/pull/292)
- [Uni-Agent PR #234](https://github.com/verl-project/uni-agent/pull/234)
- [Pinned OpenCode shell tool](https://github.com/anomalyco/opencode/blob/e3c4eb242585dcd77be1a867391b7c25fd47ce52/packages/opencode/src/tool/shell.ts)
- [Pinned OpenCode subprocess spawner](https://github.com/anomalyco/opencode/blob/e3c4eb242585dcd77be1a867391b7c25fd47ce52/packages/core/src/cross-spawn-spawner.ts)
- [Official v1.19.3 release](https://github.com/anomalyco/opencode/releases/tag/v1.19.3)
