# Image-backed workspaces

New sessions use a sparse ext4 image mounted by the execution-node agent with
`fuse2fs`. Python and native Rust use the same manifest and wire format. Container
adapters bind the mount at `/workspace`; local-process exposes the actual node
path. `CAPE_WORKSPACE_DIR` is reserved and contains that runtime-visible path.
An omitted cwd uses this directory. Explicit cwd and command text are preserved.

The distribution's fuse2fs 1.47.0 has a significant large-single-directory cost:
the Task-16 10k-file create/read/hash case was approximately 39x slower than the
matched tmpfs directory control, exceeding the predeclared budget. An isolated
1.47.3 diagnostic did not resolve it. Do not interpret bounded backing inodes as
directory-performance parity. This limitation was explicitly accepted for Task 16;
the failed measurement and original performance budgets remain unchanged.

## Configuration

Both `pycape` and `rscape` accept these commands:

```sh
rscape admin get-workspace-config --pool coding
rscape admin set-workspace-config --pool coding --image-size-gib 20 --image-root /tmp
rscape admin set-workspace-config --pool coding --node-id node-a --bytes-per-inode 4096
```

Use the normal `--controller-url` and `--token` options. Updates require pool-admin
authority. Node overrides take precedence over pool defaults. Bootstrap defaults
are 10 GiB logical capacity, a configurable 1 TiB limit, 16 KiB per internal inode,
and a 30-second tool/mount timeout. Sparse capacity is not reserved physical space.
The node validates changes before reporting their version as applied. Default changes
alone do not rewrite existing images. Their creation configuration stays fixed;
actual capacity changes only through a recorded maintenance operation.

## Reuse and maintenance

Commands in a warm session reuse the existing mount. Session release stops runtime
references, unmounts, synchronizes, and retains the original image; it does not delete
user data. The request status includes a nested `workspace` object with its identity,
node, state, path, creation config, and last error.

```sh
rscape workspace status ws-0123456789abcdef0123456789abcdef
rscape workspace release ws-0123456789abcdef0123456789abcdef
rscape workspace check ws-0123456789abcdef0123456789abcdef
rscape run --pool coding --user alice --session-id resumed \
  --workspace-id ws-0123456789abcdef0123456789abcdef -- /bin/sh -c 'ls -l'
rscape workspace delete ws-0123456789abcdef0123456789abcdef
```

Finish or cancel an active command before explicit release. Release is asynchronous:
wait for `state=retained`. Check/delete return an `operation` with a unique ID and
`state=pending`; poll workspace status until the same operation becomes `completed`
or `failed`. An accepted API request is not a completed save/check/deletion.
Only the owner or pool administrator may modify a workspace. Offline maintenance
requires a released session and exclusive filesystem ownership. A failed check
does not repair, truncate, or reformat the image. Deletion retains a small tombstone.

The historical top-level API `workspace_id` alias still means logical `session_id`.
Managed-image selection uses `workspace: {"id": "ws-...", "target": "/workspace"}`;
the CLI translates `--workspace-id` to this nested selector.

## Compatibility and safety boundaries

An explicit bind already targeting `/workspace` conflicts with the default managed
target. Preserve the existing bind/cwd by choosing `--workspace-target /cape-workspace`.
Writes into that explicit external bind remain host-backed and are outside the inode
containment guarantee. This layer is not a new sandbox for local-process workloads.
`HOME`, `TMPDIR`, tool-specific caches, and writable runtime root layers are not
silently redirected. Writes outside the managed directory remain outside inode
containment; opt tools into a cache path beneath `CAPE_WORKSPACE_DIR` when desired.

Install tools with `tests/mimic_use_case/manual/prerequisites.sh`; package installation
is separate from the agent's actual mount/write/unmount/remount readiness probe.
Missing or failed FUSE must never fall back to writing into an ordinary host directory.

The image and bounded management sidecars consume backing inodes, not one backing
inode per internal file. Internal bytes/inodes and backing free space are separate
limits. Optional command-boundary growth and offline maintenance are described below.
There is no live-command resize, automatic repair, expiry deletion, historical snapshot,
remote export, or migration. `/tmp` retention does not survive every reboot, cleanup,
allocation loss, or node loss. Unknown predecessor ownership is fenced rather than
automatically stolen; unresolved restart state requires operator reconciliation.

If persisted state records interrupted workspace commands, agent restart refuses
to assume their old process groups have stopped. Fence those processes before
operator reconciliation; do not delete the state file merely to bypass the guard.
For an unmounted image whose previous owner is gone, offline read-only checking can
establish whether it is reusable. A still-live FUSE writer must not be taken over.

Workspace status separates `space.internal` byte/inode availability from
`space.logical_bytes`, `space.allocated_bytes`, and `space.backing` filesystem/free
space. Internal usage is unavailable after unmount; mounted snapshots are cached
for at most one second. `timings` records the most recent create, format, mount,
and unmount/sync durations, not an unbounded command history. These phase values
exclude scheduling/network time and some surrounding metadata publication work.

Task-16 acceptance is complete with the explicit large-directory performance
exception above; see the repository Task-16 acceptance report for verified cells,
evidence, and environment/scope limitations.

## Offline growth, reclaim and shrink

### Per-workspace automatic reclaim control

```sh
rscape workspace set WORKSPACE_ID --auto-reclaim off
rscape workspace reclaim WORKSPACE_ID
rscape workspace set WORKSPACE_ID --auto-reclaim inherit
```

`set` updates a controller-persisted policy; it works while commands are running
without stopping them or unmounting their workspace. Only the owner or an authorized
administrator may change it. `off` prevents new automatic reclaim grants and clears
waiting candidates, but does not cancel an already granted operation. The returned
`operation` retains its ID/state when maintenance is still pending; setting `off`
is not confirmation that that maintenance stopped. Manual maintenance is unchanged.

`inherit` is the default (also for older records without a policy) and follows the
current pool/node configuration; it does not force automatic reclaim on or bypass
idle, threshold, cooldown or host-budget checks. `workspace status` and `set` return
`auto_reclaim` and `auto_reclaim_enabled`; the latter describes configuration, not
immediate maintenance eligibility. Policies survive controller restarts, session
reuse and successful maintenance. Agent telemetry cannot replace them.

Users can track a list of workspace IDs, turn automatic reclaim off, and explicitly
reclaim each once their task sequence finishes and the offline conditions below hold.
No task-group abstraction or explicit session-end API is required. Setting `off`
only protects against grants after the update; it does not undo earlier grants.
Manual reclaim remains asynchronous and does not restore `inherit` automatically.
This policy does not affect auto-grow, session eviction, retained writable layers
or authorized purge/delete. Physical reclamation does not shrink logical capacity.

### Explicit offline maintenance

Both native implementations expose separate operations:

```sh
rscape workspace grow WORKSPACE_ID --size-gib 20
rscape workspace reclaim WORKSPACE_ID
rscape workspace shrink WORKSPACE_ID --size-gib 5
rscape workspace recover WORKSPACE_ID
```

Use an actual `ws-...` ID and the normal controller/token options. Grow may be requested
for a live session: the running command finishes first, subsequent commands wait, and
the owning agent releases runtime references before unmounting and resizing. Reclaim
and shrink require a released, retained workspace. Reclaim changes physical allocation,
not logical capacity; shrink explicitly reduces capacity. Neither retries a failed user
command or formats an existing image.

Each accepted request returns an operation ID. Poll `workspace status`; `pending` and
intermediate `phase` values are not completion. A retry with `--operation-id ID` must
match the current server-issued ID and parameters; stale/conflicting IDs are rejected.
`recover` explicitly rolls back an interrupted transaction to its verified original,
or finalizes cleanup for a committed transaction. A failed original operation remains
failed even when its recovery succeeds.

The manager checks the original read-only, creates and verifies an independent sparse
candidate, refreshes candidate check metadata, and operates only on that candidate.
After checks and synchronization it commits the new identity/capacity using durable
phases and retains the old image until cleanup. This requires extra disk space and
I/O, including full-image copy verification. Failures can leave a recovery candidate;
unknown states refuse reuse/delete until reconciled. This is bounded transaction
recovery, not a promise of historical snapshots or survival of node-local storage loss.

Creation `config` remains unchanged. `current_image_size_bytes`, `geometry`, `generation`
and `maintenance` describe the actual image and last transaction. Mutated images use
manifest schema 3; older managers must not reinterpret them as unchanged images.
The node advertises `maintenance_protocol=1` only after a real grow/reclaim/shrink probe.

### Automatic policies (opt-in)

```sh
rscape admin set-workspace-config --pool coding \
  --auto-grow on --grow-threshold-percent 80 --grow-target-percent 60 \
  --grow-min-mib 1024 --grow-cooldown-seconds 60
rscape admin set-workspace-config --pool coding \
  --reclaim-on-release on --reclaim-min-mib 64 --reclaim-cooldown-seconds 300
```

Automatic growth and on-release reclaim default **off**, preserving existing behavior
until enabled. Defaults are 80% byte/inode pressure, a 60% growth target, minimum growth
1 GiB, growth cooldown 60 seconds, reclaim minimum relative signal 64 MiB and cooldown
300 seconds. `--maintenance-timeout-seconds` defaults to 300 per external helper
(not an aggregate operation-latency guarantee), and
`--backing-reserve-mib` to 512. Limits use the effective deployment max image size.
Pool/node policy overrides merge by field; reads include `maintenance_defaults` and
desired/applied configuration. Updating defaults does not directly resize old images.

Automatic reclaim uses a reference recorded after formatting or successful maintenance,
plus a clean-release sample. Its signal is
`max(0, allocated-byte change - filesystem-used-byte change)`, capped by current
allocated bytes. Fixed empty-image metadata and live file growth alone do not qualify.
This is a screening heuristic, not an exact measurement of reclaimable blocks.
Missing or stale samples defer automatic reclaim; explicit maintenance remains available.
Existing images are never reformatted to obtain a reference.

Eligible releases create `reclaim_candidate` metadata, **not** a pending operation.
They remain reusable. A fresh idle-worker heartbeat claim lets the controller recheck
policy, generation, node health and outstanding user requests before granting an
exclusive operation. After a grant, normal writer fencing applies: a new command may
wait for the running transaction, which is not killed or replayed to reduce latency.
`reclaim_status` describes eligibility; a `ready` candidate can still wait for host
budget or user work to clear. `usage_generation` distinguishes later reuse from
duplicate reports even when the execution session ID stays unchanged.

Additional controls under `admin set-workspace-config`:

| Option | Default | Meaning |
| --- | --- | --- |
| `--reclaim-idle-seconds` | 60 | Wait after observing a clean release. |
| `--reclaim-start-interval-seconds` | 30 | Minimum spacing of automatic starts on a reported host. |
| `--reclaim-max-candidates` | 32 | Oldest-first candidate selection window per host. |
| `--reclaim-retry-backoff-seconds` | 300 | Delay after failure; unsafe images still require recovery. |
| `--reclaim-low-yield-backoff-seconds` | 3600 | Delay after success recovering less than the configured threshold. |
| `--reclaim-nice` | 10 | Nonnegative nice increment, 0–19, for reclaim helpers only. |

One controller grants at most one maintenance operation per reported kernel boot
identity, including across agents in separate pools/containers. Missing identity
serializes conservatively across that controller. This trusts enrolled-agent metadata
and does not coordinate independent controllers. Host policy uses the most restrictive
interval/window among applicable workspace records. Explicit operations bypass automatic
eligibility/idle rules but share the concurrency gate. Under continuous user demand,
automatic maintenance can wait indefinitely.

Attempts, successful generations and actual reclaimed bytes are tracked separately;
lost grant responses reuse the same operation rather than create another transaction.
Nice uses the existing coreutils dependency (`CAPE_WORKSPACE_NICE_BIN` can override its
path). It is best-effort scheduling, **not** a hard CPU/I/O quota or latency guarantee.
Copy/check/hash and recoverable publication remain unchanged.

Background capacity samples have their own acknowledgement identity, independent of
manifest revisions. Crossing a threshold only requests maintenance while a command
runs. Inode-aware growth may need to cross a block-group boundary; merely doubling a
small image is not sufficient. Limits or insufficient backing space are explicit errors,
not permission to escape into host directories. A long command can still reach ENOSPC
before its boundary, and arbitrary commands are never automatically replayed.

Podman supports stateless policies only. Bubblewrap/Apptainer retain their persistent filesystem state while per-command
references are absent. Workspace ID/path and persistent files remain; processes,
connections and memory are not preserved across a maintenance restart. Unreleased
references—including another mount namespace—block maintenance. No forced/lazy unmount
is used to claim a clean save. Shutdown may defer optional reclaim; save and reclaim
outcomes remain separate, and uncertain integrity always fences reuse.

Round-1 integration acceptance is tracked separately from the original Task-16 results;
see the repository review execution records for its current verification status.
