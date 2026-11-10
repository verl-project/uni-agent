# CAPE Manual Controller, Node, And Workload Scripts

Start with [the current explicit-role workflow](#current-makefile-workflow-no-handoff).
Controller and user registration are separate; no handoff is needed.

All role wrappers accept `--token-file FILE` for one private personal credential.
Pass the URL/pool explicitly; concurrency supports a different token file per
class and polls that class with the same credential. File-mode calls do not put
the plaintext token on the CLI command line. Node `.json` records beside owned
PID files contain only enrollment/process metadata, not credentials. Shutdown
uses these records instead of parsing agent logs; `--strict-host-audit` opts
into treating unrelated CAPE processes as whole-host audit failures.

Managed image workspaces are enabled by default. `submit.sh` forwards
`--workspace-id` and `--workspace-target` (or `CAPE_SUBMIT_WORKSPACE_ID` and
`CAPE_SUBMIT_WORKSPACE_TARGET`). Its mimic presets retain their explicit
`/workspace` and `/output` binds and use `/cape-workspace` for the additional
managed image unless another target is supplied. The parent mimic workload wrapper uses the same default,
overridable with `CAPE_MANAGED_WORKSPACE_TARGET`. Existing explicit directories
remain outside the managed image's inode-containment guarantee.

This directory contains operator-facing scripts for running CAPE manually:

GPU visibility follows the assigned lease. CPU requests always receive an empty
`CUDA_VISIBLE_DEVICES`; GPU requests default to all leased UUIDs. A caller may supply
an exact UUID subset (or an empty string to hide all GPUs), but not numeric indices
or UUIDs outside the lease. Invalid selections fail at dispatch as `workload` errors
and release the request's lease. Accepted caller text is preserved after trimmed,
order-independent membership checking; narrowing does not reduce reserved capacity.

| Script | Purpose | Process lifetime |
| --- | --- | --- |
| [`prerequisites.sh`](./prerequisites.sh) | Install and verify node-local container runtimes inside a writable outer container | One-time per writable outer rootfs |
| [`controller.sh`](./controller.sh) | Initialize/restore the instance, save its admin credential, provision pools, and start the controller | Foreground until interrupted |
| [`user.sh`](./user.sh) | Independently create/resume one regular user's credential | Exits after registration/recovery |
| [`register_node.sh`](./register_node.sh) | Discover selected GPUs and register one GPU node with the controller | Foreground until interrupted or authoritatively removed |
| [`retire_node.sh`](./retire_node.sh) | Drain and logically remove one registered node allocation | Bounded command; triggers the default agent's self-exit but never releases the cluster allocation |
| [`submit.sh`](./submit.sh) | Submit exactly one asynchronous workload | Exits after printing its request ID |
| [`status.sh`](./status.sh) | Query requests, utilization, and locally visible logs without mutating CAPE | One command, except log following or request waiting |
| [`concurrency.sh`](./concurrency.sh) | Submit many workloads with a bounded cap or an open-loop arrival schedule, mixed workload classes, and an optional throughput/latency report | Waits until every submitted request is terminal |

Run `bash SCRIPT.sh --help` for the option list (except `user.sh`, which uses the
environment variables documented below). This README describes the normal
cross-host workflow, the important path and authentication rules, and common examples.

## Implementation Selection

The script-driven harness defaults to the static `rscape` executable. With no role override,
each host resolves `rscape` from `PATH`, checks `rscape --version`, and creates no CAPE Python
virtual environment. A missing or non-executable rscape is a hard resolution error; the wrappers
never fall back silently.

Python remains a fully maintained explicit choice. Set `CAPE_BIN=pycape` for the controller and
the corresponding role variables `CAPE_NODE_BIN=pycape`, `CAPE_SUBMIT_BIN=pycape`,
`CAPE_STATUS_BIN=pycape`, and `CAPE_RETIRE_BIN=pycape`. That exact bare-name selection engages the
existing role-local Python 3.12 venv creation, editable installation, and `aiohttp`/`pycape`
validation. An explicit executable path continues to mean “use this already prepared binary.”

Every role writes or logs `cape_implementation`, `cape_version`, and the resolved `cape_bin`.
Controller/node roles place structured `implementation.txt` files in their evidence directories;
submit/status roles do so when their existing bundle variables are set; retirement always emits
the same fields to its captured diagnostic log.

`node.sh` was renamed to `register_node.sh`; there is deliberately no compatibility wrapper or
symlink, so update operator commands to the new name.

## Quick Example

### Current Makefile workflow (no handoff)

Controller startup does not create regular users or distribute tokens. All roles
use an explicit controller URL; each user keeps their own private credential
file. Paths below are examples: choose writable locations on the relevant host.

```bash
cd cape/tests/mimic_use_case/manual
export CAPE_URL=http://scheduler.example:18892
export CAPE_USER=alice
export CAPE_USER_TOKEN_FILE=/private/local/alice.json

# Controller host: use persistent LOCAL storage, not NFS.
export CAPE_STATE_DIR=/local/persistent/cape/controller
export CAPE_ADMIN_TOKEN_FILE=/private/controller/admin.json
make controller

# User host: self-register independently; no admin token is needed.
make user

# Allocated GPU host: prepare dependencies and select authorized UUIDs.
make prerequisites
export CAPE_GPU_UUIDS=GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee

# Required for mix targets: an image-capable runtime and a node-visible image.
# Prewarm the workload image once to avoid cold task startup costs.
mkdir -p /local/images
apptainer pull /local/images/pytorch-26.09-py3.sif \
  docker://nvcr.io/nvidia/pytorch:26.09-py3
export CAPE_RUNTIME_ADAPTERS=bubblewrap,apptainer
# Runtime probes independently default to the smaller Ubuntu 24.04 image.
export CAPE_TEST_IMAGE=/local/images/pytorch-26.09-py3.sif

make register_exclusive
make register_shared
make exclusive_ready shared_ready
make exclusive_util shared_util

# Submitter: source/cache bind paths must also exist on the selected node.
make exclusive_host
make shared_host
make exclusive_mix
make shared_mix
make shutdown
```

For host-only use, select CAPE_RUNTIME_ADAPTERS=bubblewrap and omit image
preparation and mix targets. An ACTIVE Bubblewrap-only node cannot execute image
workloads. Package installation is not proof of runtime capability; the agent's
real probes remain authoritative.

| Target | Responsibility |
| --- | --- |
| prerequisites | Install/check runtime and filesystem-image tools, not workload images |
| controller | Initialize/restore the controller and provision exclusive-pool/shared-pool |
| user | Create or resume one regular credential |
| register_exclusive / register_shared | Enroll this host separately in each pool |
| exclusive_host / shared_host | Submit 16 host-stateless workloads |
| shared_host_file_lock | Same shared host workload, serialized GPU phase using a per-UUID file lock |
| exclusive_mix / shared_mix | Submit 16 workloads with a 1:1 host/image class mix |
| concurrency (default) | Run the exclusive host-only target |
| shutdown | Drain recorded memberships and stop owned local processes |
| clean | Shutdown, recover a matching orphan controller, then reset test logs/reports and state/credentials |

`make clean` is a destructive reset of this manual test instance, including its
credentials. Use `make shutdown` to preserve state. Cleanup matches orphan
controllers by exact `CAPE_STATE_DIR`, not name or port, and signals only that
process (not an unrecorded process group). It checks for open files before
deletion. If `.nfs*` files remain, close their holders on the owning node and
retry there; running cleanup on another host cannot release a remote descriptor.
Deletion targets must be distinct subdirectories of the manual working directory.

Makefile defaults:

### GPU contention and cooperative file-lock comparison

The default WORKLOAD_ARGS is CPU sleep 1s followed by fixed-work vecadd using a
full-wave grid. Run `make exclusive_host`, `make shared_host`, then
`make shared_host_file_lock` sequentially on the same GPU with the same arguments
and pool settings. Do not compare against old single-CTA wall-clock sleep runs.
Warm the compile cache first; compilation and cold initialization affect overall
duration, whereas the additional CUDA event measurement times only the kernel.

`GPU_LOCK_DIR` defaults to `/tmp/cape-gpulock-<uid>` and is mounted read/write at
`/runtime/gpulock` for the locked recipe. Make creates it on its local host;
for a remote submitter, provision that same path on the execution node yourself.
All participating commands must bind the same underlying directory. Cross-user
deployment must arrange compatible directory/file permissions; no automatic 1777
permissions or cross-user access is granted. Empty/multiple/numeric CVD values
are rejected for locking, never mapped to `0.lock`.

The lock covers CUDA initialization, data preparation, kernel execution and
synchronized CUDA teardown, but not CPU sleep or compilation. Initialization is
included to prevent a different participating command's setup kernels from
polluting the protected measurement. The lock file is retained; never delete it
while participants are running. It does not change shared leases or prevent
nonparticipating GPU workloads.

The three recipes retain separate `*_report.txt`/JSON/trace outputs and add
`*_report.txt.gpu-timing.json`, containing per-command CUDA `kernel_ms`, lock
waiting, geometry and median/P95 statistics. Raw helper measurements are saved
under `report/host/gpu-sleep-cache/timing/<run-id>/`. Existing trace `gpu_time`
still includes initialization and lock waiting: use the extra kernel metric to
judge contention, not that composite phase. A sleep-mode override has no vecadd
CUDA-event metric and produces a null kernel summary. Compare locked shared
kernel duration with exclusive baseline; inspect lock wait and batch throughput
separately. No automatic universal performance threshold is imposed.

New locked runs split the Perfetto trace into `processing:cpu`, optional
`processing:prepare` (CPU completion to actual lock attempt),
`processing:lock-wait`, and `processing:gpu` (after lock acquisition), followed
by the existing finish slice when present. The post-lock GPU slice still includes
CUDA initialization/teardown; it is not the CUDA-event kernel-only metric.
Report JSON/text also expose `gpu_prepare_time`, `lock_wait_time` and
`gpu_after_lock_time`, while `gpu_time` keeps its backward-compatible composite
meaning. Old/unlocked logs without both lock markers keep their original trace;
invalid marker ordering is flagged and never used to fabricate a wait interval.
Existing ledger files are upgraded atomically by appending optional columns,
preserving old rows. Rerun the recipe to collect actual timestamps; existing
traces cannot be accurately split from a lock-wait duration alone.

Node evidence defaults to `manual/evidence/pool/<node-id>/`: `agent.log`
contains agent output and `agent-logs/` contains command/session logs. Both the
Makefile and direct `register_node.sh` invocation use `manual/evidence` by
default. Override `CAPE_NODE_BUNDLE` to choose another location; more-specific
`CAPE_NODE_STATE_DIR`, `CAPE_AGENT_LOG` and `CAPE_AGENT_LOG_DIR` still take priority.
Existing agents/logs are not migrated. `make clean` retains this evidence directory.
For local file inspection with `status.sh`, set `CAPE_STATUS_BUNDLE` to this
directory (controller logs remain in the separate state bundle).

| Variable | Default / purpose |
| --- | --- |
| CAPE_URL | http://127.0.0.1:18892; localhost is not a remote-host address |
| CAPE_USER | user-000 |
| BUNDLE | $PWD/state |
| CAPE_STATE_DIR | $BUNDLE/controller |
| CAPE_ADMIN_TOKEN_FILE | $BUNDLE/credentials/admin.json |
| CAPE_USER_TOKEN_FILE | $BUNDLE/credentials/$CAPE_USER.json |
| CAPE_SHARED_GPU_SLOTS | 4 |
| CAPE_TEST_IMAGE | docker://nvcr.io/nvidia/pytorch:26.06-py3; overridden above |
| REPORT_DIR | $PWD/report; change it to retain separate target/run reports |
| PROCESSING_CONSTANT | Optional, measured on the actual node/environment |

Keep the same state directory and admin file to restart the same instance.
Changing BUNDLE may change the default credential path without changing the
scheduler you connect to. On multiple hosts, keep admin credentials on the
controller/operator host, use role-local regular credential files, and stop node
roles before stopping the controller. Never distribute an admin/user token bundle.

A standalone graceful retirement is:

```bash
bash retire_node.sh --controller-url "$CAPE_URL" \
  --token-file "$CAPE_USER_TOKEN_FILE" --pool exclusive-pool \
  --node-id NODE_ID --timeout 300
```

Graceful drain stops new dispatch and stops cached warm-idle sessions without
waiting for cache expiry. It never cancels a running command unless --force is
explicitly selected. The default agent exits 0 after authoritative shutdown and
cleanup; CAPE_STAY_AFTER_REMOVAL=1 is a diagnostic opt-out.

Registration generates a fresh node ID and a fresh unique allocation id per invocation.
CAPE_NODE_ID and CAPE_ALLOCATION_ID can explicitly override valid identities, but
retired IDs cannot be reused. Generated names satisfy the 64-character identity limit.

### How make user handles identity and credentials

- If CAPE_USER_TOKEN_FILE does not exist, the wrapper calls user token create
  with CAPE_URL and CAPE_USER: anonymous registration of a **new regular user**.
- If it exists, the wrapper calls user token resume with that file. The saved
  operation contains its original URL and identity binding. Changing CAPE_USER
  or CAPE_URL does not retarget the saved registration.
- This is not a username-based login or plaintext token recovery mechanism.
  Reuse your original credential file for an existing identity.
- A successful call stores the secret in the protected file, not terminal output.
  Each user must have their own file. Do not share admin credentials.

### outcome: rejected, recovery: user token resume

"rejected" means a create request received an HTTP 4xx response. The current CLI
does not print the detailed server rejection. A common cause is that the username
already exists on the scheduler, but the selected local credential file is absent.
The default user-000 is especially easy to reuse accidentally. Invalid username
syntax or another 4xx response can also cause rejection; this message alone does
not establish which cause applies.

The recovery field is a generic protocol hint, **not a guarantee that resume will
succeed**. The client preserves its new credential/operation file even after
rejection. Subsequent make user calls therefore resume it; repeated resume cannot
claim an existing account by name or fix a deterministic duplicate-name rejection.

Check the effective URL, username, and token-file path, and whether that file
existed before the first attempt. Never cat the credential or paste its secret.

```bash
make -n user
# If these variables were exported, print only non-secret configuration:
printf 'URL=%s\nUSER=%s\nTOKEN_FILE=%s\n' \
  "$CAPE_URL" "$CAPE_USER" "$CAPE_USER_TOKEN_FILE"

# Existing identity: reuse its ORIGINAL valid credential.
make user CAPE_USER=alice CAPE_USER_TOKEN_FILE=/private/original/alice.json
```

If you genuinely need a new identity, use an unused username and a different,
not-yet-existing credential path. Preserve the rejected file until the result is
understood. Do not delete the scheduler database, an existing account, or a valid
credential merely to make registration pass.

An existing user wanting an **additional token** must authenticate with an
existing valid credential, rather than anonymously register the same username:

```bash
rscape user token create --controller-url "$CAPE_URL" \
  --token-file /private/original/alice.json \
  --token-out /private/local/alice-extra.json
```

Use pycape or your prepared binary as appropriate. For an uncertain delivery or
transport result, preserve and resume the same file/operation; do not blindly
generate a replacement secret.

## Deployment Model

Controller, node, submitter and status client may use different hosts,
architectures, repositories and venv paths. Share connection information, not
controller-local paths or secrets. Each caller supplies its individual token file.
Bind sources and workload images must be visible at the submitted path on the node.
Do not copy a venv between architectures or incompatible outer containers.

## Pool Modes, Dual Membership, And Pool-Only Routing

The controller provisions exclusive-pool and shared-pool. Standalone controller.sh
also supports CAPE_POOL and CAPE_ADDITIONAL_POOLS. Empty pools accept queued requests
but provide no schedulable capacity.

The same physical GPU UUID may have independent memberships in both pools:
CAPE_POOL_MODE=exclusive or CAPE_POOL_MODE=shared with CAPE_GPU_SLOTS=4 for the
default shared pool. Each enrollment has its own node/allocation IDs, agent
credential and runtime/state directories. A regular token belongs to a user,
not a pool; select routing with --pool and let the scheduler check permissions.

Pool configuration belongs to the administrator. Changing CAPE_GPU_SLOTS on a
node does not redefine it; use admin pool edit with the admin credential and
respect existing work/membership restrictions. Runtime GPU enforcement can be
advisory; gpu_cotenant_count reports sharing, not an isolation guarantee.

A pool is reconfigurable-empty only when its membership, requests, sessions and
uncertain leases permit the change; zero schedulable GPUs alone is not sufficient.
After draining and resolving remaining work, an administrator may explicitly edit:

```bash
pycape admin pool edit shared-pool --controller-url "$CAPE_URL" \
  --token-file "$CAPE_ADMIN_TOKEN_FILE" --pool-mode shared --gpu-slots 4 --placement pack
```

Historical controller-shared-pool.env files are no longer emitted. Optional
standalone readers remain for older callers, but the Makefile and examples here
do not generate or require handoffs.

### Scheduling latency versus liveness

CAPE_HEARTBEAT_INTERVAL primarily controls liveness and retry cadence. In a
saturated compatible deployment, the completion-report response can dispatch
the next command without waiting for a periodic heartbeat; saturated handoff latency
is therefore not simply the heartbeat interval. Account for mixed-version peers
and fallback/retry behavior when interpreting measurements. This scheduling
handoff is unrelated to the obsolete credential handoff files.

## Prerequisites

All roles need Bash and network access to the advertised controller address. Unless an existing
client executable is supplied, a role also needs a local CAPE source checkout and one of:

- a `python3.12` executable;
- an explicit role-specific Python 3.12 executable; or
- Pixi, which the scripts can use to bootstrap Python 3.12 without root privileges.

The GPU node additionally needs:

- `nvidia-smi` and at least one visible, allocated GPU;
- any runtime executables the node should offer (`apptainer`, `bwrap`, and/or `podman`); and
- whatever kernel/runtime privileges those adapters need.

The scripts install CAPE into role-and-host-qualified, architecture-specific venvs such as
`venv/controller-host-a-x86_64-python3.12` or
`venv/node-gpu-01-aarch64-python3.12`. Two roles or hosts with the same CPU architecture can still
have different loaders and outer root filesystems, so they must not share one default venv.
Node, submit, and status installations use a fingerprint marker and are reused on later calls
unless the project identity/metadata changes or `CAPE_FORCE_INSTALL=1` is set. Existing compatible
Python 3.12 venvs are reused. `controller.sh` uses the same scoped installation fingerprint, so an
unchanged editable CAPE installation is not reinstalled on every startup. Use the documented exact
`CAPE_*_VENV` override when an operator intentionally manages another isolated location.

The controller, node-registration, retirement, submit, status, and concurrency wrappers install
the CAPE Python package only. They do not install the GPU driver or container images. On an allocated GPU node,
`prerequisites.sh` can install Apptainer, Bubblewrap, Podman, fuse-overlayfs, fuse2fs,
e2fsprogs (`mkfs.ext4`, `e2fsck`, `dumpe2fs`), util-linux, coreutils, and required helpers
into a writable root-mapped Ubuntu outer container; it does not modify the bare host.

The NVIDIA OCI tag is multi-architecture, but an expanded Enroot rootfs contains binaries for only
the architecture selected during import/create. When a shared Enroot data directory is visible to
different CPU architectures, use architecture-qualified expanded names (for example,
`cape-pytorch-26.09-py3-x86_64` and `cape-pytorch-26.09-py3-aarch64`) and verify that `uname -m`
inside the outer container matches the bare allocation. Reusing one unqualified expanded name can
silently invoke binfmt/QEMU; the role-local CAPE venv naming cannot correct a mismatched outer
rootfs.

For concurrent multi-node pools, keep each expanded writable outer rootfs node-local and give it a
unique name. One shared expanded Enroot directory can hold a rootfs lock for the container's
lifetime, causing another physical node to fail with `Could not acquire rootfs lock`. Install
prerequisites once into each node's persistent writable rootfs, then reuse that same node-local
rootfs for `register_node.sh` as the allocation user.

### Install GPU-Node Container Runtimes and Workspace Tools

Inside the allocated node's writable `nvcr.io/nvidia/pytorch:26.09-py3` outer container:

```bash
cd /gpu/repo/cape/tests/mimic_use_case/manual

CAPE_NODE_RUNTIME_STATE_DIR=/tmp/cape-manual-runtimes \
  ./prerequisites.sh

source /tmp/cape-manual-runtimes/node-runtime.env
```

The first call installs native packages with APT when necessary. Later calls detect the complete
native installation and skip APT; `--check` forbids installation and only verifies packages while
refreshing the artifacts. Use `--force` only when you intentionally want APT to run again. Every
live version/namespace probe is bounded independently (10 seconds by default); set
`CAPE_NODE_RUNTIME_PROBE_TIMEOUT_SECONDS` only when a known slow node needs a different bound.

The generated environment includes `CAPE_WORKSPACE_FUSE2FS_BIN`, `CAPE_WORKSPACE_MKFS_BIN`,
`CAPE_WORKSPACE_E2FSCK_BIN`, `CAPE_WORKSPACE_FUSERMOUNT_BIN`, and `CAPE_WORKSPACE_FINDMNT_BIN`.
The report records each tool version and the FUSE library version separately. A `fusermount`
pathname may resolve to a FUSE 3 compatibility executable; successful live mount/unmount is the
capability test, not its filename. `workspace_toolchain_installed=yes` means only that packages
and executables were verified; `workspace_mount_capability=not-tested-use-cape-node-probe`
does not claim a working workspace. Missing `/dev/fuse` has no ordinary-directory or Apptainer
unsquash fallback for managed workspaces. The installer never formats, mounts, checks, or repairs
existing workspace images.

The 26.06 image can contain the APT/dpkg executables while omitting dpkg architecture tables,
APT state directories, and the Ubuntu archive keyring. `prerequisites.sh` restores only the
bounded Noble metadata it needs and bootstraps Ubuntu's official
`ubuntu-keyring_2023.11.28.1_all.deb` over HTTPS after verifying its pinned SHA-256. It never uses
`trusted=yes`, `--allow-unauthenticated`, or another signature bypass. Package installation still
requires UID 0 inside the writable outer rootfs; that authority must not extend to the bare node.

Some shared filesystems allow the allocation user to execute this checkout but reject direct
`execve` after Enroot remaps the caller to root. In that case stage the script into the writable
outer rootfs, or invoke a readable staged copy with `bash`; do not weaken shared-filesystem
permissions and do not install into bare-host paths.

The installer also replaces stripped-image zero-byte `ldconfig` placeholders with the native
`libc-bin` implementation and creates sticky mode-1777 `/var/tmp`. Both are required before
Apptainer can perform NVIDIA-library discovery and its standard temporary-directory bind.

The sourceable environment contains only node-wrapper inputs:
`CAPE_APPTAINER_BIN=/usr/bin/apptainer`, `CAPE_BUBBLEWRAP_BIN=/usr/bin/bwrap`,
`CAPE_BUBBLEWRAP_SYSTEM_PATHS=/usr,/etc,/opt`, `CAPE_PODMAN_BIN=/usr/bin/podman`, and
`CAPE_APPTAINER_UNSQUASH`. Its sibling report records
versions plus real outer-environment observations for Bubblewrap namespaces, `/dev/fuse`, and
cgroup delegation. Installed binaries do not imply that the outer environment grants those kernel
capabilities; read warnings and the report before selecting an adapter. In particular, the report
distinguishes a userns assembled-root fresh-proc envelope from the capability-dropped root-mapped
assembled-root fresh-proc fallback. It never treats a whole-root or outer-proc host view as usable.
It deliberately records
`bubblewrap_gpu_capability=not-tested-use-cape-node-probe`: package preparation never claims GPU or
CUDA support.

For host workloads that need CPU topology (including ARM PyTorch dtype conversions), set
`CAPE_BUBBLEWRAP_READ_ONLY_PATHS=/sys/devices/system/cpu` before running `register_node.sh`.
The wrapper forwards comma/newline-separated entries as repeated `--bubblewrap-read-only-path`
options. The default is empty. Each entry must remain strictly beneath the same virtual root
after symlink resolution; the runtime skips absent entries and binds present ones read-only
for both host policies, without overlays. The system-path allowlist still rejects virtual trees.

By default the two files live below architecture-qualified node-local `/tmp`. Override
`CAPE_NODE_RUNTIME_STATE_DIR`, `--env-file`, or `--report-file` when the outer rootfs uses another
node-local persistent directory. These artifacts contain no credentials and must not be put in
a shared credential bundle because runtime paths are node-local.

## Direct Submission And Query

`register_node.sh` places retained execution layers under
`$CAPE_RUNTIME_ROOT/retained` unless `CAPE_EXECUTION_STATE_DIR` selects another
node-local directory. Choose durable storage when retention must survive reboot;
do not treat this directory as disposable probe cache. Suspension retains layers,
while explicit execution purge removes them without deleting the workspace image.

Use explicit connection arguments and the same owner credential for submission,
status/logs/wait, and the core CLI's cancel operation. Submission is asynchronous.

For generic commands, omitting `--session-id` (and `CAPE_SUBMIT_SESSION_ID`) lets the
CLI select its last accepted logical session for this controller, user and project
directory, or generate one if none exists. The wrapper does not invent a separate
default identity. Read the selected ID from CLI stderr or request status; the
submission CSV may leave its requested session field empty when selection was implicit.
The mimic A/B sequence and concurrency harness retain their documented explicit
session choices.

```bash
bash submit.sh --controller-url "$CAPE_URL" --token-file "$CAPE_USER_TOKEN_FILE" \
  --pool exclusive-pool --user "$CAPE_USER" --session-id smoke \
  --isolation-policy host-stateless --gpus 1 -- nvidia-smi
bash status.sh --controller-url "$CAPE_URL" --token-file "$CAPE_USER_TOKEN_FILE" \
  --pool exclusive-pool utilization
bash status.sh --controller-url "$CAPE_URL" --token-file "$CAPE_USER_TOKEN_FILE" \
  status REQUEST_ID
bash status.sh --controller-url "$CAPE_URL" --token-file "$CAPE_USER_TOKEN_FILE" \
  wait REQUEST_ID --timeout 300
```

## Built-In Mimic Workloads

`submit.sh --mimic A` runs `gemm_silu.py`. `--mimic B` runs `cutedsl_add.py` and installs the
CUTLASS DSL inside the workload when necessary. Mimic mode creates and mounts:

```text
USER_ROOT/workspace -> /workspace (rw)
USER_ROOT/output    -> /output    (rw)
MIMIC_SOURCE        -> /cape-mimic (ro)
```

It defaults to one GPU, automatic runtime selection, the NVIDIA PyTorch 26.06 image, `/workspace`
as the cwd, and `USER/task` as the session key. The default isolation policy prefers a proven
Apptainer adapter, then Podman; pass `--runtime-adapter` only when a diagnostic run must pin one.

```bash
CAPE_SUBMIT_REPO_ROOT=/submit/repo \
CAPE_SUBMIT_MIMIC_DIR=/shared/cape-mimic-source \
./submit.sh \
  --controller-url "$CAPE_URL" --token-file "$CAPE_USER_TOKEN_FILE" --pool exclusive-pool \
  --user "$CAPE_USER" \
  --user-root /shared/cape-mimic-runs/user-000 \
  --mimic A
```

Both `--user-root` and `CAPE_SUBMIT_MIMIC_DIR` become bind sources. They must be available at the
submitted absolute paths on the GPU node. If the submitter's repository is not shared at the same
path, stage `gemm_silu.py` and `cutedsl_add.py` into a node-visible directory and point
`CAPE_SUBMIT_MIMIC_DIR` there.

Use `--record FILE` to append an analyzer-compatible submission CSV. The record is submitter-local
unless `FILE` names shared storage.

## Concurrent Workload Arrivals

`concurrency.sh` calls `submit.sh` repeatedly, waits for request state through `status.sh`, and
keeps at most `--concurrency N` non-terminal requests from its own run. The cap applies to CAPE
requests, not to simultaneous `submit.sh` processes.

For a generic workload, the first `--` ends concurrency options and the second `--` ends forwarded
submit options:

```bash
./concurrency.sh \
  --workloads 20 \
  --concurrency 4 \
  --arrival-window 0.5:3.0 \
  --arrival-distribution poisson \
  --seed 42 \
  --poll-interval 1 \
  --timeout 3600 \
  --ledger /submit/evidence/manual-smoke-001/concurrency.csv \
  --label-prefix gemm \
  -- \
  --controller-url "$CAPE_URL" --token-file "$CAPE_USER_TOKEN_FILE" --pool exclusive-pool \
  --user "$CAPE_USER" \
  --session-id user-000-concurrency \
  --image docker://nvcr.io/nvidia/pytorch:26.06-py3 \
  --isolation-policy image-stateless \
  --bind /shared/cape-work/user-000:/workspace:rw \
  --cwd /workspace --workspace-target /cape-workspace \
  -- \
  python gemm_silu.py
```

Because every submission in a run is an equivalent client, `concurrency.sh` prepares a uniform
submitter environment once at startup: it runs `submit.sh --resolve-client` (install-or-reuse of
the submitter CAPE client) and exports the printed executable as `CAPE_SUBMIT_BIN` and
`CAPE_STATUS_BIN` to every child, so individual submissions and status polls skip their own
client re-validation. That per-child validation cost is otherwise visible in the report and
Perfetto trace as `submit-wait` (harness overhead). Pre-set `CAPE_SUBMIT_BIN`/`CAPE_STATUS_BIN`
to override the shared client; if the submit script does not support `--resolve-client`, each
child falls back to resolving its own.

The script sleeps before every submission, including the first. A uniform distribution draws
directly from `MIN:MAX`; `poisson` draws a clipped exponential inter-arrival time. Use equal bounds,
for example `--arrival-window 2:2`, for a fixed delay.

Each workload receives these environment variables:

- `CAPE_CONCURRENCY_INDEX`, from 1 through the requested workload count;
- `CAPE_CONCURRENCY_TOTAL`; and
- `CAPE_CONCURRENCY_RUN_ID`.

The harness also assigns a unique `RUN_ID-NNNNNN` `--session-id` to each request by default.
Without it, requests sharing the same `session_id` match one active reusable session and CAPE
correctly serializes them as `waiting_for_session`, defeating a concurrency measurement. Forward
an explicit `--session-id KEY` when that shared-session serialization/reuse is intentional; the
harness preserves the operator's value instead of injecting per-request keys.

Use the index in a generic command when every request needs a distinct output path. Request IDs are
printed to stdout as they are submitted; state transitions and the final summary go to stderr. The
optional ledger records submission, state, terminal, and timeout events.

With --token-file, submission and polling use the same personal credential.
Per-class --token-file overrides are also used for that class's status polling.
Do not combine --no-token with --token-file.

For repeated mimic workloads, give each request a separate user root:

```bash
./concurrency.sh \
  --workloads 8 \
  --concurrency 4 \
  --arrival-window 0.25:1.5 \
  --work-root /shared/cape-mimic-runs/concurrency-001 \
  -- \
  --controller-url "$CAPE_URL" --token-file "$CAPE_USER_TOKEN_FILE" --pool exclusive-pool \
  --user "$CAPE_USER" \
  --mimic A
```

The wrapper adds `workload-000001`, `workload-000002`, and so on below `--work-root`. The root must
still be node-visible. `concurrency.sh` waits for all submitted requests, exits nonzero if any fail
or become untrackable, and intentionally does not cancel active requests on timeout or interruption.

### Gap Mode Versus Schedule Mode

`concurrency.sh` decides *when workload k is submitted* in one of two ways.

**Gap mode (default, `--arrival-window MIN:MAX`)** is relative: each iteration draws a fresh
delay just before its own submission and sleeps it, after passing the concurrency gate. The
actual arrival time is therefore the accumulated sum of everything that came before:

```text
arrival_k = sum over i=1..k of (cap_wait_i + gap_i + submit_latency_i)
```

Example: `--workloads 3 --concurrency 10 --arrival-window 1:1` (fixed 1 s gaps) with a
`submit.sh` that takes 2 s per call:

```text
t=0..1   sleep gap 1        t=3..4   sleep gap 2        t=6..7   sleep gap 3
t=1..3   submit #1 (2 s)    t=4..6   submit #2          t=7..9   submit #3
```

The requested 1 s spacing silently became 3 s spacing. With a cap the coupling is stronger:
under `--concurrency 1` and 60 s workloads, submission #2 waits at the gate until #1
completes, so the arrival rate is dictated by service time. That is the intended behavior for
a polite bounded-footprint smoke run, but it makes "arrival delay" unusable as an independent
input to a latency decomposition.

**Schedule mode (`--arrival-schedule uniform:HORIZON | poisson:RATE`)** is absolute: all N
offsets from a common time-zero `t0` are drawn *before the first submission* (seeded and
reproducible). With `--concurrency 0`, the parent launches bounded `submit.sh` workers at those
deadlines instead of waiting for each request id. Every worker writes a private node-local result
envelope; the parent validates and records ids in workload-index order. `--submitter-cap` defaults
to 16 and applies back-pressure rather than allowing unbounded forks.

Example: `--workloads 4 --arrival-schedule uniform:6 --seed 7` might pre-draw sorted offsets
`0.8, 2.1, 3.4, 5.9`. If submit #1 takes 3 s and the submitter cap has room:

```text
t=0.8  launch submit #1 (worker remains active)
t=2.1  launch submit #2 on time
t=3.4  launch submit #3 on time; #1 may return independently
t=5.9  launch submit #4 on time
```

Example: `--workloads 8 --arrival-schedule poisson:2 --seed 42` draws exponential
inter-arrivals with mean 0.5 s and cumulatively sums them (e.g. offsets `0.51 ... 3.20`), a
genuine 8-request burst in ~3.2 s — against a 2-GPU pool the excess waits in CAPE's queue,
which is exactly what the report's queuing-time column measures.

Interaction rule: schedule mode wants `--concurrency 0` (open-loop). With a nonzero cap the
admission gate still runs before the deadline sleep, so a full pool holds submissions past
their offsets; the script warns at startup and the slip appears as skew.

Schedule fidelity is a verdict, not just a chart. By default, submit-skew p95 must be no more than
`0.5 * requested mean inter-arrival` (with a 50 ms timer/process floor). Override the ratio with
`--schedule-skew-threshold-ratio`. A miss sets `run_status: degraded`, states the measured and
allowed p95, and makes the command return nonzero after sealing evidence.

| | Gap mode | Schedule mode |
| --- | --- | --- |
| Delay drawn | just before each submission | all N at t0, seeded |
| Reference point | previous submission | common time-zero |
| Slow submit / busy pool | stretches all later arrivals | workers overlap; cap saturation becomes measured skew |
| Arrival independent of service? | no (closed-loop) | yes with `--concurrency 0` (open-loop) |
| Use for | bounded-footprint smoke runs | throughput/latency measurement |

### Workload Classes

Workload classes let one run submit a heterogeneous mix instead of N identical clones. A class
is a named bundle of extra `submit.sh` options (plus an optional command override) with a
weight; each of the N workloads is assigned to exactly one class before the run starts.

**Defining classes.** Two input forms:

- Inline, repeatable, for simple mixes:

  ```bash
  --class 'host:1:--isolation-policy host-stateless' \
  --class 'image:3:--isolation-policy image-stateful --gpus 2'
  ```

  The format is `NAME:WEIGHT:` followed by extra submit options. Inline args are
  whitespace-split, so values must not contain spaces — the deliberate trade-off for never
  reconstructing forwarded arguments with `eval`.

- A JSONL manifest (`--mix-file FILE`), one class per line, for anything inline cannot express:

  ```json
  {"name": "calibrated", "weight": 2, "args": ["--env", "NOTE=two words ok"], "command": ["bash", "-c", "echo done"]}
  {"name": "gemm",       "weight": 1, "args": ["--isolation-policy", "image-stateless"]}
  ```

  `args`/`command` are proper string lists (spaces fine), and `command` replaces the shared
  trailing command after `--` for that class only.

Validation applies to both forms: names match `[A-Za-z0-9_.-]+`, weights are positive
integers and duplicate names are rejected. Use an explicit URL in the shared
options and an individual --token-file for each user's class.

**Assignment.** Weights are expanded into a cycle and workload indexes 1..N draw from it:

```text
class_cycle: host, image, image, image          # weights 1:3
round-robin (default): assignment[k] = cycle[(k-1) % 4]   # host,image,image,image,host,...
random:                assignment[k] = cycle[RANDOM % 4]  # seeded via --seed, reproducible
```

With `--workloads 8` and weights `host:1`/`image:3`, round-robin yields exactly 2 host + 6
image requests. Assignment happens before the submission loop, so a workload's class is fixed
regardless of timing.

**What a class changes at submission.** The effective `submit.sh` argv is assembled as:

```text
shared options + class args + injected values + -- + command
                              --label PREFIX-CLASS-NNNNNN
                              --env CAPE_CONCURRENCY_CLASS=NAME
                              --env CAPE_CONCURRENCY_INDEX/TOTAL/RUN_ID
                              [--user-root WORK_ROOT/CLASS/workload-NNNNNN]
```

Because `submit.sh` takes the last occurrence of a repeated flag, class args appended after
the shared options win. Changing --user alone cannot change request ownership:
a class selecting user-001 must also supply user-001's credential file. Mixed `--isolation-policy`, `--gpus`, `--image`, or
`--runtime-adapter` per class work the same way — override only what differs, share the rest.
The class name also lands in the workload environment, the request label, and a `class`
ledger column, so the report can break every metric down by class. For `--mimic` classes the
injected user root becomes `WORK_ROOT/CLASS/workload-NNNNNN`, so repeated mimics across
classes cannot overwrite each other's outputs.

**Per-class polling identity.** Each class may select its own --token-file.
That credential authenticates both submission and polling; --user is only an
identity consistency check, not authorization. Do not fall back to an admin token
for cross-user status. Give each test user their own private credential file.

**Without classes** the behavior reduces exactly to the original single-mix contract: labels
stay `PREFIX-NNNNNN`, the `class` ledger column is empty, and one polling identity is used.

Worked example (two users, warm-session identities, 1:1 round-robin):

```bash
--class 'usera:1:--user user-000 --token-file /private/user-000.json --session-id smoke-a' \
--class 'userb:1:--user user-001 --token-file /private/user-001.json --session-id smoke-b' \
--mix-order round-robin
```

With 8 workloads, indexes 1,3,5,7 become `usera` (owned by user-000, session identity
smoke-a) and indexes 2,4,6,8 become `userb`; the report then shows queuing/elapsed statistics
separately per class.

### Measurement Mode: Schedules, Classes, And Reports

Beyond the bounded-cap smoke above, `concurrency.sh` can run open-loop measurement experiments:

- `--concurrency 0` removes the local cap (open-loop): requests are submitted on their arrival
  schedule and CAPE's own queueing absorbs the load.
- `--arrival-schedule uniform:HORIZON | poisson:RATE` pre-generates every request's absolute
  arrival offset from a common time-zero before the first submission (reproducible with `--seed`).
  `uniform:30` spreads all arrivals over 30 seconds; `poisson:2` is a Poisson process with two
  arrivals per second. The ledger records both the scheduled offset and the actual submit offset,
  so schedule slip is visible. Honoring a schedule exactly requires `--concurrency 0`; the
  open-loop submitter pool is bounded by `--submitter-cap` (default 16).
- `--class` / `--mix-file` / `--mix-order` define a heterogeneous workload mix — see the
  "Workload Classes" section above for the full semantics (definition forms, weighted
  assignment, argv merge order, per-class polling identity).
- `--report FILE` writes a run report (text to `FILE`, machine-readable `FILE.json`, run metadata
  `FILE.meta.json`), records `status.sh utilization` snapshots at run start and end
  (`FILE.utilization-*.json`, detecting the schedulable GPU capacity actually offered), and
  collects `CAPE_CALIBRATED work_start/cpu_done/gpu_done/work_end` markers from each terminal
  request's logs.
  The JSON/text report also records the schedule-fidelity verdict, scheduled and actual spacing,
  dispatched/completed/failed submitter counts, peak workers, and `submit.sh` phase statistics.
  Per-call phase files are copied after dispatch to `FILE.submitters-RUN_ID/`; transient worker
  stdout/stderr/result envelopes stay on node-local `${TMPDIR:-/tmp}` so NFS evidence writes cannot
  move an arrival timestamp.
  Ledger `TERMINAL` rows also capture placement: `node_id`, `isolation_policy`,
  `runtime_adapter`, and the leased `gpu_uuids` from the request status JSON.
- `--report` also emits a Perfetto timeline, `FILE.trace.json` (Chrome trace-event format —
  open it at <https://ui.perfetto.dev>). The timeline shows one track group per GPU UUID and,
  inside each GPU, one row per request served on that GPU (multi-GPU requests appear under each
  of their GPUs). Complete calibrated markers draw `queuing`, `processing:cpu`,
  `processing:gpu`, and a positive-duration `processing:finish`; stable distinct names render in
  distinct Perfetto colors. CPU/CPU or CPU/GPU overlap is benign CPU-phase co-location, while two
  overlapping `processing:gpu` slices on one GPU track are real GPU co-location contention.
  Generic two-marker workloads retain one `processing` slice, fallback requests draw one submit
  → terminal slice, and schedule-mode runs additionally draw `submit-wait` from the scheduled
  offset to the actual submit. Slice arguments carry the phase name, request id, class, state,
  isolation policy, runtime adapter, and node. Timestamps are relative to the run's t0 (absolute
  epoch recorded in the trace's `otherData`).

The report decomposes each request's elapsed latency as
`arrival delay + queuing time + actual processing time`. When a workload printed calibration
markers (see `../calibrated_workload.sh`: `sleep 1` CPU work plus a single-CTA GPU kernel sleeping
1 s on-device, theoretical processing 2 s), queuing and processing are measured directly; a
complete marker set additionally reports `cpu_time = cpu_done − work_start` and
`gpu_time = gpu_done − cpu_done`. Without markers the behavior depends on the optional
`--processing-constant`: when a calibrated constant is supplied (median elapsed of a few direct
`calibrated_workload.sh` runs in the same execution environment), the report estimates
`queuing = elapsed − arrival − constant`; when it is unset, marker-less requests honestly report
elapsed only (`measurement_path: elapsed-only`) instead of a fabricated decomposition, and the
processing-vs-calibration anomaly cross-check is skipped. The report includes overall and
steady-state (middle 80%) throughput, mean/p50/p95/max per latency component (including CPU/GPU
sub-times where measured), per-class / per-isolation-policy / per-node breakdowns, submit-skew
statistics, wrapper phase decomposition (entry→handoff, client resolution, argv preparation,
binary round trip, exit), and anomaly notes (negative queuing; and, when a constant is supplied, measured
processing far from it).

`make concurrency` runs 16 host-stateless requests. Use `make exclusive_mix` or
`make shared_mix` for the 1:1 host/image matrix. All require a reachable CAPE_URL and a valid
CAPE_USER_TOKEN_FILE;
`PROCESSING_CONSTANT=<median>` is optional and must be non-negative when supplied.

```bash
./concurrency.sh \
  --workloads 32 \
  --concurrency 0 \
  --arrival-schedule poisson:2 \
  --seed 42 \
  --report /submit/evidence/measure-001/report.txt \
  --class 'host:1:--isolation-policy host-stateless' \
  --class 'image:1:--isolation-policy image-stateful' \
  -- \
  --controller-url "$CAPE_URL" --token-file "$CAPE_USER_TOKEN_FILE" --pool exclusive-pool \
  --user "$CAPE_USER" \
  --gpus 1 \
  -- \
  /usr/bin/bash -c 'PATH="$PATH:/usr/local/cuda/bin" exec /path/visible/on/node/calibrated_workload.sh --cache-dir /path/visible/on/node/gpu-sleep-cache'
```

For a cold host-stateless Bubblewrap cache, `/usr/local/cuda` may be mounted while its `bin`
directory is absent from the sanitized workload PATH. Add it explicitly as above (or pass an
equivalent `--env PATH=...`) so the script can find `nvcc`; mounting a tool does not add it to
PATH.

Ledger schema v4 note: the measurement columns (`label,class,scheduled_offset,submit_offset,
node_id,isolation_policy,runtime_adapter,gpu_uuids,work_start,work_end,cpu_done,gpu_done,
created_at,leased_at,started_at,gpu_mode`) are
appended after the original columns; a ledger created by an older `concurrency.sh` has a different
header and is rejected rather than silently mixed.

## Read-Only Status And Logs

The complete read-only surface is:

```bash
./status.sh [CONNECTION_OPTIONS] health
./status.sh [CONNECTION_OPTIONS] status REQUEST_ID
./status.sh [CONNECTION_OPTIONS] logs REQUEST_ID
./status.sh [CONNECTION_OPTIONS] wait REQUEST_ID [--timeout SECONDS]
./status.sh [CONNECTION_OPTIONS] utilization
./status.sh [CONNECTION_OPTIONS] info
./status.sh [CONNECTION_OPTIONS] controller-log [--lines N] [--follow]
./status.sh [CONNECTION_OPTIONS] node-log [NODE_ID|all] [--lines N] [--follow]
./status.sh [CONNECTION_OPTIONS] command-log COMMAND_ID [NODE_ID|all] [--lines N] [--follow]
./status.sh [CONNECTION_OPTIONS] session-log SESSION_ID [NODE_ID|all] [--lines N] [--follow]
```

`status`, `logs`, `wait`, and `utilization` query the controller and therefore work from any host
that can reach it and has valid credentials. `controller-log`, `node-log`, `command-log`, and
`session-log` use `tail` on the current host. Local log files are not transported by the controller.
`utilization` includes `active_nodes`, `registering_nodes`, `gpus_schedulable`, `gpus_available`,
`node_states`, and the established `pool_config`, so node readiness, usable capacity, and an empty
pool's mode/slots/placement can be checked without access to node-local logs.
Controller `logs REQUEST_ID` returns the unchanged captured workload stdout/stderr. A node-local
`command-log cmd-...` additionally shows the exact shell-quoted runtime argv, raw argv JSON,
runtime cwd/env overrides, adapter, policy, per-phase exit codes, and the unchanged output body.
`session-log sess-...` records later stop/unmount/cleanup commands and may appear only when CAPE
actually tears down that session.

On a host that can read the controller bundle:

```bash
CAPE_STATUS_CONTROLLER_LOG=/controller/evidence/manual-smoke-001/pool/controller/controller.log \
./status.sh controller-log --lines 100 --follow
```

On a GPU node that can read its own agent log:

```bash
CAPE_STATUS_NODE_LOG=/gpu/evidence/manual-smoke-001/pool/gpu-node-01/agent.log \
./status.sh node-log all --lines 100 --follow

CAPE_STATUS_AGENT_LOG_DIR=/gpu/evidence/manual-smoke-001/pool/gpu-node-01/agent-logs \
./status.sh command-log cmd-000001 --lines 200

CAPE_STATUS_AGENT_LOG_DIR=/gpu/evidence/manual-smoke-001/pool/gpu-node-01/agent-logs \
./status.sh session-log sess-000001 --lines 200
```

Alternatively, set `CAPE_STATUS_BUNDLE` to a locally visible bundle. Then `controller-log` derives
`BUNDLE/pool/controller/controller.log`, while `node-log all` discovers
`BUNDLE/pool/*/agent.log`; command/session lookup discovers
`BUNDLE/pool/*/agent-logs/ID.log`. Use `status.sh info` to print resolved non-secret settings; it
reports whether a token is set but never prints the token.

## Retiring A Node

`retire_node.sh` is the mutating counterpart to read-only `status.sh`. Run it from any trusted
operator host that can reach the controller and use the contributor's regular credential. Identify the
logical CAPE allocation directly, or resolve it from one currently visible node id:

```bash
# Default graceful path: drain, wait, then remove.
./retire_node.sh \
  --controller-url "$CAPE_URL" --token-file "$CAPE_USER_TOKEN_FILE" --pool exclusive-pool \
  --allocation-id manual-smoke-001-gpu-node-01 \
  --reason planned_maintenance \
  --timeout 300

# Equivalent identity lookup while the node remains visible.
./retire_node.sh \
  --controller-url "$CAPE_URL" --token-file "$CAPE_USER_TOKEN_FILE" --pool exclusive-pool \
  --node-id gpu-node-01
```

The graceful path prints labelled JSON Lines for the utilization before the change, the drain
result, every wait observation, the remove result, and final utilization. It first makes the
allocation unschedulable, then waits until `active_session_count` and `nonterminal_requests` are
both zero and waits for scheduler-confirmed retirement/cleanup. It never cancels a request. On the transition to
`DRAINING`, CAPE immediately stops reusable warm/cached-idle sessions and releases their retained
CPU/memory accounting. Work that was already running finishes naturally, receives runtime cleanup,
and goes to `STOPPED` instead of returning to the cache. Choose `--timeout` for the longest
remaining workload plus cleanup and heartbeat/polling delay; ordinary idle-cache expiration is not
part of the bound. If `--timeout` expires, the command returns nonzero and leaves the allocation
drained; it does not silently escalate to force.

The explicit alternatives are:

```bash
# Stop new placements but leave existing sessions/work and the allocation record in place.
./retire_node.sh --controller-url "$CAPE_URL" --token-file "$CAPE_USER_TOKEN_FILE" --pool exclusive-pool \
  --allocation-id manual-smoke-001-gpu-node-01 --drain-only

# Explicitly cancel assigned work, then wait for verified cleanup and retirement.
./retire_node.sh --controller-url "$CAPE_URL" --token-file "$CAPE_USER_TOKEN_FILE" --pool exclusive-pool \
  --allocation-id manual-smoke-001-gpu-node-01 --force
```

Success means that this allocation contributes no schedulable capacity afterward. Repeating an
allocation-id retirement after removal is an idempotent, legible no-op; add `--strict` when absence
must instead fail automation. A removed node cannot be resolved by `--node-id`, because the
read-only utilization result intentionally omits removed nodes, so use the allocation id for an
idempotent retry. Use the contributor's regular --token-file, or an explicitly selected admin credential.
When the removed allocation contained the pool's last visible member, the wrapper says that the
pool is now empty and reconfigurable. That note is informational; the controller still rejects a
new tuple if a non-terminal request, cached/live session, or uncertainty lease remains.

Retirement remains a controller mutation: `retire_node.sh` never signals a remote process and never
releases a Slurm/ssh-gw allocation. The controller retains a structured `REMOVED` tombstone. On its
next heartbeat, the default `rscape node enroll` agent verifies that positive authoritative response,
stops accepting work, cleans its local process/session/runtime state, logs the stable removal
reason, and exits 0. Network errors, timeouts, 5xx/auth failures, and an ambiguous missing-node
response do not trigger exit; they remain retry/recovery paths.

That tombstone retires an enrollment, not the physical machine. Running
`register_node.sh` again creates a fresh node ID and a fresh unique allocation ID
per invocation. Explicitly reusing retired IDs is rejected; it does not create a
new contribution. A saved non-retired enrollment may use `CAPE_NODE_RESUME=1`.

Set `CAPE_STAY_AFTER_REMOVAL=1` before `register_node.sh` only when diagnostics or an external
supervisor deliberately require the older heartbeat-after-removal behavior. The wrapper then
forwards `pycape node enroll --stay-after-removal`, and the operator must stop that opted-out process.
In either mode, cluster allocation release remains a separate operator action.

## Runtime Detection On GPU Nodes

By default `register_node.sh` lets `rscape node enroll` automatically probe
Apptainer, Bubblewrap and Podman. Absent or failed automatic adapters are skipped;
healthy peers remain eligible, and no successful adapter means no ACTIVE node.
Set `CAPE_RUNTIME_ADAPTERS=apptainer,podman,bubblewrap` (or a subset) to require
specific adapters: a requested adapter failing its execution probe is an error.
`local-process` must be explicitly selected and is never a fallback.
One node registration therefore serves all proven policies
without double-counting its CPU, memory, or GPUs.

Apptainer and Podman default to `docker://docker.io/library/ubuntu:24.04` for
their basic/stateful probes. Override with `CAPE_APPTAINER_PROBE_IMAGE` and
`CAPE_PODMAN_PROBE_IMAGE`; the former may also be a prepared local SIF. These are
independent of `CAPE_TEST_IMAGE` and the workload's `--image`. Ubuntu contains
the shell/core utilities needed by current probes, not a complete CUDA test
environment. Bubblewrap keeps its separate real GPU checks. Prepare/cache the
probe image where registry access is unavailable; do not claim capability solely
because a binary is installed or an image was downloaded.

The agent self-exits after authoritative removal by default. Set `CAPE_STAY_AFTER_REMOVAL=1` to
forward the explicit `--stay-after-removal` diagnostic opt-out; transient controller failures never
select either behavior implicitly.

| Adapter | Required executable setting | Important notes |
| --- | --- | --- |
| `apptainer` | `CAPE_APPTAINER_BIN` or `apptainer` on `PATH` | `CAPE_APPTAINER_PROBE_IMAGE` proves supported policies; lack of `/dev/fuse` may limit them, while `CAPE_APPTAINER_UNSQUASH=1` requests the unsquash path |
| `bubblewrap` | `CAPE_BUBBLEWRAP_BIN` or `bwrap` on `PATH` | `CAPE_BUBBLEWRAP_SYSTEM_PATHS` is a node-local comma/newline allowlist (default `/usr,/etc,/opt`). CAPE prefers userns + fresh PID/proc; an effective-UID-0 fresh-proc fallback drops all capabilities. There is no outer-proc/whole-root fallback. GPU policies require exact `nvidia-smi`, `cuInit`, and a real kernel. |
| `podman` | `CAPE_PODMAN_BIN` or `podman` on `PATH` | `CAPE_PODMAN_PROBE_IMAGE` plus rootless namespaces, storage, and cgroup delegation must work |
| `local-process` | None | Executes directly in the node/outer environment; no container filesystem isolation |

Node-local runtime state defaults below `/tmp/cape-RUN_ID-NODE_ID`. Override
`CAPE_RUNTIME_ROOT` or the documented per-adapter cache/tmp/runtime variables when `/tmp` is
unsuitable. Each adapter receives separate storage roots. These settings belong to each GPU node
and are not inferred from a credential file. `CAPE_RUNTIME_ADAPTERS` is an optional comma/space-separated
probe filter; the removed singular `CAPE_RUNTIME_ADAPTER` is rejected.

Bubblewrap host policies do not expose the node's home, scratch, repository, credentials, or
datasets by default. Every workload path outside the configured system/toolchain allowlist must be
passed with `submit.sh --bind SOURCE:TARGET:ro|rw`; `--cwd` names the in-container target. The target
can be new (for example `/workspace` or `/cache`) because CAPE assembles a fresh root. System paths
are node-local deployment configuration, not credential metadata.

## Role-Local Configuration Reference

Use the role prefix matching the host on which a script runs:

Repository-root variables point to the combined `cape/` checkout. Project variables point to the
installable Python sibling and default to `<repo-root>/pycape`; override them only when that Python
directory is staged elsewhere. Shared mimic/test paths remain relative to the repository root.

| Role | Source/install settings | Evidence/path settings |
| --- | --- | --- |
| Controller | `REPO_ROOT`, `CAPE_PROJECT`, `CAPE_VENV_ROOT`, `CAPE_VENV_HOST_ID`, `CAPE_VENV`, `CAPE_PYTHON_312`, `CAPE_PIXI_BIN`, `CAPE_BIN` | `BUNDLE`, `CAPE_POOL`, `CAPE_ADDITIONAL_POOLS`, contention policy/ratio/overrides, `CAPE_PROFILES_FILE`, three `CAPE_MAX_*` quotas, idle/reconciliation bounds, `CAPE_STATE_DIR`, `CAPE_ADMIN_TOKEN_FILE`, `CAPE_CONTROLLER_LOG` |
| GPU node | `CAPE_NODE_REPO_ROOT`, `CAPE_NODE_PROJECT`, `CAPE_NODE_VENV_ROOT`, `CAPE_NODE_VENV_HOST_ID`, `CAPE_NODE_VENV`, `CAPE_NODE_PYTHON_312`, `CAPE_NODE_PIXI_BIN`, `CAPE_NODE_BIN` | `CAPE_NODE_BUNDLE`, `CAPE_POOL_MODE`, `CAPE_GPU_SLOTS`, `CAPE_POOL_PLACEMENT`, lifecycle/cache fields, `CAPE_STAY_AFTER_REMOVAL`, `CAPE_NODE_CACHED_IMAGES`, `CAPE_NODE_LOG_REDACT_ENV_KEYS`, `CAPE_NODE_STATE_DIR`, `CAPE_AGENT_LOG`, `CAPE_AGENT_LOG_DIR`, `CAPE_AGENT_STATE_PATH`, runtime-local variables |
| Retirement operator | `CAPE_RETIRE_REPO_ROOT`, `CAPE_RETIRE_PROJECT`, `CAPE_RETIRE_VENV_ROOT`, `CAPE_RETIRE_VENV_HOST_ID`, `CAPE_RETIRE_VENV`, `CAPE_RETIRE_PYTHON_312`, `CAPE_RETIRE_PIXI_BIN`, `CAPE_RETIRE_BIN` | `CAPE_RETIRE_URL`, `CAPE_RETIRE_POOL`, `CAPE_RETIRE_TOKEN`, `CAPE_RETIRE_ALLOCATION_ID`, `CAPE_RETIRE_NODE_ID`, `CAPE_RETIRE_REASON`, `CAPE_RETIRE_TIMEOUT`, `CAPE_RETIRE_POLL_INTERVAL` |
| Submitter | `CAPE_SUBMIT_REPO_ROOT`, `CAPE_SUBMIT_PROJECT`, `CAPE_SUBMIT_VENV_ROOT`, `CAPE_SUBMIT_VENV_HOST_ID`, `CAPE_SUBMIT_VENV`, `CAPE_SUBMIT_PYTHON_312`, `CAPE_SUBMIT_PIXI_BIN`, `CAPE_SUBMIT_BIN` | `CAPE_SUBMIT_BUNDLE`, `CAPE_SUBMIT_USER_ROOT`, `CAPE_SUBMIT_USER_TOKENS`, `CAPE_SUBMIT_RECORD` |
| Status host | `CAPE_STATUS_REPO_ROOT`, `CAPE_STATUS_PROJECT`, `CAPE_STATUS_VENV_ROOT`, `CAPE_STATUS_VENV_HOST_ID`, `CAPE_STATUS_VENV`, `CAPE_STATUS_PYTHON_312`, `CAPE_STATUS_PIXI_BIN`, `CAPE_STATUS_BIN` | `CAPE_STATUS_BUNDLE`, `CAPE_STATUS_CONTROLLER_LOG`, `CAPE_STATUS_NODE_LOG`, `CAPE_STATUS_AGENT_LOG_DIR` |
| Concurrency driver | Uses the submit/status settings forwarded after `--` | `CAPE_CONCURRENCY_BUNDLE`, `CAPE_CONCURRENCY_WORK_ROOT`, `CAPE_CONCURRENCY_LEDGER` |

Pass explicit --controller-url, --pool where needed, and --token-file.
Runtime and installation paths stay local to the role executing the command.

## Troubleshooting

### The controller URL contains an unroutable hostname

Set `CONTROLLER_HOST` before starting the controller. The default comes from `hostname -f`, which
may resolve only locally in some allocations. Confirm from each consumer with:

```bash
./status.sh --controller-url "$CAPE_URL" --token-file "$CAPE_USER_TOKEN_FILE" --pool exclusive-pool health
```

### Python 3.12 was not found

Set the variable for the role that is failing:

```text
controller: CAPE_PYTHON_312 or CAPE_PIXI_BIN
node:       CAPE_NODE_PYTHON_312 or CAPE_NODE_PIXI_BIN
retirement: CAPE_RETIRE_PYTHON_312 or CAPE_RETIRE_PIXI_BIN
submitter:  CAPE_SUBMIT_PYTHON_312 or CAPE_SUBMIT_PIXI_BIN
status:     CAPE_STATUS_PYTHON_312 or CAPE_STATUS_PIXI_BIN
```

An executable and version match are not enough. The selected interpreter must have `venv` plus one
of two pip bootstrap paths: a usable pip wheel in `ensurepip`'s bundled/configured system-wheel
directory, or a working base pip that supports `--python`. Some stripped images omit the ensurepip
wheel while `ensurepip.version()` still succeeds. In that envelope the wrappers create the venv
with `--without-pip`, then use base pip's `--python` option to install pip into the new venv. Pixi
is the final fallback when neither native path works. An explicit incomplete interpreter fails
with a prerequisite error.

Do not point one architecture, host, role, or outer-container envelope at another one's venv. The
default names include the role, sanitized short hostname, `uname -m`, and Python version. Override
the exact venv path only when you know the selected interpreter is executable in that role's
current root filesystem.

### A request remains queued

Check controller health, pool utilization, the request JSON, and the node-agent log. Common causes
are no joined node, insufficient GPU/CPU/memory capacity, an unavailable requested runtime adapter,
or an isolation policy the node did not advertise after its launch probe.

Interpret `QUEUED (pool_empty)` as "no node has announced" and `QUEUED (node_probing)` as "a node is
still proving its runtimes." Confirm `active_nodes`, `gpus_schedulable`, and the matching
`node_states` entry across at least one heartbeat interval before diagnosing a scheduler stall.

### A bind source is missing or the workload sees a read-only filesystem

The source must exist on the selected GPU node, not merely on the submitter. Stage it to a
node-visible path or use shared storage. Add `:rw` only for directories the workload must modify;
keep source code or datasets `:ro` when writes are unnecessary. `--session-id` does not create or
mount a directory.

### Apptainer reports a user-namespace or FUSE error

Use a node/outer container that permits the required namespaces and mounts. If `/dev/fuse` is not
available, set `CAPE_APPTAINER_UNSQUASH=1` only when unsquash execution is appropriate for that
node. The join-time runtime probe is authoritative: unsupported isolation policies are not
advertised to the controller.

### Bubblewrap lists a GPU but CUDA is unavailable

Do not treat a package smoke, device node, `nvidia-smi`, or NVML result alone as proof. Read the
node registration's per-adapter capability JSON. A GPU request is eligible only when its selected
host policy appears in `gpu_isolation_policies`; CPU-only host work can still use a policy that is
absent there. Some nested environments cannot create the required fresh procfs. CAPE now fails that
host-policy probe closed; it never binds outer `/proc`, even read-only, because that exposes foreign
processes and can also break CUDA thread setup. Use an envelope that can create fresh PID/proc, or
the effective-UID-0 fresh-proc fallback that explicitly drops all capabilities. Final operator
acceptance runs `nvidia-smi`, `cuInit`, and then a real CUDA kernel.

### Local log commands cannot find logs

Query commands are remote, but `controller-log`, `node-log`, `command-log`, and `session-log` are
filesystem-local. Run the log command on a host that can read the file and set
`CAPE_STATUS_CONTROLLER_LOG`, `CAPE_STATUS_NODE_LOG`, `CAPE_STATUS_AGENT_LOG_DIR`, or
`CAPE_STATUS_BUNDLE` there.

### A controller namespace already exists

Use a new state directory/admin file for a new instance. To restore an existing
instance, retain the original state and admin credential. CAPE_REPLACE_EXISTING=1
replaces wrapper-owned logs/evidence, not the persistent instance or its credentials.
Regular users are not recreated on restart.

On another host, securely provision your own existing credential or issue an
additional token authenticated by it. Do not anonymously re-register the same
username just because that host has no local credential file yet.

### A re-registered node immediately exits after authoritative shutdown

Inspect the wrapper's `joining ... allocation=...` line. If `CAPE_ALLOCATION_ID` or
`ALLOCATION_ID` repeats an id that `retire_node.sh` already removed, a new enrollment cannot reuse it. The original retired agent exits 0
after authoritative shutdown; a fresh contribution must use fresh identities. Unset the override to use the fresh
per-invocation default, or supply a new externally managed allocation id. Do not delete the
tombstone or weaken controller removal semantics.

## Workspace capacity maintenance

Keep the explicit controller URL, implementation selector and individual credential files.
The workspace manager supports command-boundary growth and offline reclaim/shrink;
these are separate CLI operations, not extra shell text inserted into submitted commands.

```bash
"${CAPE_BIN:-rscape}" admin set-workspace-config \
  --controller-url "$CAPE_URL" --token-file "$CAPE_ADMIN_TOKEN_FILE" --pool "$CAPE_POOL" \
  --auto-grow on --grow-threshold-percent 80 --grow-target-percent 60
"${CAPE_BIN:-rscape}" workspace grow "$WORKSPACE_ID" --size-gib 20 \
  --controller-url "$CAPE_URL" --token-file "$CAPE_USER_TOKEN_FILE"
"${CAPE_BIN:-rscape}" workspace status "$WORKSPACE_ID" \
  --controller-url "$CAPE_URL" --token-file "$CAPE_USER_TOKEN_FILE"
```

Use the actual workspace ID from request status and an owner token for user maintenance.
Automatic growth and on-release reclaim default off; enabling them is an explicit admin
choice. Grow waits for the current command to finish. Reclaim/shrink require a released
workspace, and shrink needs an explicit target. Poll the operation ID to completion;
do not treat an accepted request as completed maintenance. `workspace recover` handles
an interrupted transaction. The original command is never automatically retried.

`prerequisites.sh` also resolves resize2fs/dumpe2fs and the copy/checksum tools. Actual
maintenance readiness is established by the agent's grow/reclaim/shrink probe, not by
package installation alone. See [workspace operations](../../../docs/workspaces.md#offline-growth-reclaim-and-shrink)
for policy flags, backup overhead, stateful-runtime restarts and recovery limits.

## Shutdown

Prefer `retire_node.sh` for a legible logical drain/removal. The default foreground
`register_node.sh` cleans local sessions and exits 0 after its next authoritative heartbeat; verify
both that exit and final zero schedulable capacity, then interrupt `controller.sh` with `Ctrl-C`.
Only an agent explicitly started with `CAPE_STAY_AFTER_REMOVAL=1` still needs an operator signal.
Graceful `retire_node.sh`, `status.sh`, and `concurrency.sh` do not perform implicit
request cancellation, signal a remote registration process, or release a cluster
allocation. Use `make shutdown` for the locally recorded background roles;
allocation release remains the operator's responsibility.
