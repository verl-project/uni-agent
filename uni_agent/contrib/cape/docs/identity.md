# Persistent identities and resource ownership

Both `rscape` and `pycape` implement this protocol natively. CAPE is usable without
a website, SSO, or an external identity provider. Examples use `rscape`; substitute
`pycape` to select Python. The controller, contributors and clients may run on
different machines with an authenticated, trusted transport between them.

## Initialize once; start or recover explicitly

Create a private credential directory on the controller host:

```sh
mkdir -m 700 controller-secrets
rscape controller init --state-dir controller-state --user admin \
  --token-out controller-secrets/admin.json
rscape controller start --state-dir controller-state \
  --token-file controller-secrets/admin.json --host 127.0.0.1 --port 8765
```

`init` refuses an existing instance directory. It generates a bootstrap admin
credential, saves its secret locally, and creates a stable scheduler identity.
The CLI prints identifiers and the credential path, not the secret. This credential
is not a single-use login: its plaintext is never retrievable from the controller,
but the saved file can authenticate subsequent starts and administrative calls.

`start --state-dir` requires existing, valid state, OS access, and an enabled admin
identity authenticated with an **admin-kind** credential. A regular credential
belonging to the admin account does not suffice. A missing or incomplete instance
is not initialized automatically. A process lock excludes a second controller
using the same state directory. Do not run cloned copies of a live instance.

Authority, audit records and the existing event journal share `state.sqlite3`.
Keep the whole state directory on durable **node-local** storage with verified SQLite/locking
semantics. Live SQLite WAL state on NFS or other network filesystems is unsupported;
shared storage is suitable for stopped/consistent backups and logs, not the running
instance. See [SQLite's WAL filesystem requirements](https://sqlite.org/wal.html).
`/tmp` is only appropriate for disposable tests, not production recovery. Use SQLite backup/checkpoint procedures
or stop the controller before copying state: copying only a live WAL database's
main file is not a consistent backup. Back up credentials separately and securely.
Losing all usable credentials has no plaintext-recovery shortcut.

Persistent mode barriers command delivery and completion acknowledgements before
returning them to an agent. Recovery still reconciles live agents; it does not
migrate or automatically rerun uncertain commands. Node retirement requires
cleanup evidence, including after controller restart. CAPE does not provide
distributed failover or make a command's external side effects transactional.

## Users and personal credentials

On a user's machine, create a private directory and register a new identity:

```sh
mkdir -m 700 my-cape-secrets
rscape user token create --controller-url http://controller.example:8765 \
  --user alice --name laptop --token-out my-cape-secrets/alice.json
rscape user whoami --controller-url http://controller.example:8765 \
  --token-file my-cape-secrets/alice.json
```

Anonymous registration can only create a new regular identity, with membership in
`default`. It cannot take over an existing name or issue another user's tokens.
Anonymous token-registration attempts are limited to 64 per minute per controller
process. Protect the API with network admission controls appropriate to your
deployment; self-registered names are not independently verified corporate identities.

An authenticated user can create another regular credential without repeating a
username or changing identity/groups:

```sh
rscape user token create --controller-url http://controller.example:8765 \
  --token-file my-cape-secrets/alice.json --name workstation \
  --token-out my-cape-secrets/workstation.json
rscape user token list --controller-url http://controller.example:8765 \
  --token-file my-cape-secrets/workstation.json
rscape user token revoke TOKEN_ID --controller-url http://controller.example:8765 \
  --token-file my-cape-secrets/workstation.json
```

Use `--expires-at UNIX_SECONDS` for an expiry. The server stores only verifiers.
Regular credentials cannot list, revoke or otherwise operate on admin credentials,
even when owned by an admin account. The bootstrap admin credential cannot be
revoked through the regular-token rotation flow.

### Registration failures and recovery

The private JSON envelope contains both the secret and immutable retry parameters.
The client exclusively creates it with mode 0600 in an owner-only directory,
rejecting symlink traversal, and syncs file and directory before registration.
Do not edit, overwrite, commit, print or send this envelope to another person.
For authenticated issuance, the CLI first reads the caller's stable user ID and
includes it in the saved registration parameters. Enrollment similarly saves its
contributor ID. The server checks these targets against the authenticated caller,
so a request that never committed cannot be retried under a different identity.
These preflight reads do not introduce activation or additional manual steps.

- Exit 0: the registration acknowledgement matched the saved operation.
- Exit 1 with `outcome=rejected`: a new operation received an explicit rejection.
- Exit 3 with `outcome=unknown`: the result cannot be established, including a lost
  response. The token may already be active; its file is preserved.

Recover without generating a different secret or operation ID:

```sh
rscape user token resume --token-out my-cape-secrets/alice.json
```

The saved controller URL is binding. For a not-yet-committed additional-token
operation, also supply an existing valid `--token-file`. Recovery never revives a
revoked token or disabled account. The presence of a local file alone is not proof
of successful registration. The normal path requires no activation file or second
manual activation command.

Caller commands accept either `--token-file` or `--token`; `CAPE_TOKEN` is also
available for caller credentials. Direct token arguments can be visible in process
listings and shell history. Credential environment aliases are cleared before
workloads are launched. Use TLS or a trusted tunnel/network: plain HTTP does not
encrypt bearer credentials. A file changes local exposure, not network encryption.

## Pools and permission groups

Only an effective admin credential can create or manage pools and group policy:

```sh
rscape admin pool create pool-a --max-nodes 8 \
  --controller-url http://controller.example:8765 --token-file controller-secrets/admin.json
rscape admin pool group set default \
  --controller-url http://controller.example:8765 --token-file controller-secrets/admin.json
rscape admin pool group set access-a --pool pool-a \
  --controller-url http://controller.example:8765 --token-file controller-secrets/admin.json
rscape admin user list \
  --controller-url http://controller.example:8765 --token-file controller-secrets/admin.json
rscape admin user edit USER_ID --group access-a \
  --controller-url http://controller.example:8765 --token-file controller-secrets/admin.json
```

`default` initially grants all pools. `group set default` above deliberately changes
it to no pools; `--all-pools` explicitly grants all current and future pools.
Repeat `--pool` to grant a finite set. Groups store stable pool IDs, not names.
Deleting and recreating a pool with the same name does not inherit old grants.

Repeat `--group` to replace a user's complete membership set; use `--clear-groups`
for no memberships. Effective submission permission is the union of current group
policies. `default` cannot be deleted; other groups can be removed even with members,
without assigning those users back to `default`. Updates affect existing members.

`admin user edit --disable/--enable` changes account availability. `admin user remove`
soft-disables the identity, preserving ownership, credentials' revocation records,
and historical results; it does not erase images or cancel commands. There is no
`admin user add`: first-use registration creates regular identities.

`admin pool edit POOL` supports `--name`, `--max-nodes`/`--unlimited-nodes`, and
`--pool-mode`, `--gpu-slots`, `--placement`. Pool execution policy changes require an
empty pool. Node limits cannot be lowered below active enrollment count. Pool
creation accepts the same execution-policy options. Nodes adopt administrator
policy; a contribution cannot change an entire pool's policy.

## Contribute a node without Join Tokens or approval

On the node, use the contributor's personal credential once to enroll:

```sh
mkdir -m 700 node-secrets
rscape node enroll --controller-url http://controller.example:8765 --pool pool-a \
  --token-file my-cape-secrets/alice.json --node-token-out node-secrets/agent.json \
  --cpu-cores 8 --memory-gb 32 --gpu-uuid GPU-REPLACE-WITH-PHYSICAL-UUID \
  --runtime-adapter bubblewrap --state-path node-secrets/agent-state.json \
  --log-dir node-logs --runtime-dir node-runtime
```

Repeat `--gpu-uuid` for additional contributed GPUs. Omitting GPUs contributes CPU
capacity only. The CLI checks local CPU/memory/GPU availability before enrollment;
runtime and workspace probes still determine actual schedulability. A controller
cannot remotely attest an untrusted host: contributors/agents remain part of the
execution trust boundary, and local-process is not a security sandbox.

The controller binds the contribution to the authenticated owner and issues no
Join Token or approval workflow. A separate client-generated agent secret is saved
before its verifier is registered. It is bound to node, pool, allocation and process
epoch; personal tokens cannot impersonate agents and agent credentials cannot
submit user work or read private requests. Disabling the owner does not cut off the
agent's existing-work cleanup channel.

Enrollment starts the foreground agent. Without a runtime filter, it probes
Apptainer, Bubblewrap and Podman and advertises only their proven capabilities.
Missing or failed automatic probes do not suppress healthy peers; if none pass,
the node cannot become ACTIVE. Repeat `--runtime-adapter` to require particular
adapters; a failed explicitly requested adapter is an error. `local-process`
requires explicit selection and is never an automatic fallback. Apptainer and
Podman default to `docker://docker.io/library/ubuntu:24.04` for probing; image
overrides remain available. `node enroll` is the only node CLI entrypoint for starting an agent.
Recover an interrupted enrollment using the same `--node-token-out` and `--resume`,
with the owner's personal credential and matching pool/resources. Do not launch two
agents with the same saved enrollment, or reuse it after retirement; a new enrollment
gets a new node/allocation identity. Reconnecting to an existing process is not
permission to take over another node's name.
An uncertain enrollment POST returns exit 3 with `outcome=unknown`; keep the saved
file and resume. No replacement secret is generated behind the user's back.

## Submit, observe and retire

```sh
rscape user pools --controller-url http://controller.example:8765 --token-file my-cape-secrets/alice.json
rscape user nodes --pool pool-a --controller-url http://controller.example:8765 --token-file my-cape-secrets/alice.json
rscape run --pool pool-a --session-id example --gpus 1 --runtime-adapter bubblewrap \
  --isolation-policy host-stateless --controller-url http://controller.example:8765 \
  --token-file my-cape-secrets/alice.json -- python -c 'print("hello")'
```

Identity is derived from the token; `--user` is not needed for persistent-mode
submission. Empty pools are `unavailable` but accept authorized queued work and
new contributions. Private `status`, `logs`, `wait`, `cancel`, sessions and
workspaces remain owner-only, with the effective-admin exception. Losing submission
permission does not revoke access to one's existing resources. Public inventory
contains resources and allowlisted GPU telemetry, not other users' identities,
session IDs, commands, environment, log paths or workspace paths.

Queued requests are checked again before assignment. Account/group changes may fail
unassigned work with `permission_revoked`; already assigned work is not automatically
cancelled or moved. Token revocation/expiry blocks new caller operations, not already
accepted commands. Pool drain fails queued work with an explicit reason.

```sh
rscape node drain NODE_ID --controller-url http://controller.example:8765 --token-file my-cape-secrets/alice.json
rscape node exit NODE_ID --controller-url http://controller.example:8765 --token-file my-cape-secrets/alice.json
rscape admin pool drain pool-a --controller-url http://controller.example:8765 --token-file controller-secrets/admin.json
rscape admin pool remove pool-a --controller-url http://controller.example:8765 --token-file controller-secrets/admin.json
```

An effective **admin token** can forcibly retire a node already marked `LOST`:

```sh
rscape admin node purge NODE_ID --reason "Slurm allocation expired" \
  --controller-url http://controller.example:8765 --token-file admin.json
```

This does not wait for agent cleanup. It removes the membership from default node
inventory and capacity accounting, retaining a durable tombstone and audit reason.
ACTIVE/UNHEALTHY/DRAINING nodes are rejected; there is no force bypass. Ordinary
contributors and an admin account using a regular token cannot perform purge.
Repeating purge returns the original result, without overwriting the original reason.

Old agent credentials cannot register, heartbeat or report after purge, including
after controller restart. Replaying the original enrollment operation returns its
historical receipt, not a fresh authorization. Use a fresh node/allocation/credential
identity to contribute again. Do not downgrade controllers after recording purges.

Purge is **not** proof of remote process termination: it does not contact Slurm, kill
processes, delete logs, erase workspace images, or assert successful unmount/save.
The response includes `cleanup_confirmed: false` and `quarantined_gpu_uuids`.
Terminal request results are preserved; remaining assignments are failed rather
than retried. Workspace records remain, marked `node_retired: true`; existing
uncertain maintenance operations keep their fences.

The advertised physical GPU UUIDs are conservatively quarantined across other
agents/pools and fresh enrollments. New work cannot use them; already accepted work
on another contribution is not cancelled. A visible GPU with this restriction is
shown as `QUARANTINED`. Removing the old pool does not clear the quarantine. Purge cleans
membership, **not** permission to immediately reuse uncertain hardware. Retained
node-local data remains subject to allocation/node loss as before.

After confirming that the old agent and its workloads stopped, an effective admin
can release **that old node's quarantine source**:

```sh
rscape admin node release-quarantine OLD_NODE_ID --confirm-stopped \
  --reason "Confirmed old allocation ended and its processes stopped" \
  --controller-url http://controller.example:8765 --token-file admin.json
```

Both the explicit confirmation flag and a nonblank reason are required. This is an
administrator assertion, not automatic verification against Slurm. Only purged nodes
are eligible. A regular credential, including an admin account's regular token, cannot
release quarantine. No old node, credential, epoch, command or workspace is restored.
Workspace `cleanup_confirmed` stays false, and uncertain image-maintenance fences remain.

Overlapping sources are independent: releasing node A does not release node B's fence
on the same GPU UUID. The response contains an immutable `release_receipt` (first
actor/time/reason/confirmation), `released_source_gpu_uuids`, `still_quarantined_gpu_uuids`
and `unquarantined_gpu_uuids`. GPU lists describe current quarantine within this source's
scope; retries retain the receipt without duplicating the audit event, but the lists
may change after another source is released. Unquarantined does not mean lease-free.

An already enrolled replacement agent needs no re-enrollment. Once no source still
quarantines a GPU, subsequent scheduling can assign it subject to normal availability,
permissions and leases. Release survives controller restart, and replaying the original
purge does not re-quarantine the source. The old agent credential stays revoked permanently.

Only the contributor or effective admin can change a node. `drain` stops new
assignments, waits for all already-assigned commands (including unacknowledged
assignments), then requires fresh process/runtime/mount cleanup evidence before
retirement. `exit` requires verified idle/clean state. `drain --force` explicitly
cancels remaining work, potentially including another user's command on your node,
but grants no access to that user's data and does not bypass cleanup confirmation.
Offline/lost nodes are not assumed clean. The command returns current lifecycle
state; poll inventory until retirement rather than treating HTTP success as cleanup.

`node edit NODE_ID --cpu-cores N --memory-gb N` reduces a contribution; repeat
`--gpu-uuid` to keep a smaller GPU set, or use `--no-gpus`. Reductions stop new
assignments, wait for existing work and cleanup, then reactivate the same node with
its reduced capacity. They preserve workspace data and ownership. Adding capacity
requires a fresh enrollment and local resource verification. Resume an existing
agent with its original saved enrollment arguments: re-registration applies the
controller's reduced effective capacity and cannot restore the old limit.

Pool drain is admin-only, closes submission/enrollment while draining, and reopens
as an empty unavailable pool after cleanup. Removal requires emptiness. Neither
action deletes retained workspace images. Sensitive administrative access and
mutations are audited without secrets; inspect recent records with
`admin audit --limit 100`. Retain the instance database under your audit/backup policy.

## Supported CLI boundary

`controller init/start --state-dir` is the only CLI controller lifecycle. Missing
state or invalid credentials fails closed; there is no unauthenticated startup mode.
The old `--auth-file`, `--journal-path`, `node join`, `--join-token`, offline
`admin add-*-token`, and allocation/pool-config legacy commands are removed.
Use `user token`, `node enroll/drain/exit`, and `admin pool` instead. Existing old
auth files and website/SSO stores are not implicitly imported. Keep old instance
backups; a new `init` creates a different scheduler, not a migration of old data.
