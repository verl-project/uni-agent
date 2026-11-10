# Optional DCGM GPU metrics

Both `pycape` and `rscape` expose the configuration/query interfaces below. Metrics
are opt-in at both controller and node enrollment. The native rscape shared helper
collects DCGM fields for enrolled GPU UUIDs and writes directly to Prometheus;
Python agents use the same helper. Disabled Python operation does not require it.
There is no implicit NVML or nvidia-smi utilization polling.

`bash install.sh` installs missing DCGM, Prometheus and Telegraf tools as well as the
normal runtime dependencies, and rebuilds rscape from the current source. Supported
Linux packages/archives are architecture-checked; static downloads are checksum-checked.
DCGM packages are extracted privately, without installing GPU drivers or enabling
system services. Installed tools alone do not certify device access or network readiness.

## Instance lifecycle

Metrics are disabled by default. `controller init --metrics` starts an instance-local
Prometheus, and `controller start` restores the enabled configuration and reuses that
process. `controller start --metrics` can also enable it on an existing instance.
Normal SIGINT/SIGTERM shutdown stops this instance's Prometheus without deleting data.
Missing binaries make monitoring unavailable, not the scheduler or identity database.

The executable is found as `prometheus` on PATH; `CAPE_PROMETHEUS_BIN` can override it.
The private persistent directory is `prometheus/` beside the controller state directory.
Use separate deployment parents for separate schedulers. Startup serializes ownership,
checks the scheduler identity, and recovers a running process even if publication of
`process.json` was interrupted. It does not adopt another scheduler's directory.

The listener uses an available port and publishes a hostname URL. The receive endpoint
is unauthenticated and binds to the host network; use only the agreed trusted network.
Hostname discovery does not open a firewall or make an unreachable host reachable.
The scheduler queries the local Prometheus independently of its mutation lock.

Initial retention is seven days with a 1 GB TSDB retention target. This is not a hard
filesystem quota: database overhead/WAL and deletion timing require storage headroom.

## Configuration

```sh
rscape admin metric get --controller-url "$CONTROLLER" --token-file admin.json
rscape admin metric set --controller-url "$CONTROLLER" --token-file admin.json \
  --interval-seconds 5 --basic on --profiling off --expected-version 1
```

`set` accepts `--enabled on|off`, `--basic on|off`, `--profiling on|off`,
`--interval-seconds` (1–3600), and `--stale-periods` (1–120). At least one change is
required. `--expected-version` is optional optimistic concurrency control.
An identical effective configuration is a no-op; real changes increment a durable
version and write an audit event.

Without a node target, commands access the scheduler default. `--node-id ID` selects
the whole shared metrics service registered by that node, not just its enrollment.
A per-service override is a full configuration snapshot; later default changes do not
overwrite that snapshot. A node without a registered metrics service cannot be targeted.
Only an effective admin token may modify configuration, including for an admin user
whose regular token deliberately lacks admin capability.

## Queries

```sh
rscape metric status --controller-url "$CONTROLLER" --token-file user.json
rscape metric latest --controller-url "$CONTROLLER" --token-file user.json \
  --metric sm_active_ratio --pool shared
rscape metric history --controller-url "$CONTROLLER" --token-file user.json \
  --metric memory_used_bytes --gpu-uuid GPU-EXAMPLE \
  --start 1793900000 --end 1793900300 --step 5
```

Times are Unix seconds. Latest/history accept `--node-id`, `--pool` and `--gpu-uuid`.
Known metrics: `gpu_utilization_percent`, `memory_utilization_percent`,
`memory_used_bytes`, `memory_total_bytes`, `temperature_c`, `power_watts`,
`sm_active_ratio`, `sm_occupancy_ratio`, `tensor_active_ratio`, `dram_active_ratio`.
The GPU/memory utilization percentages measure activity, not peak compute/bandwidth
efficiency. Memory used/total are capacity in bytes. Profiling ratios are 0–1.

Queries require a normal CAPE user credential, restrict UUIDs to the requested
enrolled-resource scope, and never accept arbitrary PromQL or a caller-supplied backend
URL. History is limited to 31 days/request, step >=1 second and 100,000 estimated points;
actual result size is bounded too. Unknown/unsupported data are not replaced with zero.
Responses distinguish `disabled`, `unavailable`, `missing` and `ready`; nonfinite values
are returned as JSON null. Each sample carries its own expiry based on the collection
interval and stale-period settings at collection time. Queries mask expired samples;
changing the interval does not retroactively fill historical gaps.

HTTP interfaces: GET `/v1/metrics/status`, GET `/v1/metrics/query`, and admin actions
`metric-get`/`metric-set` through POST `/v1/identity/admin`.

## Agent control

An authenticated agent can report `status.metrics` in an existing heartbeat:
`service_id`, `applied_version`, and `state` (`pending`, `applied`, `error`). The response
includes `metrics_control` with desired configuration/version and receiver URL. Invalid
optional metrics metadata does not break an otherwise valid heartbeat or command mailbox.
POST `/v1/agent/metrics/control` offers discovery, location lease renewal and the
same read/acknowledgment boundary for the per-agent worker. It cannot set configuration.
Retired/purged agents and wrong epochs cannot retrieve control. Credentials are not
returned in status, receipts or queries. Unchanged receipts do not rewrite SQLite.

Enroll using the normal node parameters plus `--metrics`. The first enabled enrollment
starts one helper for the host/boot and scheduler identity. Other enrollments reuse it,
including other Linux UIDs when the endpoint is reachable. Registrations expire after
30 seconds; collection uses their GPU UUID union, without duplicate samples per pool.
No active registrations means no device sampling. Agents poll outbound for control;
neither scheduler nor Prometheus needs to open a connection to the node.

The shared helper receives no agent token. It reads only public configuration through
GET `/v1/metrics/service-config`; mutation still requires an admin credential. Local
registration and snapshots are unauthenticated under the trusted-network model.
Location leases prevent independently isolated network namespaces from simultaneously
collecting under the same service identity. Recovery can take a lease-expiry interval.

Temporary discovery files live under `/tmp/cape-metrics-<service-id>` (override parent
with `CAPE_METRICS_RUNTIME_DIR`). Sticky directories and public read-only lock/metadata
files permit cross-UID discovery without exposing credentials. This is not historical
storage. It does not make a process survive Slurm allocation termination.

The collector uses `dcgmi dmon`, not CUDA workloads. An automatically selected existing
hostengine must pass a client compatibility check; otherwise the helper starts an owned
foreground hostengine on loopback. Explicit `CAPE_DCGM_HOST` selects an operator-managed
endpoint without automatic replacement. `CAPE_DCGMI_BIN` overrides the client executable.
Device permissions and profiling support remain environment requirements; missing
metrics are unavailable, not zeros. Never broaden device permissions to make probes pass.

Telegraf performs bounded one-shot remote-write serialization. Failed batches are
dropped, with no node-side WAL or historical replay. Central receiver outages therefore
produce gaps. `CAPE_TELEGRAF_BIN` selects the sender binary; Python agents can select
the native helper using `CAPE_METRIC_HELPER_BIN`. Local status/latest access needs no token:

```sh
rscape metric status --local-url http://NODE:PORT
rscape metric latest --local-url http://NODE:PORT --metric memory_used_bytes
```

These local endpoints contain GPU telemetry, not command data or execution credentials.
They do not provide a security boundary against malicious peers on the trusted network.

## Resource bounds

The helper keeps at most 1,024 latest GPU snapshots and 4,096 registrations, not a
history database. A serialized send batch is limited to 10,000 samples and 4 MiB;
overflow evicts the oldest complete GPU sample, including its expiry marker. There
is one in-flight sender, with a two-second HTTP timeout and three-second process
deadline. DCGM reads use at most eight concurrent child processes with three-second
deadlines. Slow cycles do not spawn overlapping cycles to catch up.

These are application data/concurrency limits, not hard OS CPU/RSS quotas. DCGM,
Prometheus and Telegraf have their own runtime overhead. Central disk retention is a
target, not a filesystem quota; provision storage headroom and monitor the deployment.

## Validation scope

Tests cover strict contracts, paired CLI/API authorization, real Prometheus lifecycle,
bounded delivery and outage gaps, native shared-service reuse, and opt-in real DCGM
and cross-UID scenarios. Live tests require explicit environment opt-ins and dependencies;
skipped live tests are not evidence of device readiness. No website changes are included.
