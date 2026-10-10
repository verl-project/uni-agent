# Agent-Aware KV Cache Offload

Agent workloads often return to the same conversation after executing tools.
Agent-aware KV offload lets the [Gateway](gateway-and-trajectories.md) attach
runtime state to each generation request so a vLLM connector can compute cache
retention priorities when admitting and evicting KV blocks. Gateway supplies
state; the connector owns scoring, leases and admission.

The feature is enabled by selecting the custom connector spec. It targets the native CPU offload interfaces
in **vLLM 0.23.0**. The [Agent Aware Router](agent-aware-router.md) chooses an
inference replica; this policy chooses which blocks to admit to, and retain in,
that replica's CPU cache. The two mechanisms serve different purposes.

## How It Works

```text
Gateway session
    -> attach agent runtime state for a capable backend
    -> vLLM OffloadingConnector
    -> PriorityCPUOffloadingManager: score once per request, assign lease, admit and evict
    -> native vLLM workers: GPU/CPU transfers
```

Hints are added to `SamplingParams.extra_args.kv_transfer_params.agent_hint`.
`extra_args` and `kv_transfer_params` are vLLM extension interfaces; `agent_hint`
is Uni-Agent's versioned state protocol. The Gateway builds hints before each
backend generation call without changing provider-visible sampling parameters.
External API inference that bypasses the Gateway does not generate these hints.

Version 2 hints describe prepared pre-generation state:

```json
{
  "schema_version": 2,
  "trajectory_id": "session-1",
  "tools_available": true,
  "has_active_chain": true,
  "received_tool_result": true,
  "context_tokens": 40000,
  "trajectory_capacity": 65536,
  "remaining_capacity": 20000,
  "rollback_applied": false,
  "active_chain_count": 1
}
```

`trajectory_id` identifies the Gateway session. `has_active_chain` indicates an
existing chain; `received_tool_result` means the current provider turn supplies
a tool result. Anthropic adapters preserve this fact when a tool-result image
is lowered to a template-facing user message. Historical tool results do not
mark a later ordinary user turn; direct normalized requests fall back to the
last message's `tool` role.
`tools_available` describes tool schemas, not actual tool use. Token counts
describe context and remaining trajectory budget. Unknown capacity is `null`
for both capacity fields. The payload contains no priority or deadline.

## Enable the Feature

Configure the vLLM connector to consume hints. Install Uni-Agent in the vLLM runtime as well as the
Gateway runtime so the engine can import the custom spec.

### Backend Capability

In verl-managed runs, the framework recognizes the custom connector/spec
combination below and exposes `supports_agent_runtime_hint = True` on the backend.
Gateway automatically supplies state for capable backends. Other connectors and
SGLang do not receive generated hints. A custom backend can expose this boolean
capability explicitly; its declaration takes precedence over automatic detection.
Direct Python users can wrap a configured backend with
`uni_agent.gateway.backend.AgentHintBackend`.

There is no Gateway cache-policy configuration or separate enable flag. In a
verl-managed configuration, use this engine section:

```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    engine_kwargs:
      vllm:
        kv_transfer_config:
          kv_connector: OffloadingConnector
          kv_role: kv_both
          kv_connector_extra_config:
            spec_name: PriorityGPUOffloadingSpec
            spec_module_path: uni_agent.gateway.kv_offload.priority_policy
            cpu_bytes_to_use: 68719476736
            agent_hint_config:
              active_lease_seconds: 300.0
              min_offload_priority: 30
              admit_without_hint: false
```

### vLLM Connector Configuration

Pass the following object as the engine's `kv_transfer_config` (the
`--kv-transfer-config` JSON argument when launching vLLM directly). With a
verl-managed engine, forward it through that launcher's vLLM engine configuration.
It belongs to the engine configuration.

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "PriorityGPUOffloadingSpec",
    "spec_module_path": "uni_agent.gateway.kv_offload.priority_policy",
    "cpu_bytes_to_use": 68719476736,
    "agent_hint_config": {
      "active_lease_seconds": 300.0,
      "min_offload_priority": 30,
      "admit_without_hint": false
    }
  }
}
```

This example selects a 64 GiB CPU buffer; size it for the host memory available
to your deployment. `cpu_bytes_to_use` is required by the native CPU spec. Native
vLLM still owns block sizing and transfer handlers; this policy does not replace
them. See the [vLLM 0.23.0 CPU spec](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/v1/kv_offload/cpu/spec.py).

All policy settings belong to `kv_connector_extra_config.agent_hint_config`:

| Policy option | Default | Meaning and tuning |
|---|---|---|
| `active_lease_seconds` | `300.0` | Finite positive claim lifetime. Increase for longer gaps between turns; decrease to expire stale priorities sooner. It does not reserve memory. |
| `min_offload_priority` | `0` | Integer in `[0, 100]`; minimum score for a store. Increase to restrict admission to more reusable contexts; decrease to admit more. Existing blocks remain readable. |
| `admit_without_hint` | `true` | Boolean; permits requests without a valid, unexpired hint, subject to the minimum score. Such requests have score `0`. |

Configure these before engine launch; omit `agent_hint_config` for the defaults.
Use actual YAML booleans and numbers. Strings, boolean numeric values, NaN,
infinity and non-positive lease durations are rejected.

Native `store_threshold` and `max_tracker_size` stay directly under
`kv_connector_extra_config`. Through the spec, their defaults are `0` and
`64000`. Values below `2` disable the native lookup-count filter.

The example's admission settings intentionally differ from defaults. Setting
`admit_without_hint: true` alone does not admit unhinted requests if
`min_offload_priority` is greater than zero. Invalid or expired hints follow
these same no-hint rules. The custom manager always uses `PriorityCachePolicy`;
the native `eviction_policy` option does not select ARC for this spec.

## Dynamic Priority

Static mode and its `priority_mode`, `priority` and `tool_priority` settings have
been removed. Fixed scores helped initial protocol testing but do not distinguish
requests using one connector's common configuration. Tests can inject fixed
scores internally; use the native vLLM CPU spec for baseline comparisons.

The connector sums the following components, then clamps the result to `[0, 100]`.
These are fixed heuristics, not learned probabilities or additional configuration
options. All inputs are available before generation.

| Component | Scoring rule |
|---|---|
| Continuation | `55` if the current provider turn supplies a tool result (including Anthropic tool-result images); otherwise `40` for an existing chain, `25` if tools are available, or `10`. Only one case applies. |
| Context/recomputation cost | `0`, `5`, `10`, `15`, `22`, or `30` for context lengths below 1,024; 4,096; 16,384; 32,768; 65,536; or at least 65,536 tokens, respectively. |
| Remaining capacity | `-15` if fewer than 1,024 tokens or 5% remain; otherwise `+10` if at least 8,192 tokens and 25% remain; otherwise `0`. Unknown capacity contributes `0`. |
| Chain state | `+5` when last-assistant rollback was applied; `-5` when more than one chain was active during input preparation. These adjustments can combine. |

For example, a tool-result continuation with 40,000 context tokens, 20,000 tokens
remaining out of 65,536, rollback applied and one active chain scores
`55 + 22 + 10 + 5 = 92`. The engine's threshold controls admission.
The manager caches one decision per engine request; a new generation receives a
fresh decision. Lookup, touch and store operations do not repeat scoring.

## Lease and Shared-Prefix Semantics

A lease bounds the lifetime of a priority claim. It is not a lock, an eviction
prohibition or a guarantee that blocks stay cached. It starts when the engine
scheduler receives a new request; scheduler waiting and generation time consume
it. The deadline uses the engine process's monotonic clock, avoiding Gateway/engine
clock skew. Repeated operations on one request never extend its deadline.
Decisions are released on request completion; block metadata retains its claim.

For a cached block touched by another request:

- A lower-priority hint updates recency but cannot extend a live higher-priority
  claim. The lower claim is not queued for activation after expiry.
- An equal-priority hint can extend the deadline.
- A higher-priority hint replaces both priority and deadline, without inheriting
  an older, lower-priority claim's longer deadline.
- An expired incoming hint updates recency only. After an existing claim expires,
  a new unexpired hint can replace it.

The manager admits new stores only after applying hint validity and admission
rules. Busy blocks remain protected by native vLLM reference counting regardless
of their lease state.

## Eviction and Limits

Among blocks that are not busy or explicitly protected by the native manager,
the policy selects expired claims first, then lower priorities, then LRU ties.
When touching an ordered prefix, it iterates in reverse, matching native vLLM
LRU so suffixes are discarded first when priority and expiry status tie.

This favors prefix heads but does not guarantee a contiguous prefix across
different priorities, expiry states or protected blocks. Admission is a fixed
threshold, not a comparison against each victim: an admitted lower-priority
request can still displace a higher-priority block if no cheaper victim exists.
Eviction uses an indexed heap ordered by expiry, priority and recency. A second
indexed heap tracks lease deadlines. Insert, touch and remove update these
indices in `O(log N)` time, with at most one record per block per index; repeated
touches do not accumulate stale records. Evicting `K` blocks checks candidates
in order and stops once enough eligible blocks are found, avoiding a full-cache
scan and sort on every replacement. Busy or protected candidates are skipped
and restored; if too few eligible blocks exist, no blocks are removed.

Newly expired leases are processed before selecting victims, once per deadline.
A large expiry batch or many busy/protected candidates can still increase one
eviction call's cost. Measure performance at the intended CPU-cache capacity
and workload, including touch overhead and expiry bursts.

The integration targets vLLM 0.23.0. It does not implement an SGLang offload policy
or an external-model-API cache. Native GPU/CPU transfer support and hardware
requirements still apply.

## Diagnostics and Validation

The manager tracks lookup hits, pending writes, misses, admission decisions and
transfer completions. The policy tracks eviction calls, evicted blocks and failed
eviction attempts. `debug_snapshot()` exposes these counters and a full cache
snapshot for explicit diagnostics; do not call it per block because it scans
the cache. Normal lookup and completion logging do not take full snapshots.
`PRIORITY_KV_LOG_LEVEL` controls the module's log level; per-block lookup events
are DEBUG, while the default level is INFO.

If no blocks are admitted, check backend capability, hint validity/expiry, the
minimum score, and the native `store_threshold`. If hits are poor, also check
prefix compatibility, routing, capacity pressure and whether leases expire before
the next turn. Raising a priority does not guarantee a hit or reserve memory.

Run the CPU CI selection with the project's test dependencies installed:

```bash
pytest tests/uni_agent/gateway/kv_offload -m "cpu and level0" -q
```

The suite includes isolated policy tests and real vLLM 0.23.0 manager lifecycle
tests. The latter skip on Windows and require the pinned engine on Linux. They
do not launch GPU workers or establish end-to-end transfer performance; validate
those separately in the deployment environment.

## Migration

Remove the old `agent_framework.kv_cache_offload` block. Move
`active_lease_seconds`, `min_offload_priority` and `admit_without_hint` to the
connector's `agent_hint_config`; remove `enable`, `priority_mode`, `priority` and
`tool_priority`. Old configuration locations are rejected with migration errors.

Upgrade Gateway and connector together. Version 1 hints carrying `kv_priority`
and `lease_until` are rejected; incomplete or unsupported state snapshots follow
the same no-hint admission rules. For direct vLLM OpenAI API requests, place
`kv_transfer_params` at the request body's top level; vLLM forwards it through
`SamplingParams.extra_args` internally.
