# Agent-Aware KV Cache Offload

Agent workloads often return to the same conversation after executing tools.
Agent-aware KV offload lets the [Gateway](gateway-and-trajectories.md) attach
reuse hints to each generation request so a vLLM CPU cache can prioritize these
requests when admitting and evicting KV blocks.

This feature is disabled by default. It targets the native CPU offload interfaces
in **vLLM 0.23.0**. The [Agent Aware Router](agent-aware-router.md) chooses an
inference replica; this policy chooses which blocks to admit to, and retain in,
that replica's CPU cache. The two mechanisms serve different purposes.

## How It Works

```text
Gateway session
    -> attach request priority and lease deadline
    -> vLLM OffloadingConnector
    -> PriorityCPUOffloadingManager: admission and eviction
    -> native vLLM workers: GPU/CPU transfers
```

Hints are added to `SamplingParams.extra_args.kv_transfer_params.agent_hint`.
They contain a schema version, session ID, priority, absolute lease deadline,
and whether tool schemas are available. The Gateway builds them before each
backend generation call without changing provider-visible sampling parameters.
External API inference that bypasses the Gateway does not generate these hints.

## Enable the Feature

Both sides must be configured: the Gateway generates hints, and the vLLM
connector consumes them. Enabling hints alone does not allocate a CPU cache or
enable GPU-to-CPU transfers. Install Uni-Agent in the vLLM runtime as well as the
Gateway runtime so the engine can import the custom spec.

### Gateway Configuration

Add this section to the verl-managed inference or training configuration:

```yaml
actor_rollout_ref:
  rollout:
    custom:
      agent_framework:
        kv_cache_offload:
          enable: true
```

This uses static priorities and a 300-second lease. All other Gateway parameters
are optional. To use dynamic scoring, add `priority_mode: dynamic`.

The complete set of options is shown below with its defaults:

```yaml
kv_cache_offload:
  enable: false
  priority_mode: static
  active_lease_seconds: 300.0
  priority: 50
  tool_priority: 90
```

| Parameter | Default | Meaning |
|---|---|---|
| `enable` | `false` | Boolean; enables Gateway hint injection. It does not configure the engine connector. |
| `priority_mode` | `static` | `static` selects configured priorities; `dynamic` computes a score from the request's trajectory state. |
| `active_lease_seconds` | `300.0` | Positive, finite duration in seconds. Each generation gets a deadline of current Gateway time plus this duration. Used in both modes. |
| `priority` | `50` | Integer in `[0, 100]`; static priority when no tool schemas are present. |
| `tool_priority` | `90` | Integer in `[0, 100]`; static priority when tool schemas are present. This does not require that the model actually calls a tool. |

The two static priorities are independent tuning knobs, not required inputs.
In dynamic mode they do not contribute to the score; omit them unless the same
configuration also needs static-mode settings. If supplied, they are still
validated. Use actual YAML booleans and numbers, not quoted strings or fractional
priorities. NaN, infinity and non-positive lease durations are rejected.

### vLLM Connector Configuration

Pass the following object as the engine's `kv_transfer_config` (the
`--kv-transfer-config` JSON argument when launching vLLM directly). With a
verl-managed engine, forward it through that launcher's vLLM engine configuration.
It belongs to the engine configuration, not the Gateway `kv_cache_offload` block.

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "PriorityGPUOffloadingSpec",
    "spec_module_path": "uni_agent.gateway.kv_offload.priority_policy",
    "cpu_bytes_to_use": 68719476736,
    "min_offload_priority": 30,
    "admit_without_hint": false
  }
}
```

This example selects a 64 GiB CPU buffer; size it for the host memory available
to your deployment. `cpu_bytes_to_use` is required by the native CPU spec. Native
vLLM still owns block sizing and transfer handlers; this policy does not replace
them. See the [vLLM 0.23.0 CPU spec](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/v1/kv_offload/cpu/spec.py).

| Connector option | Default | Meaning |
|---|---|---|
| `min_offload_priority` | `0` | Integer in `[0, 100]`; minimum score for admitting a store request. It does not prevent reads of already cached blocks. |
| `admit_without_hint` | `true` | Boolean; permits requests without a valid, unexpired hint, subject to the minimum score. Such requests have score `0`. |
| `store_threshold` | `0` through the spec | Native vLLM lookup-count filter. Values below `2` disable it; larger values require repeated lookups before storage. |
| `max_tracker_size` | `64000` | Native vLLM lookup tracker capacity when the count filter is active. |

The example's admission settings intentionally differ from defaults. Setting
`admit_without_hint: true` alone does not admit unhinted requests if
`min_offload_priority` is greater than zero. Invalid or expired hints follow
these same no-hint rules. The custom manager always uses `PriorityCachePolicy`;
the native `eviction_policy` option does not select ARC for this spec.

## Static and Dynamic Priority

Static mode is useful for a controlled baseline: requests with tools get
`tool_priority`, and requests without tools get `priority`.

Dynamic mode sums the following components, then clamps the result to `[0, 100]`.
These are fixed heuristics, not learned probabilities or additional configuration
options. All inputs are available before generation.

| Component | Scoring rule |
|---|---|
| Continuation | `55` if the last message is a tool result; otherwise `40` for an existing chain, `25` if tools are available, or `10`. Only one case applies. |
| Context/recomputation cost | `0`, `5`, `10`, `15`, `22`, or `30` for context lengths below 1,024; 4,096; 16,384; 32,768; 65,536; or at least 65,536 tokens, respectively. |
| Remaining capacity | `-15` if fewer than 1,024 tokens or 5% remain; otherwise `+10` if at least 8,192 tokens and 25% remain; otherwise `0`. Unknown capacity contributes `0`. |
| Chain state | `+5` when last-assistant rollback was applied; `-5` when more than one chain was active during input preparation. These adjustments can combine. |

For example, a tool-result continuation with 40,000 context tokens, 20,000 tokens
remaining out of 65,536, rollback applied and one active chain scores
`55 + 22 + 10 + 5 = 92`. The same threshold configured on the engine controls
admission for either static or dynamic scores.

## Lease and Shared-Prefix Semantics

A lease bounds the lifetime of a priority claim. It is not a lock, an eviction
prohibition or a guarantee that blocks stay cached. The deadline starts before
generation, so queueing and generation time consume the lease. Keep Gateway and
engine host clocks synchronized because deadlines use wall-clock timestamps.

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
Eviction currently scans and sorts eligible blocks; performance should be
measured at the intended CPU-cache capacity and workload.

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

If no blocks are admitted, check both configuration layers, hint expiry, the
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

## Python configuration

The framework maps the YAML settings above into a frozen `KVCacheHintConfig`
from `uni_agent.gateway.config`. `enable` maps to `enabled`, and
`active_lease_seconds` maps to `lease_seconds`; the YAML keys remain unchanged.
`GatewayActorConfig.kv_cache_offload_config` holds this object, which the gateway
passes directly to each session. The hint config owns validation for its fields:
`enabled` must be a boolean, the mode must be `static` or `dynamic`, the lease
must be finite and positive, and both priorities must be integers in `[0, 100]`
(booleans are not accepted as numbers).

For direct Python construction:

```python
from uni_agent.gateway.config import GatewayActorConfig, KVCacheHintConfig

gateway_config = GatewayActorConfig(
    tokenizer=tokenizer,
    kv_cache_offload_config=KVCacheHintConfig(
        enabled=True,
        priority_mode="static",
        lease_seconds=300.0,
        priority=50,
        tool_priority=90,
    ),
)
```

All hint fields are optional. The default `KVCacheHintConfig()` disables hint
injection; enabling it with the remaining defaults is sufficient for static mode.
