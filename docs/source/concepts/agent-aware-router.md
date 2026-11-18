# Agent Aware Router

The agent-aware router is an external plugin load balancer: verl loads it from a router YAML via `rollout.router_config_path` and instantiates the `router_class` FQN declared in that YAML (`uni_agent.agent_aware_router.balancer.KVCAwareBalancer`) with the YAML as its config. Its user-tunable knob is `load_threshold` (YAML key `strategy.load_threshold`, overridable at runtime via `rollout.custom.agent_framework.router`). To run it, see [Run the Agent Aware Router](../quickstart/agent-aware-router.md).

## Motivation

### Workload characteristics of agentic RL rollout

The rollout phase of agentic RL differs substantially from traditional single-turn RL rollout in workload shape:

- **Multi-turn dialogue**: One sample maps to one agent session, often spanning hundreds of tool-calls per session, with a single prefill frequently reaching tens of thousands of tokens. Turn counts vary widely across sessions — some converge in a few turns, others take dozens or hundreds.
- **Strong prefix reuse**: Multiple requests within the same session share a continuously growing prefix history. If each turn can hit the KV-cache retained from the previous turn, large amounts of prefill recompute are skipped; otherwise, per-turn recompute causes prefill cost to accumulate linearly.
- **KV-cache eviction**: Sessions are long-lived, and their growing prefix KV is only valuable if it stays on the original replica. But replica KV capacity is limited — a single session's working set can consume more than half of it. The engine evicts older prefix blocks to make room for new requests, the next turn misses, and prefill must be recomputed — a vicious cycle of "fill → evict → miss → recompute."
- **GPU bubbles**: Because the average request context length is large and the distribution is wide, there are typically many GPU bubbles within a batch or across replicas, further lowering MFU and degrading end-to-end performance.

### Design goals

- **Prefix-hit priority**: Subsequent turns of a session return to the original replica and hit the retained KV whenever possible, breaking the evict → miss → recompute cycle.
- **Migration triggered only on genuine overload**: Unbind and migrate only when a replica is truly overloaded — the prefill-recompute cost paid for shedding load must be justified by an explicit criterion.
- **Reduce GPU bubbles**: Through scheduling and multi-level KV-cache management, effectively alleviate the GPU-bubble problem.

## Architecture

`agent_aware_router` is a router with pluggable strategies. Core components:

```
agent_aware_router/
   ├── balancer.py                   ← pure shell: lifecycle + routing delegation
   ├── strategies/                   ← scoring algorithms (core)
   │    ├── kvc_aware.py             ← KVCacheAwareStrategy (main strategy)
   │    ├── routing.py               ← weighted ranking over strategy scores
   │    ├── base.py                  ← ReplicaInfo
   │    └── registry.py              ← config-type → runtime-strategy-class registry
   ├── store/                        ← singleton store
   │    ├── data_store.py            ← metrics + sticky binding + active_sessions
   │    ├── kv_cache_store.py        ← prefix-hash chain + three-layer hit rate + in-flight pin book
   │    ├── per_replica_store.py / per_request_store.py
   ├── collectors/                   ← collect vLLM signals
   │    ├── collector.py/provider.py ← collector lifecycle
   │    ├── parse/vllm/metrics.py  ← /metrics polling → dict
   │    ├── parse/vllm/kv.py, kv_event.py  ← kv-events zmq → incremental block state
   │    └── transport/{http,zmq,callback}.py
   ├── config/                       ← KVCAwareConfig + StrategyConfig
   ├── types/                        ← SlowCut / OverloadMode / Layer / MetricKey enums
   └── insight/emitter.py            ← push to rl-insight → Prometheus → Grafana
```

## Routing decision flow

On each `acquire_server(request_id, prompt_ids)`, the balancer scores the candidate replicas and dispatches the request to the top-ranked one. The default path goes straight to capacity-token-aware scoring — the sticky short-circuit is disabled by default (`do_shortcut=False`):

```
1. Capacity-token-aware scoring → capacity gate + prefill increment
   ├── eligible replicas exist → soft-pick max remaining among them
   └── no replica passes (all under water) → soft-pick min need (cheapest assignment)
```

### Capacity-token-aware scoring

The scoring "picks the replica with the most capacity left after this request's prefill, among those that still have spare capacity." The computation is:

```
cap[i]       = replica i's allocatable shared-pool blocks
held[i]      = distinct group-qualified prefix references + private group budgets
avail[i]     = cap[i] - held[i]
saved[i]     = common reusable prefix tokens - required replay tokens
need[i]      = sum((prompt_tokens - saved[i]) / group_block_size for shared groups)
               + request-private group budgets
remaining[i] = avail[i] - need[i]
eligible[i]  = avail[i] >= cap[i] × (1 - load_threshold)
pick         = soft-pick(max remaining among eligible)
             (no eligible replica → soft-pick min need)
```

- The pin book counts physical pool blocks independently of the local hash unit. Shared prefixes are deduplicated within each cache group, while request-private state, including Mamba `align` state, is reserved per request. Bounded attention groups conservatively retain complete prompt-prefix reservations until release. This is a dispatch estimate rather than sampled engine occupancy.
- Disabling prefix caching keeps capacity routing available for known layouts with usable budgets. Cache savings are zero, and prompt blocks are reserved separately for each request, including requests with identical prompts.
- Capacity is per replica. The router prefers the native reservation profile's allocatable block count and otherwise uses that replica's normalized `cache_config_info` metric. Replicas with unknown capacity or private budgets are excluded while fully known candidates exist. If no candidate has usable capacity and budgets, the router falls back to least-inflight selection. Another replica's capacity is never borrowed.
- Cache estimates intersect all required groups using their own Full, Sliding, Chunked or Mamba matching rules. The final prompt token remains available for recomputation. Bounded replay reduces the saved prefill work. Dispatch predictions can contribute provisional cache benefit before native events confirm the corresponding endpoints.
- Prefill hit rate participates via the `need` term: a replica with a higher hit rate needs fewer recomputed tokens, so `remaining` is larger and it wins among eligible replicas.
- Load balancing first filters via the pure-capacity hard gate `eligible` — replicas that are full are filtered out and no longer assigned. If no replica passes (all under water), the pick switches to `min need`: deficit magnitudes would dominate a `remaining` ranking (`|avail| ≫ plen`) and drown the cache signal, so the saturated pick is the cheapest assignment instead — the replica where this request recomputes the fewest tokens.
- Ranking participates via the `avail` term: among eligible replicas, one with higher `avail` has lower load and thus larger `remaining`.
- The final pick goes through `soft-pick`: among candidates within a small tie tolerance of the best `remaining`, one is drawn at random, spreading new sessions over near-ties instead of stampeding a single replica.


### Native hybrid metadata and recovery

The inference entry point registers the metadata-enabled vLLM replica before creating the server manager. Custom verl entry points using this router should call `register_vllm_metadata_replica()` from `uni_agent.agent_aware_router.server` before `LLMServerManager` creation. Install the current uni-agent package in every engine environment so the `uni_agent_native_hash` general plugin is discoverable; an explicit `VLLM_PLUGINS` allowlist must include that name. The engine's reported `native_hash_config.unit` is the authoritative hash granularity, including the effective `prefix_match_unit`. Group metadata describes matching rules, alignment and allocation budgets; missing metadata is never replaced by an inferred Single-Full layout.

The metadata collector waits up to five seconds during startup, then continues recovery in the background without blocking routing. Engine metadata RPCs share a four-second budget. Failed sources retry after 1, 2, 4, 8, 16 and 30 seconds, then every 30 seconds until recovery or removal. There is at most one outstanding metadata call per replica. Successful identical snapshots preserve cache state; a changed source profile rebuilds only the affected replica. Requests dispatched before allocation metadata is available retain an unknown reservation until release, so recovery cannot mistake them for zero occupancy.

Native hash reconstruction is a separate capability from reading the engine's unit. Unsupported reconstruction returns its reason while preserving the unit and usable capacity information. An event with an invalid native digest is rejected and pauses reconstruction only for its source replica; verified shared identities and other replicas remain usable. Background metadata verification can restore that replica without replaying an unchanged profile. `get_status()` exposes cache capabilities and metadata recovery diagnostics; `cache_diagnostics` counters and `cache_capability_ready` gauges distinguish degradation from an actual cache miss. Warnings are emitted on degradation and periodically during persistent failures, with an information log on recovery.

Transport reconnects discard the affected replica's resident index. Estimates and predictions are gated while replay rebuilds it. When the publisher is unavailable, fresh dispatch predictions may resume from an empty view; a new replay gates them again. Native Clear events reset residency while preserving active request reservations and readiness. Explicit full replica cleanup also removes request reservations. Neither operation changes another replica's cache view.

The collector source set is fixed when the balancer is constructed. Initial endpoints may be discovered after startup if a query times out, but an installed address is not replaced. Changes to the routing pool do not change collector subscriptions: a newly added replica has no cache-event source, and replacing an initial replica's server handle requires a new balancer. Same-source reconnect and replay do not require a new balancer.

### Optional sticky short-circuit (off by default)

The sticky binding maps `request_id → replica_id` and is still recorded on every dispatch — the router notes where each request actually landed, so with the short-circuit re-enabled, a fallback re-pick also rebinds the session to the new winner. By default, though, the binding is not consulted at routing time: every turn, returning sessions included, is scored by capacity-token-aware, and prefix locality is preserved through the `gpu_hit` term instead of by binding — a returning session has a real hit rate on its previous replica, so `need` shrinks there and the score keeps it there.

When re-enabled (the `do_shortcut` strategy knob), a binding hit dispatches the turn straight back to the bound replica. The overload check behind it is a per-replica verdict, not a pool-relative gate:

```
avail[i] = cap[i] - held[i]
gate[i]  = avail[i] >= cap[i] × (1 - load_threshold)        # ≥ 50% headroom at the default 0.5
```

- Bound replica passes the gate → not overloaded → keep sticky.
- Bound replica under water → overloaded → fall through to capacity-token-aware scoring, unconditionally: the verdict is local to the bound replica, and the routing pool is not consulted.
- `cap[i]` unknown (native and sampled capacity are unavailable) → keep sticky — no gate can be evaluated, so missing data never breaks sticky.
