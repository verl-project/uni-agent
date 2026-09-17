# Agent Aware Router

The `kvcaware` router is a pluggable load balancer registered in `LoadBalancerRegistry` under the name `"kvcaware"` and wired in on the `LLMServerManager` side via `get_router_handle`. To run it, see [Run the Agent Aware Router](../quickstart/agent-aware-router.md).

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
   ├── balancer.py                   ← pure shell: lifecycle + route() delegation
   ├── strategies/                   ← scoring algorithms (core)
   │    ├── kvc_aware.py             ← KVCacheAwareStrategy (main strategy)
   │    ├── routing.py               ← route() weighted ranking
   │    ├── base.py                  ← ReplicaInfo
   │    └── registry.py              ← config-type → runtime-strategy-class registry
   ├── store/                        ← singleton store
   │    ├── data_store.py            ← metrics + sticky binding + active_sessions
   │    ├── kv_cache_store.py        ← prefix-hash chain + three-layer hit rate
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

On each `acquire_server(request_id, prompt_ids)`, the `Balancer` calls `route()`, which invokes each strategy's `score()` for a weighted ranking and takes `ranking[0]`. The decision tree of `KVCacheAwareStrategy.score()`:

```
1. STICKY short-circuit (do_shortcut=True)
   ├── request_id present and a sticky-bound replica exists
   │    └── not overloaded? → HIT, return [STICKY_TOP_SCORE, 0, ...] (preserve prefix locality)
   │        overloaded? → fallback
   └── else → enter slow-cut
2. SLOW-CUT (fallback scoring after the short-circuit misses)
   └── CAPACITY_TOKEN_AWARE  → capacity gate + token increment, pick max remaining
```

### CAPACITY_TOKEN_AWARE scoring

CAPACITY_TOKEN_AWARE "picks the replica with the highest prefill hit rate among those that still have spare capacity." The computation is:

```
avail[i]     = cap × (1 - kv_cache_usage_perc[i])                   # pure available tokens
need[i]      = len(prompt_ids) × (1 - gpu_hit[i])                   # tokens to recompute at replica i
remaining[i] = avail[i] - need[i]                                   # remaining after assignment
eligible[i]  = avail[i] >= cap × (1 - capacity_reserve_threshold)   # pure capacity hard gate
pick         = argmax(eligible, remaining)                          # among eligible, pick max remaining
```

- Prefill hit rate participates via the `need` term: a replica with a higher hit rate needs fewer recomputed tokens, so `remaining` is larger and it wins among eligible replicas.
- Load balancing first filters via the pure-capacity hard gate `eligible` — replicas that are full are filtered out and no longer assigned. Then ranking participates via the `avail` term: a replica with higher `avail` has lower load and thus larger `remaining`.
