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

On each `acquire_server(request_id, prompt_ids)`, the `Balancer` calls `route()`, which invokes each strategy's `score()` for a weighted ranking and takes `ranking[0]`. The decision tree of `KVCacheAwareStrategy.score()`:

```
1. STICKY short-circuit (do_shortcut=True)
   ├── request_id present and a sticky-bound replica exists
   │    └── not overloaded? → HIT, return [STICKY_TOP_SCORE, 0, ...] (preserve prefix locality)
   │        overloaded? → fallback (REMAINING_CAP: rebind only on positive evidence — see below)
   └── else → enter slow-cut
2. SLOW-CUT (fallback scoring after the short-circuit misses)
   └── CAPACITY_TOKEN_AWARE  → capacity gate + token increment, soft-pick max remaining
```

### Sticky short-circuit and rebind gating

The sticky binding maps `request_id → replica_id` and is rewritten on every dispatch — the router records where each request actually landed, so a fallback re-pick also rebinds the session to the new winner. On a hit — a binding exists, the bound replica is in the pool, and it is not overloaded — `score()` short-circuits to `[STICKY_TOP_SCORE, 0, ...]`: the turn prefills against the retained prefix (`need ≈ plen × (1 - gpu_hit)`) instead of recomputing it on another replica.

"Overloaded" is configured by `overload_mode` (`none` / `kv_cache_usage_perc` / `kv_load` / `remaining_cap`; `remaining_cap` is the deployed mode). In `remaining_cap` the check is a relative gate against the whole pool, not an absolute load flag:

```
avail[i] = cap - inflight_blocks[i] × block_size
gate[i]  = avail[i] >= cap × (1 - capacity_threshold)   # ≥ 10% headroom at the default 0.9
```

- Bound replica passes the gate → not overloaded → keep sticky.
- Bound replica under water but at least one sibling passes → overloaded → fall through to CAPACITY_TOKEN_AWARE. This is the only case that breaks locality, and it requires positive evidence that a sibling has real headroom.
- No replica passes (all under water) → keep sticky.
- `cap` unknown (first `num_gpu_blocks` poll not in yet) → keep sticky — no gate can be evaluated, so missing data never breaks sticky.

The all-under-water case keeps sticky on purpose: every candidate is below the gate, and the `remaining` ranking is then below the noise floor of the in-flight book — releases and re-acquires swing each replica's book by far more than the top-1/top-2 gap over a request's queue-wait horizon. Rebinding would pay a full re-prefill to land on a replica that is equally under water. Instead, existing sessions hold position while new sessions still spread via CAPACITY_TOKEN_AWARE, and the books drain as turns release, re-opening the gate.

### CAPACITY_TOKEN_AWARE scoring

CAPACITY_TOKEN_AWARE "picks the replica with the most capacity left after this request's prefill, among those that still have spare capacity." The computation is:

```
cap          = num_gpu_blocks × block_size                   # first positive poll wins; block_size learned from the first KV event (default 16)
avail[i]     = cap - inflight_blocks[i] × block_size         # free capacity, from the router's own pin book
need[i]      = len(prompt_ids) × (1 - gpu_hit[i])            # tokens to recompute at replica i
remaining[i] = avail[i] - need[i]                            # remaining after assignment
eligible[i]  = avail[i] >= cap × (1 - capacity_threshold)    # pure capacity hard gate
pick         = soft-pick(max remaining among eligible)       # random within tie_tolerance
```

- `inflight_blocks[i]` is the router's own load account: a per-replica pin book of the KV blocks held by in-flight requests, ref-counted at acquire (pin the request's full-block prefix hashes) and at release (unpin the snapshotted set; a shared prefix block leaves the book when its last holder releases). kv-events carry no request identity, so engine-side residency alone cannot separate "cached and idle" from "held and about to grow" — the pin book is the occupancy signal the router owns synchronously on the acquire/release path. `avail` is deliberately not derived from the polled `kv_cache_usage_perc`: that is an async snapshot with cold-start blind spots, while the pin book is up to date at decision time.
- `gpu_hit[i]` comes from the per-replica remote→local block-hash index built from kv-events (extended at dispatch time with the request's own prefix), so a returning session is ranked with its real hit rate.
- Prefill hit rate participates via the `need` term: a replica with a higher hit rate needs fewer recomputed tokens, so `remaining` is larger and it wins among eligible replicas.
- Load balancing first filters via the pure-capacity hard gate `eligible` — replicas that are full are filtered out and no longer assigned. If no replica is eligible, the gate is dropped (pool = all replicas) so the ranking still returns a winner. Then ranking participates via the `avail` term: a replica with higher `avail` has lower load and thus larger `remaining`.
- Cold start: before the first `num_gpu_blocks` poll lands, `cap` is 0 and no gate can be evaluated; scoring falls back to ranking by minimum in-flight request count — the only load term the router owns synchronously at startup.
- The final pick goes through `soft-pick`: among candidates within `tie_tolerance` (default 0.05) of the best `remaining`, one is drawn at random, spreading new sessions over near-ties instead of stampeding a single replica.
