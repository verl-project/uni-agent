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
   │    ├── data_store.py            ← capability composition + compatibility facade
   │    ├── inflight.py              ← dispatch accounting and pin lifecycle
   │    ├── capabilities.py          ← metrics / KV / named request-state operations
   │    ├── contracts.py             ← read interfaces for scoring and observation
   │    ├── kv_cache_store.py        ← prefix-hash chain + three-layer hit rate + in-flight pin book
   │    ├── per_replica_store.py / per_request_store.py
   ├── collectors/                   ← collect vLLM signals
   │    ├── collector.py/provider.py ← collector lifecycle
   │    ├── sink.py/updater.py       ← typed application boundary + observation dispatch
   │    ├── parse/vllm/metrics.py  ← /metrics polling → dict
   │    ├── parse/vllm/kv.py, kv_event.py  ← kv-events zmq → incremental block state
   │    └── transport/{http,zmq,callback}.py
   ├── config/                       ← KVCAwareConfig + StrategyConfig
   ├── types/                        ← SlowCut / OverloadMode / Layer / MetricKey enums
   └── insight/collector.py / emitter.py ← observe applied updates → rl-insight → Prometheus → Grafana
```

### Collector application boundary

`Collector` owns transport lifecycle and ordered parser dispatch. Its optional
`sink` argument implements `UpdateSink`: one operation per typed update kind.
Tests can inject a sink without constructing `DataStore`. The default
`StoreUpdater` applies KV, metrics and sticky updates and forwards inflight
facts to `DataStore.inflight`. DataStore composes four capabilities:

- `metrics`: merged polling snapshots and signed metric increments.
- `kv`: cache residency, block references, and acquire-time pin markers.
- `requests`: sticky bindings, historical turn/prompt memos and the prefix-hash memo.
- `inflight`: acquire/release transitions, active bookings and their accounting.

`InflightParser` only decodes the callback into an `InflightUpdate` containing
its action, replica, request ID and acquire prompt tokens. It receives no store
context and emits no derived metric deltas or pin/booking commands. There is no
contextual parser wrapper. DataStore resolves prefixes using its learned block
size, allocates the next turn, reverses stale bookings and debits the acquire-time
turn/prompt length on release. Direct DataStore callers use the same logic.
`StoreUpdater` translates the returned applied effects into observer calls; it
does not decide accounting amounts.

DataStore resolves the new prefix before making any writes, so invalid token
hashing leaves an old dispatch intact. It commits booking state after block and
metric writes succeed, then returns the applied metric deltas and their post-write
values for observation. A handler lock serializes parsing through application,
including callbacks arriving from different threads. Each inflight collector owns
one DataStore inflight capability through its updater; this lock does not
coordinate separate collectors or processes sharing the underlying singletons.

The inflight capability keeps active dispatch bookings (replica, turn and prompt
length) until release or re-acquire, independently of request memo LRU eviction
and block markers. Duplicate/unknown releases are ignored, including releases
from a stale replica after migration. Re-acquire reverses the old replica's count,
tokens and turn contribution before charging the new dispatch, even without pinned
blocks. Unpinning uses the acquire-time marker. A missing release retains the
active booking until re-acquire or capability teardown. Historical turn counts
and other optional request memos remain LRU-bounded.

Callbacks carry no dispatch generation: a delayed release from an earlier turn
on the *same* replica cannot be distinguished from the current turn's release.
Callers must maintain acquire/release ordering for each request. Unattributed
callbacks (`request_id=None`) retain the legacy count-only release behavior.
The existing store locks remain in place; a sequence of commands is not an atomic
transaction across stores and arbitrary write/observer failures are not rolled
back.

`CollectorObserver` receives successful applications and post-write metric
values. It owns `WriteEvent` construction, metric emission and evidence-log
windows; it only receives read interfaces. Router-recorded block totals are logged
by the inflight observer, separately from the KV-event observer's tally. Metric
names, labels and routing configuration are unchanged. Scoring likewise reads metrics/KV state through
`RoutingState`; its only request write is the existing prefix-hash memo.

Flat `DataStore` methods remain compatibility delegates. New production callers
use the grouped capabilities and named request operations, rather than generic
scratch keys. The underlying stores, shared singleton identity, request LRU
lifetime for optional memos and cache algorithms are unchanged.

The balancer's drain counter follows the same one-active-dispatch rule for named
requests: a replacement acquire moves the count from the previous replica to the
selected replica. Duplicate releases and releases naming a different replica do
not decrement it. Removing a replica drops its drain counts and named assignments;
later releases still reach collectors to clean up their outstanding bookings, but
cannot debit unrelated requests if that replica is added again. Named callers
must supply `request_id` on release. The legacy release without an ID remains
count-only; it cannot identify or retire a named booking. Without a dispatch
generation, a delayed release from an earlier turn on the same replica is still
indistinguishable from the current release, so per-request ordering is required.

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
cap          = num_gpu_blocks × block_size                   # first positive poll wins; block_size learned from the first KV event (default 16)
avail[i]     = cap - inflight_blocks[i] × block_size         # free capacity, from the router's own pin book
need[i]      = len(prompt_ids) × (1 - gpu_hit[i])            # tokens to recompute at replica i
remaining[i] = avail[i] - need[i]                            # remaining after assignment
eligible[i]  = avail[i] >= cap × (1 - load_threshold)        # pure capacity hard gate
pick         = soft-pick(max remaining among eligible)       # random within a small tie tolerance
             (no eligible replica → soft-pick min need)
```

- `inflight_blocks[i]` is the router's own load account: a per-replica pin book of the KV blocks held by in-flight requests, ref-counted at acquire (pin the request's full-block prefix hashes) and at release (unpin the snapshotted set; a shared prefix block leaves the book when its last holder releases). kv-events carry no request identity, so engine-side residency alone cannot separate "cached and idle" from "held and about to grow" — the pin book is the occupancy signal the router owns synchronously on the acquire/release path. `avail` is deliberately not derived from the polled `kv_cache_usage_perc`: that is an async snapshot with cold-start blind spots, while the pin book is up to date at decision time.
- `gpu_hit[i]` comes from the per-replica remote→local block-hash index built from kv-events (extended at dispatch time with the request's own prefix), so a returning session is ranked with its real hit rate.
- Prefill hit rate participates via the `need` term: a replica with a higher hit rate needs fewer recomputed tokens, so `remaining` is larger and it wins among eligible replicas.
- Load balancing first filters via the pure-capacity hard gate `eligible` — replicas that are full are filtered out and no longer assigned. If no replica passes (all under water), the pick switches to `min need`: deficit magnitudes would dominate a `remaining` ranking (`|avail| ≫ plen`) and drown the cache signal, so the saturated pick is the cheapest assignment instead — the replica where this request recomputes the fewest tokens.
- Ranking participates via the `avail` term: among eligible replicas, one with higher `avail` has lower load and thus larger `remaining`.
- The final pick goes through `soft-pick`: among candidates within a small tie tolerance of the best `remaining`, one is drawn at random, spreading new sessions over near-ties instead of stampeding a single replica.


### Optional sticky short-circuit (off by default)

The sticky binding maps `request_id → replica_id` and is still recorded on every dispatch — the router notes where each request actually landed, so with the short-circuit re-enabled, a fallback re-pick also rebinds the session to the new winner. By default, though, the binding is not consulted at routing time: every turn, returning sessions included, is scored by capacity-token-aware, and prefix locality is preserved through the `gpu_hit` term instead of by binding — a returning session has a real hit rate on its previous replica, so `need` shrinks there and the score keeps it there.

When re-enabled (the `do_shortcut` strategy knob), a binding hit dispatches the turn straight back to the bound replica. The overload check behind it is a per-replica verdict, not a pool-relative gate:

```
avail[i] = cap - inflight_blocks[i] × block_size
gate[i]  = avail[i] >= cap × (1 - load_threshold)        # ≥ 50% headroom at the default 0.5
```

- Bound replica passes the gate → not overloaded → keep sticky.
- Bound replica under water → overloaded → fall through to capacity-token-aware scoring, unconditionally: the verdict is local to the bound replica, and the routing pool is not consulted.
- `cap` unknown (first `num_gpu_blocks` poll not in yet) → keep sticky — no gate can be evaluated, so missing data never breaks sticky.
