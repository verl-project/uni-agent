# RFC: Context-aware Scheduling and KV Cache Pooling for Agentic RL Rollout

> **Stack**: LLM Router (AiBrix / Dynamo / ...) + vLLM (primary) / SGLang (comparison) + Mooncake KV Store
> **Status**: Draft · 2026-05-14

## 1. Background: The Prefix Growth Curve in Agentic RL Rollout

Agentic RL rollout differs from classical single-turn inference in one fundamental way: the prompt is not given once. It is **assembled turn by turn, monotonically growing** as the agent loops. A typical SWE-Bench trajectory looks like this:

```
turn 1:  send = system_prompt + tools_schema + user_task
         recv = assistant_msg_1 (with tool_call_1)
         exec = tool_runtime → observation_1

turn 2:  send = turn-1 send + assistant_msg_1 + observation_1
         recv = assistant_msg_2
         ...

turn N:  send = turn-(N-1) send + assistant_msg_(N-1) + observation_(N-1)
```

Every prompt sent to the inference engine from turn 2 onward **contains every token sent in the previous turn, plus the previous response and the new observation**. From turn 2 onward there is always a shared prefix between successive requests — a prefix that grows turn by turn, and one the inference engine has just finished computing in the previous turn.

This observation hands us an obvious optimization target. If the engine can reuse the KV from the previous turn, turn-N's prefill only has to compute the small increment of "previous response + new observation". If it cannot, the engine re-prefills tens or hundreds of thousands of tokens accumulated over N−1 turns. At rollout scale — many workers, long trajectories — the difference is the difference between productive GPU time and burning the budget on redundant computation.

## 2. Problem: Where Prefix Cache Hit Rate Breaks Down

The prefix-growth property from §1 puts **prefix cache hit rate** at the center of rollout performance — hit a long prefix and its prefill is free; miss it and pay the full re-prefill cost. In real agentic RL workflows the hit rate is undermined from three independent directions. verl's existing sticky session scheduler (`GlobalRequestLoadBalancer` + `AsyncLLMServerManager`: record `request_id → server_id` in a global LRU cache; subsequent turns of the same episode reuse the binding) addresses part of one category. The other two have no solution at all.

### 2.1 Same Session, Multi-turn, but the KV Drifts Away

This is the case sticky scheduling is designed for: keep all turns of an episode on the same replica, hit the local GPU KV. But the binding is just an LRU table — it does not observe KV state. Three independent failure paths slip past it:

- **LRU eviction**: a long-lived session gets bumped out of the sticky table by newer requests; its next turn is reassigned as if it were brand new.
- **Local eviction**: the mapping is still there, but vLLM's PagedAttention blocks for this episode have long since been evicted — the longer the prompt grows, the more likely the episode evicts itself.
- **Replica failure / scaling**: a replica OOMs, restarts, is replaced, or is removed by scale-in. Every session bound to it must be reassigned, and a freshly added replica is an isolated KV island for some time, with no old session ever migrating there.

### 2.2 Shared Prefix Across Sessions, but Not Shared

Inside an RL step the same task usually runs as several parallel samples (a GRPO group). The samples share `system_prompt + task_description + tool_schema` — typically thousands of tokens. But each sample has its own `session_id`, so they are scattered across replicas and **launched concurrently**: every replica misses locally and misses in the pool (the prefix has never been computed yet), and N replicas start the same prefill in parallel. A textbook cache stampede.

Sticky routing cannot see this class of failure at all — it handles within-session reuse, not across-session sharing.

### 2.3 After a Weight Update, the Whole Pool Becomes Stale

Sync RL must reset the engine's KV at every step — KV computed under different weights cannot be mixed without breaking the gradient estimate. So the next step's turn 1 prefills from scratch regardless of how requests are routed: `N_session × (system_prompt + task + tools_schema)` tokens, every step. This is the pain unique to RL and sticky session has nothing to offer.

Putting the three together: existing sticky scheduling has partial coverage on §2.1 only; §2.2 and §2.3 are fully uncovered. Those are the gaps we need to fill.

## 3. Design Goals

**Single objective: maximize the prefix cache hit rate during rollout.**

Three complementary mechanisms address the three failure classes from §2:

1. **Affinity scheduling** (§5.1 + §5.3 routing rule fast path): route turns of the same session to the same replica whenever possible, preferring a local GPU KV hit. The steady-state path uses session_id consistent hashing plus a lightweight check (healthy + GPU hit + not overloaded); if it passes, route in O(1). Otherwise run the full two-stage rule across all replicas.
2. **CPU-resident L2 KV pool** (§5.1 connector + §5.2 version key): Mooncake serves as a cross-replica shared L2 KV pool, resident in CPU memory and interconnected over RDMA. When affinity breaks down (sticky-table eviction, local eviction, replica restart, scaling), KV loads from the pool rather than being recomputed. This layer is particularly critical for the agentic-RL pattern of monotonically growing prompts — once the tens-to-hundreds of thousands of tokens accumulated by a long session are evicted locally, only the pool can rescue them.
3. **Prefill pre-warming** (§5.4): after every weight update completes and before rollout opens, each replica independently pre-prefills the shared prefix (`system_prompt + tools_schema + the current step's task descriptions`). The first turn after a weight update hits locally, eliminating §2.3's step-turn-1 full miss and §2.2's GRPO-group cold-start stampede.

A cross-cutting semantic constraint binds all three mechanisms: **every KV reuse must strictly match weight version** (§5.2's version-tagged key), guaranteeing that any reused KV is provably computed under the correct weights and never pollutes the gradient.

Agent-side transparency is a non-functional requirement: keep the OpenAI-compatible interface; the agent only adds two metadata fields per request — `session_id` and `weight_version`.

## 4. Architecture: Three Layers of Context-aware Scheduling

```
                ┌─────────────────────────────────────────┐
                │                Agent Loop                │
                │   per-request metadata:                  │
                │       session_id, weight_version         │
                └────────────────────┬────────────────────┘
                                     │ OpenAI-compatible API
                                     ▼
                ┌─────────────────────────────────────────┐
                │   LLM Router (AiBrix / Dynamo / ...)     │
                │  · session_id consistent hashing         │
                │  · rule-based fallback (§5.3)            │
                │  · replica health & load aware           │
                └──────────┬───────────────────┬──────────┘
                           │                   │
              primary       ▼     fallback     ▼
                ┌──────────────────┐   ┌──────────────────┐
                │  vLLM replica 1   │   │  vLLM replica 2   │
                │  PagedAttention   │   │  PagedAttention   │
                │  + Mooncake conn. │   │  + Mooncake conn. │
                └────────┬──────────┘   └────────┬──────────┘
                         │  load/store KV blocks │
                         ▼                       ▼
                ┌─────────────────────────────────────────┐
                │   Mooncake KV Pool  (CPU + RDMA-shared)  │
                │   key = (hash(token_prefix), weight_ver) │
                │   eviction: LRU                          │
                └─────────────────────────────────────────┘
                                     ▲
                                     │ trainer pushes new weights
                                     │ tagged with a new weight_version
                ┌─────────────────────────────────────────┐
                │              Trainer (verl)              │
                └─────────────────────────────────────────┘
```

Three layers, three responsibilities:

- **Routing Layer (LLM Router)**: the single entry point. Steady-state requests take a fast path — session_id consistent hashing plus a lightweight rule-(1) check, O(1). Requests that fail the check fall through to the two-stage rule of §5.3: prefer a GPU-hit + low-load replica, otherwise pick the least-loaded one. This RFC does not bind to a specific implementation — any open-source LLM router (AiBrix, Dynamo, etc.) can plug in by speaking the reporting protocol described in §5.3.
- **Inference Layer (vLLM)**: PagedAttention untouched; a Mooncake KV connector is added. On local miss, the connector loads from the pool; before a block is evicted, the connector dumps it to the pool.
- **KV Pool (Mooncake)**: a cross-replica L2 KV pool living in CPU memory, moving blocks over RDMA. The cache key combines a prefix hash and the weight version; eviction is LRU.

## 5. Key Mechanisms

### 5.1 Request Path

A typical turn-N request flows as follows:

1. **Agent issues the request**. The agent calls the OpenAI-compatible endpoint, attaching two metadata fields: `session_id` (= episode id) and `weight_version` (the current actor weight version this rollout worker holds).
2. **LLM Router routes**. The router applies the two-stage rule in §5.3 to choose a replica. The steady-state fast path: pick the primary by session_id consistent hashing, run rule (1) on it alone (healthy ∧ GPU hit ∧ not overloaded), and forward in O(1) if it passes. Otherwise scan all replicas with the full rule.
3. **vLLM receives**. The replica looks up its local PagedAttention prefix cache:
   - **Local hit**: prefill only the remaining suffix.
   - **Local miss** (first arrival / evicted locally / routed here by rule (2) fallback): the Mooncake connector queries the pool with `(hash(prefix), weight_version)`:
     - **Pool hit**: asynchronously RDMA-load the missing KV blocks into local GPU memory.
     - **Pool miss**: full prefill, then asynchronously write the newly computed KV blocks back to Mooncake.
4. **Decode + respond**. Decode runs as usual; the OpenAI-compatible response is returned to the agent.
5. **Dump before eviction**. Before PagedAttention evicts a block, a connector hook writes it to Mooncake (if not already present), so a local "eviction" is really a "demotion to L2" and never a loss.

### 5.2 Coordination with Weight Updates

Putting `weight_version` into the Mooncake key turns weight coordination into a natural-expiration problem:

- **Sync RL**: at step boundary the trainer pushes new weights to each inference replica, tagged with a new `weight_version`. Local KV is reset as before; pool entries for the old version are *not* explicitly invalidated. The next step's requests carry the new version, so old keys never match — they age out under LRU on their own. This converts §2.3's "must explicitly flush the entire pool" requirement into "old entries expire naturally", removing the need for a trainer↔router↔pool synchronous broadcast.
- **Fully-Async RL**: training and rollout run concurrently, with multiple weight versions coexisting by design. Each rollout worker carries its current `weight_version`; the corresponding KV occupies its own key space in the pool, never colliding with others.

This does have one direct consequence: pool capacity is shared across weight versions. But the LRU prefers cold old-version entries first, and the truly hot prefixes — `system_prompt + tool_schema`, shared across virtually all sessions — stay resident. The hit rate impact is small.

### 5.3 Routing Rule

To keep the router easy to implement and easy to explain, this RFC uses a naive two-stage rule rather than a complex cost function:

1. **Preferred**: pick a replica whose prefix hits the local GPU (`L_gpu(r) ≥ hit_threshold`) **and** whose load is below threshold (`wait(r) < load_threshold`); among multiple candidates, take the one with the largest `L_gpu(r)`.
2. **Fallback**: if (1) yields no candidate, pick the least-loaded replica (`argmin wait(r)`) and let it either re-prefill locally or load from Mooncake.

The three concerns are addressed separately:

- **GPU hit**: the first condition of rule (1).
- **Load balance**: the load threshold in rule (1) keeps overloaded replicas out of the GPU-hit candidate set; rule (2) then falls through to least-loaded.
- **Recompute vs Mooncake load**: **not decided at the routing layer.** Once the request lands on a replica, §5.1's request path handles it automatically — a pool hit triggers RDMA load, a pool miss triggers prefill. The router does not need to predict which is cheaper.

**Inputs.** `L_gpu(r)` comes from each replica's periodically-reported local prefix index; `wait(r)` comes from each replica's periodically-reported in-flight count. `hit_threshold` and `load_threshold` are profiling-derived and configurable.

**Fast path.** In the steady state of multi-turn agentic rollout, the primary picked by `session_id` consistent hashing is almost always the rule-(1) winner — its local KV is intact, the binding is continuous, and the load is stable. The router can pick the primary first, run the rule-(1) check on it alone, and forward in O(1) on success — skipping the full scan over all replicas.

**Degenerate behavior:**

- Primary failure: treated as `wait(primary) := +∞` (or rejected by the health check), naturally dropping out of the candidate set.
- All replicas miss locally: rule (1) yields no candidate; rule (2) picks the least-loaded one.
- Pool also misses: the chosen replica re-prefills, exactly tying the no-cache baseline.

### 5.4 Prefix Pre-warming

§2.2 and §2.3 — the GRPO-group cold-start stampede and the Sync-RL turn-1 full miss — are at heart the same situation: right after a weight update, no replica holds any KV matching the new `weight_version`, and the moment rollout opens, N concurrent requests pile in. Rather than letting the stampede happen on the request path, pre-warming pushes the unavoidable prefill forward in time — to *before* rollout opens.

**Scope.** The default warm set is `system_prompt + tools_schema + all task descriptions of the current step`. The first two are virtually constant across training and only need to be pre-warmed once per weight version. The task descriptions are exactly what a GRPO group shares; once warmed, every sample in the group hits locally on turn 1.

**Execution.** Each replica **independently and in parallel** prefills the entire warm set. We deliberately avoid a "leader prefills, others load from Mooncake" pattern — the warm set is only a few thousand tokens, so local re-prefill beats Mooncake RDMA round-trips in both latency and simplicity. Wall-clock is one prefill, not R prefills. The produced KV enters PagedAttention's normal prefix cache; because every subsequent request touches it, the LRU keeps it at the hot end naturally — no explicit pinning is needed.

**Timing.** Blocking. Rollout opens only after every replica reports warm-done. Sync RL's weight update is already on the critical path, so attaching pre-warming behind it merely moves the "first-step first-turn prefill" earlier without lengthening the wall-clock; the payoff is that every subsequent turn-1 request hits locally, eliminating the stampede outright.

This section is fully decoupled from §5.1–§5.3: routing, the Mooncake pool, and the `weight_version` protocol are untouched. The pre-warmed KV is consumed on the local-hit path, never going through Mooncake at all.

## 6. Comparison with Existing Approaches

| Capability | vLLM native prefix cache | verl `AsyncLLMServerManager` (sticky) | SGLang RadixAttention | Context-aware Scheduling (this RFC) |
|---|---|---|---|---|
| Same-episode cross-turn reuse | within a replica | ✓ (sticky binding) | within a replica | ✓ (session_id routing) |
| Cross-replica KV sharing | ✗ | ✗ | ✗ | ✓ (Mooncake pool) |
| Recovery after replica failure | ✗ | ✗ (binding lost) | ✗ | ✓ (load from pool) |
| Scale-out utilization for new replicas | ✗ | ✗ (only new sessions land there) | ✗ | ✓ (rule picks best replica) |
| Semantic safety across weight updates | manual flush | manual flush | manual flush | ✓ (version-tagged keys) |
| Long session, locally evicted | full re-prefill | full re-prefill | full re-prefill | hit if still in pool |
| Cold-start (step turn-1 / GRPO group) | ✗ | ✗ | ✗ | ✓ (pre-warming) |
| Steady-state routing cost | N/A | O(1) (LRU lookup) | N/A | O(1) (consistent hash fast path) |
| Non-fast-path routing cost | N/A | none | N/A | O(R) rule scan (R = #replicas) |

In one sentence: vLLM and SGLang solve in-process prefix reuse; verl sticky solves same-episode routing stability; Context-aware Scheduling adds the two missing pieces on top — a cross-process KV-sharing layer, and a routing strategy that is aware of KV state.

## 7. SGLang Variant

Swapping vLLM for SGLang keeps the three-layer architecture intact. The differences concentrate in the inference layer:

- **Local prefix index comes for free.** SGLang's RadixAttention already maintains a prefix index as a radix tree internally — reporting it to the router is a tree traversal, no extra data structure required.
- **Different connector hook points.** vLLM's connector hooks live in the PagedAttention block manager; SGLang's hooks live on radix-tree node allocation and eviction callbacks. The contract is identical: turn "block eviction" into "write to Mooncake".
- **Different weight-update API.** SGLang exposes `/update_weights`; carrying `weight_version` as a parameter is a one-line change. Router and pool protocols are untouched.

The routing layer (LLM Router) and KV pool (Mooncake) are engine-agnostic. One router can manage vLLM and SGLang replicas at once; mixed deployments are unusual but legal.

## 8. Open Questions / Future Work

- **Cross-weight-version KV reuse.** This RFC requires a strict `weight_version` match. In principle KV from adjacent weight versions is approximately reusable — combined with an importance-sampling correction, this could lift hit rate further. The cost is mathematical complexity and additional empirical validation. Out of scope here.
- **Prefill/Decode Disaggregation.** Mooncake natively supports PD-disaggregated deployment, which would isolate prefill load into a dedicated pool and reduce its interference with decode latency on long prompts. Left for a follow-up phase.
- **Multimodal KV prefix hashing.** Image patch tokens require semantic hashing different from text token-id hashing (the same image may produce different tokens through different preprocessing paths). A reproducible hashing protocol is needed before multimodal KV can enter the pool.

---

*This RFC is the public-facing technical-blog version; internal implementation details, dependency versions, and milestones live in the corresponding engineering documents.*
