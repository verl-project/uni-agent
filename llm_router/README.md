# llm_router

Drop-in replacement for verl `AgentLoopManager` with pluggable routing policies.

## Usage

Set in your verl trainer config:

```yaml
actor_rollout_ref:
  llm_router:
    policy: legacy_sticky   # or: rule_based
    routing_cache_size: 10000
  rollout:
    agent:
      agent_loop_manager_class: "llm_router.LLMRouter"
```

## Policies

- `legacy_sticky` — exact behavior of verl's `GlobalRequestLoadBalancer`. Use this to validate the drop-in is byte-identical to the original.
- `rule_based` — RFC §5.3 two-stage rule. Fast path: session consistent hashing + health/GPU-hit/load feasibility check. Slow path: rule 1 picks the replica with the largest GPU prefix hit under the load threshold; rule 2 falls back to least-loaded.

### Enabling rule_based

The two-stage rule requires `prefix_signatures` to flow in on every request,
plus `report_prefixes` calls after each generation. **verl's
`AsyncLLMServerManager` already does both** (see
`verl/verl/experimental/agent_loop/agent_loop.py:287, 360`) — you just need
to turn the context-aware flag on in your trainer config:

```yaml
actor_rollout_ref:
  llm_router:
    policy: rule_based
    gpu_hit_threshold: 256            # min HBM prefix tokens to count as a GPU hit
    cpu_hit_threshold: 1024           # min Mooncake CPU prefix tokens to count as an L2 hit
    load_threshold: 1024              # in-flight queue cap before rule 1 rejects
    max_prefix_entries_per_server: 8192
    routing_cache_size: 10000
  rollout:
    agent:
      agent_loop_manager_class: "llm_router.LLMRouter"
    context_aware_scheduling:
      enable: true                      # verl flag — drives worker-side prefix_signatures + report
      prefix_probe_stride: 256
```

With `enable: true`, verl's `AsyncLLMServerManager` computes
`prefix_signatures` from each request's `prompt_ids + weight_version` and
threads them through `acquire_server`. After generation, it reports the
cached-prompt signatures back via `report_prefixes`. `LLMRouter`'s
`LoadBalancer` consumes both and feeds `RuleBasedPolicy`.

`RuleBasedPolicy` keeps GPU/HBM and Mooncake/CPU prefix locations in separate
indexes. Routing order is: GPU hit above `gpu_hit_threshold`, then Mooncake
CPU placement hit above `cpu_hit_threshold`, then least-loaded fallback.

### Enabling prewarm

After every weight update, RFC §5.4 calls for each replica to pre-prefill
the shared prefix (`system_prompt + tools_schema + current step's task
descriptions`) so that the first request of the next step never sees a
cold GPU cache. verl already does this end-to-end:

  - `verl/.../agent_loop.py:805` — `_maybe_prewarm_prefixes` runs at the
    start of each `generate_sequences` step.
  - `verl/.../agent_loop.py:372` — `AsyncLLMServerManager.prewarm_prefixes`
    dedups, calls each replica's prewarm RPC (real prefill on GPU), then
    reports the prewarmed signatures to the LoadBalancer for routing.
  - `verl/.../agent_loop.py:405` — the report path feeds into
    `RuleBasedPolicy._prefix_locations`, so the next acquire_server can
    fast-path or rule-1 hit immediately.

Turn it on by adding to your trainer config:

```yaml
actor_rollout_ref:
  rollout:
    context_aware_scheduling:
      enable: true                  # Plan C — drives routing
      prewarm_enable: true          # Plan D — drives prefill warm-up
      prewarm_max_prefixes: 0       # 0 = unlimited (dedup by version+hash)
      prewarm_decode_tokens: 1      # tokens to decode beyond prefill (≥1)
      prefix_probe_stride: 256      # stride for prefix_signatures sampling
```

`prewarm_decode_tokens=1` is sufficient because the goal is to land KV
in PagedAttention, not to generate useful output. Setting it higher
warms a longer suffix at extra GPU cost.

With `prewarm_enable: true`, every weight update + rollout dispatch
warms every replica with the step's shared prefixes. `RuleBasedPolicy`
sees the reports via `lb.report_prefixes.remote(...)` and the next
turn-1 of every session lands on a replica with a GPU hit. The
GRPO-group cold-start stampede described in RFC §2.2 is eliminated;
the Sync-RL step-1 turn-1 full miss in RFC §2.3 is also eliminated.

The end-to-end behavior is verified by
`llm_router/tests/test_prewarm_routing_e2e.py`.

### Remaining Plan E follow-ups

- **RDMA async reads**: the connector uses the async `KVStore` contract. The
  production backend now talks to Mooncake's pooled `MooncakeDistributedStore`,
  but it still exposes synchronous get/upsert calls behind `begin_get` /
  `poll_get`. A lower-latency follow-up should drive true nonblocking Mooncake
  transfers.
- **Negative reporting/invalidation**: connector hooks now send positive
  availability hints to the LB. Evictions should also retract stale prefix
  locations before the router prefers a replica for KV it no longer has.

## Manual parity check

On a GPU host:

    CUDA_VISIBLE_DEVICES=2 LLM_ROUTER_PARITY_MODEL=/path/to/small/model \
      pytest llm_router/tests/test_manager_parity.py -v -s --tb=short

This runs a 2-prompt rollout twice (once via stock AgentLoopManager, once via
LLMRouter) in isolated subprocess/Ray lifecycles and asserts the output shapes
and first greedy response tokens match exactly.
