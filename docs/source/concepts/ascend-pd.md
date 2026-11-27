# Ascend PD decode routing

Updated: 2026-11-17.

This module provides optional decode-selection policies for the prefill/decode
(PD) path in [verl PR #8686](https://github.com/verl-project/verl/pull/8686).
It does not replace Uni-Agent's existing `agent_aware_router` stack or implement
KV transfer. Each prefill engine owns its own selector instance; there is no
shared cross-P decode-load ledger.

## Configuration and ownership

Uni-Agent owns the policy classes, `DecodeCandidate`, and
`uni_agent.rollout.router_balancer.SelectorRouter`. The adapter implements
verl's router protocol, while the companion verl integration owns request
dispatch, engine startup and KV-transfer configuration.

```yaml
actor_rollout_ref:
  rollout:
    disaggregation:
      enabled: true
      transfer_backend: mooncake
    custom:
      pd_router_config_path: /absolute/path/to/session_sticky.yaml
```

This is a fragment, not a complete training recipe. The companion integration
currently enables the optional inner D router on NPU. An unset path keeps
round-robin; GPU PD retains the companion's native selection behavior.

Policy YAMLs are packaged in `uni_agent.rollout.configs` and can be located with
`importlib.resources.files`. The selected path must be readable by the Ray
actor. The loader expects a filesystem path, not a `pkg://` resource URL.

```yaml
router_class: uni_agent.rollout.router_balancer.SelectorRouter
selector_class: uni_agent.rollout.decode_selectors.SessionStickyDecodeSelector
inflight_token_budget: 2000000
```

## Policies

| YAML | Decision |
| --- | --- |
| random | Random candidate |
| least_load / power_of_two | Sample candidates and minimize local in-flight request count |
| consistent_hash | Virtual-node hash ring keyed by session |
| rendezvous_hash | Highest session/server hash score |
| cache_aware | Estimated token-prefix affinity with load-imbalance fallback |
| session_sticky | Session hash affinity with KV/queue/token-pressure fallback |

Prefix affinity estimates do not establish actual engine KV residency. Metrics
may be stale or unavailable, and the all-hot fallback still dispatches a
request; these policies do not implement strict admission control. No throughput
or cache-hit improvement is claimed without target-workload measurements.

## Request identity and accounting

Upgrade the adapter and companion verl integration together. `request_id`
remains the session-affinity key. Each active PD call supplies a fresh
`reservation_id` through the declared acquire/release fields, so concurrent
turns in one session can finish in any order without releasing each other's
load. Release must use the same reservation identity as acquisition.

The adapter records reservation-to-server ownership. Duplicate or unknown
releases are no-ops, wrong-server releases retain ownership and raise, and
duplicate active acquisition is rejected before changing selector load.
Session-sticky releases the recorded prompt weight for the exact reservation.
Resetting cache affinity preserves active reservations and their load.

`get_total_inflight()` and `get_status()["total_inflight"]` always return the
number of active adapter reservations, for every policy. Zero-token requests
also count. `get_status()["pending_prompt_tokens"]` separately exposes the
selector's prompt-token weight where tracked, or `None` otherwise. For example,
one 10000-token request and one empty request produce a count of 2 and a token
weight of 10000. Finishing the long request leaves count 1 and token weight 0.

Consumers of the earlier experimental fork that treated `get_total_inflight()`
as token weight must migrate to `pending_prompt_tokens`. The selector's internal
token budget and exact token-release accounting are unchanged.

Callers omitting `reservation_id` use `request_id` as the reservation identity
and must keep it unique across active calls. Identity-free direct selector calls
retain the legacy FIFO release API. Never reuse reservation IDs: a delayed
release could otherwise match a later call.

## Server replacement and lifecycle

Adding/removing members or replacing an existing server ID's handle requires
all adapter reservations to drain. An update that would change the pool while
requests are active is rejected before any mapping changes. Re-registering the
same handle is allowed and preserves active accounting.

After draining, `add_servers({server_id: new_handle})` updates the handle without
changing that ID's candidate position. Replacement clears selector cache
affinity because the new engine cannot be assumed to own the old engine's KV.
This clears routing estimates, not device KV. It does not provide transparent
failover or recover reservations lost with an actor process.

A zero adapter count is not a complete P-to-D transfer barrier: other P routers,
unregistered work and native transfers are outside this instance's accounting.
The trainer and companion runtime must coordinate admission, sleep/wake and
weight updates. Lost acquire replies, release failures and actor failures do not
have distributed transaction or lease recovery here.

This adapter adds no private peer-disconnect RPC or Mooncake patch. Install a
vLLM-Ascend/Mooncake stack compatible with the companion's Connector; transport
memory unregister/reregister belongs to the runtime worker lifecycle, not to the
selector. Per-role memory and batch settings belong to the companion verl
configuration and are independent of this policy YAML.

## Validation

```bash
python -m pytest -q tests/test_decode_selectors.py tests/test_router_balancer.py --noconftest
pre-commit run --all-files --show-diff-on-failure
```

The CPU regressions cover policy loading, count/token separation, zero-token
requests, exact out-of-order release, duplicate/wrong-server release, active-pool
guards and same-ID handle replacement with affinity reset. They do not start
live Ray actors, inference engines or native KV transfers. Target-device
sleep/wake, weight-update and performance validation remain separate requirements.

AI assistance was used; human review and target-environment validation remain
required before merge.
