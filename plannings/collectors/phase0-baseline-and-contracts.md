# Phase 0 Baseline and Contracts

## Status

- Source baseline: `b588b683590fe8869df4b2cb5badfccdd66c8d14`
- Contract status: frozen for Phase 1 implementation
- Runtime status: no event runtime has been added
- Known baseline defect: the pinned veRL commit lacks the `merge_context_tokens()` method called by Gateway continuation code; full CPU level0 cannot be declared green until the dependency baseline is corrected independently

## Compatibility Baseline

| Boundary | Current owner | Behavior to preserve | Regression evidence |
|---|---|---|---|
| Task result | `GatewayAgentFramework` | Map `TaskResult.reward/accuracy/finished` to `Trajectory.reward_score/{"acc": value}/finished`; keep metrics when reward is absent | Framework CPU level0 cases for runner result, no reward, unfinished and failure paths |
| Gateway generation | `GatewaySession` | Return `GenerationOutcome`; backend work stays outside the session lock; token counts and finish reason retain current meaning | Gateway CPU level0 generation cases |
| Gateway finalization | `GatewaySession` through `_GatewayActor` and `GatewayManager` | External result remains `list[Trajectory]`; Gateway does not assign reward fields | Gateway Actor and Manager boundary cases |
| Gateway snapshot | `GatewaySession` | JSON-serializable session/chain/rollback view; timestamps are observed, not compared as fixed values | Gateway state case |
| Router decision | `KVCAwareBalancer` and current strategy | Preserve selected server ordering, sticky fallback and inflight accounting | Router unit, sticky and inflight cases |
| Router commands | `KVCAwareBalancer` | `acquire_server()` and `release_server()` remain synchronous, confirmable control operations | Router acquire/release cases |

The Phase 0 baseline suite is the smallest existing set that covers these boundaries. It must pass before and after each later phase; broader CPU level0 remains the final regression gate.

## Identity Rules

| Identity | Created by | Lifetime and propagation | Constraint |
|---|---|---|---|
| `run_id` | rollout composition layer | One rollout run; injected into process-local publishers | Never inferred from a process ID |
| `producer_id` | component composition layer | Stable logical Framework, Gateway, Runner or Router source | Identifies the source, not one restart |
| `producer_epoch` | source process/Actor startup | New value after every source restart | Old epoch state cannot become healthy without resync |
| `episode_id` | Framework before `_run_agent_episode()` | One actual rollout; passed to Gateway and Runner | Separate from prompt `uid` and `global_step` |
| `session_id` | Framework session creation | Stable for the Gateway session and sticky affinity | Never reused as `generation_id` or `attempt_id` |
| `generation_id` | Gateway when accepting one logical generation | Stable across delivery retry of that generation | Does not identify a repeated backend execution |
| `attempt_id` | Gateway immediately before backend execution | New for each real execution; reused only for transport retry of the same execution | Capacity and completion correlate by this ID |
| `chain_id` | Gateway prepare/commit path | Optional until a chain is selected or created | Late binding updates correlation; it does not create a new attempt |
| `operation_id` | Runner at operation start | One tool or sandbox operation | Optional outside Runner events |
| `event_id` | `EventPublisher` | One immutable fact | Retransmission keeps the original ID |
| `fragment_id` | Task Metrics Projector | One source fragment for an episode | A higher revision replaces the same fragment; it is not added twice |

Phase 1 may initially map `episode_id` one-to-one with the Framework-created Gateway session, but both fields remain explicit so later fan-out does not change the contract.

## Public Event Contract

The common layer owns only the envelope, context, subscriptions and transport DTOs. Framework, Gateway, Runner and Router own their payloads.

### Event envelope

| Field | Type | Owner and compatibility rule |
|---|---|---|
| `event_id` | `str` | Publisher; stable across retransmission |
| `event_type` | constrained string or Enum | Publishing module; new types are additive |
| `schema_version` | positive `int` | Payload owner; optional fields are backward compatible, unsupported major versions fail state sync |
| `run_id` | `str` | Composition layer |
| `producer_id` | `str` | Composition layer |
| `producer_epoch` | `str` | Source startup |
| `producer_seq` | non-negative `int` | Publisher; monotonic only inside one producer epoch |
| `context` | `EventContext` | Lifecycle owner; copied into the event |
| `occurred_at_unix_ns` | non-negative `int` | Publisher; for observation, not cross-machine ordering |
| `payload` | immutable mapping | Business module; copy/freeze before publish |

`EventContext` contains optional `episode_id`, `session_id`, `generation_id`, `attempt_id`, `chain_id`, `runner_name` and `global_step`. New optional correlation fields are additive. Ray boundaries carry versioned dict/list/scalar DTOs, never Bus instances, locks, callbacks or ContextVar tokens.

### Subscription contract

`SubscriptionSpec` contains `subscription_id`, `event_types`, `source_filter`, `scope`, `delivery`, `direct_target`, `max_queue_events` and `max_queue_bytes`.

- `scope`: `local`, `direct` or `global`.
- `delivery`: `inline`, `queued` or `state_sync`.
- `local + inline` is bounded and performs no I/O.
- `direct + state_sync` requires an authoritative snapshot and applied ACK.
- A remote endpoint calls `LocalEventBus.deliver()`; it never republishes the event.

## Fragment and Transport DTOs

### Metrics fragment

| Field | Rule |
|---|---|
| `episode_id` | Required correlation identity |
| `source_role` / `source_instance` | Identify Framework, Gateway or Runner fragment owner |
| `fragment_id` / `revision` | Idempotent replacement key |
| `schema_version` | Version the metric payload contract |
| `complete` / `incomplete_reasons` | Never replace missing data with zero |
| `metrics` | Numeric sequences or reduced statistics only; metadata stays outside |

### Direct state sync

- `DirectStateBatch`: `run_id`, `subscription_id`, `source_epoch`, `stream_epoch`, `first_seq`, `last_seq`, `events`.
- `DirectAck`: `stream_epoch`, `contiguous_cursor`, `stage=applied`, `status` in `OK/BUSY/RESYNC_REQUIRED`.
- `StateSnapshot`: `owner`, `source_epoch`, `scope`, `revision`, `watermark`, `source_health`, `current_entities`, `terminal_entities`.

The entity reducer and applied cursor commit together. Gaps, overflow, unknown schemas or epoch changes keep the view stale until snapshot installation completes.

### Global telemetry

- `TelemetryBatch`: `run_id`, `source_id`, `source_epoch`, `batch_id`, `events`.
- `TelemetryAck`: `batch_id`, `stage=accepted`, `status` in `OK/BUSY`.

Accepted means admission to a bounded queue, not Reporter persistence. Direct and Global never share queue, retry, ACK or health state.

## Single Writer Ownership

| Data | Before cutover | Shadow phase | Primary phase |
|---|---|---|---|
| Training `Trajectory` result fields | Existing Framework mapping | Existing mapping remains the only writer; fragments are comparison-only | Task Metrics merge writes once after parity is accepted |
| Gateway token trajectories | `GatewaySession` | Unchanged | Unchanged; a finalization DTO only wraps the list internally |
| Router DataStore, sticky and inflight | Existing collectors/callbacks | Router Projector computes comparison state without committing | One input family at a time moves to the Projector; old writer is removed in the same slice |
| Telemetry backend | Existing output path, if present | New Reporter uses an isolated comparison namespace | Exactly one configured Reporter owns each production series |

Shadow data must not re-enter Task aggregation or Router decisions. Every compatibility adapter is removed in the same phase that switches its consumer, or carries a named removal criterion.

## Configuration Boundaries

There is no master switch whose meaning changes by phase. Planned keys are independent:

| Key | Values | Default | Effect |
|---|---|---|---|
| `collectors.task_metrics.mode` | `off`, `shadow`, `primary` | `off` | Local Task Projectors and fragment merge only |
| `collectors.direct_state_sync.enabled` | boolean | `false` | Direct Bridge/Endpoint and source snapshots only |
| `collectors.global_telemetry.enabled` | boolean | `false` | Global forwarder, broker and Reporters only |
| `collectors.router_state.mode` | `legacy`, `shadow`, `projector` | `legacy` | Router input commit owner only |

Invalid combinations fail at startup. A control event cannot be configured with only a Global subscription.

## Performance Budget

| Measurement | Phase 0 baseline | Allowed regression |
|---|---|---|
| Focused compatibility suite wall time | 19.14 s for 32 tests on Python 3.11.14 | Informational; investigate a repeatable increase above 20% |
| Gateway generation | 2,000 samples: median 41.769 µs, P95 77.036 µs, P99 118.263 µs | P95 increase at most 2% or 0.5 ms, whichever is larger, using the same host and fake backend |
| Gateway local request handling | 2,000 samples: median 56.742 µs, P95 90.477 µs, P99 171.099 µs | P95 increase at most 2% or 0.5 ms, whichever is larger |
| Gateway finalization | 4,000 samples across both paths: median 5.999-6.529 µs, P99 18.102-19.945 µs | P95 increase at most 2% or 0.1 ms, whichever is larger |
| Router acquire/release microbenchmark | 20,000 samples after 1,000 warm-ups: median 35.660 µs, P95 49.861 µs, P99 72.990 µs | Median and P99 increase at most 2% before Router migration |
| Disabled feature path | Current behavior | No extra Actor/RPC/thread and no output difference |
| Phase 1 `LocalEventBus.publish()` with no matching subscription | Not present yet | P99 at most 25 microseconds on the baseline host |
| Phase 1 one local inline Projector | Not present yet | P99 at most 100 microseconds; handler performs no I/O |

Performance comparisons use the same interpreter, host, test data, warm-up and sample count. A noisy one-off result is rerun; budgets are not relaxed from a single measurement.

The Router sample uses two replicas, four prompt tokens, 100 rotating request IDs and an acquire/release pair per sample. Loguru handlers are removed after construction and verified absent before timing; the final inflight count must be zero.

The Gateway sample uses a local `_GatewayActor`, `FakeTokenizer`, an immediate two-token backend, 200 warm-ups and fresh single-generation sessions. Generation measures `GatewaySession.run_generation()`; request handling measures the normalized OpenAI adapter-to-session path; finalization is measured separately. No HTTP socket or Ray startup time is included, and the final live-session count must be zero.

## Validation Commands

```bash
pytest -q \
  tests/uni_agent/gateway/test_gateway_actor_on_cpu.py::test_gateway_actor_get_session_state_reports_default_chain_tips \
  tests/uni_agent/gateway/test_gateway_actor_on_cpu.py::test_gateway_actor_finalizes_unannotated_trajectories \
  tests/uni_agent/gateway/test_gateway_manager_on_cpu.py::test_gateway_manager_finalizes_each_session_on_its_owning_gateway \
  tests/uni_agent/framework/test_generate_sequences_on_cpu.py::test_runner_reward_is_used_without_custom_scorer_even_when_worker_exists \
  tests/uni_agent/framework/test_generate_sequences_on_cpu.py::test_metrics_survive_without_any_reward_source \
  tests/uni_agent/framework/test_generate_sequences_on_cpu.py::test_generate_sequences_reports_unfinished_episode_count \
  tests/uni_agent/framework/test_generate_sequences_on_cpu.py::test_generate_sequences_marks_prompt_failure_when_all_sessions_fail \
  tests/uni_agent/agent_aware_router/balancer/test_balancer_unit.py \
  tests/uni_agent/agent_aware_router/balancer/test_balancer_inflight.py \
  tests/uni_agent/agent_aware_router/balancer/test_balancer_sticky.py \
  -m "cpu and level0" --durations=20

pytest -q tests/uni_agent/ -m "cpu and level0"
```

The project declares Python 3.10-3.12 support. GitHub CPU level0 CI currently runs 3.11 and 3.12; Python 3.10 is a separate compatibility run until it is added to the matrix.

The exact compatibility suite is expected to stay green even while the independent continuation-token dependency mismatch remains open. Full CPU level0 is still required before Phase 0 can be marked complete; the narrower suite does not waive that exit criterion.
