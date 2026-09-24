# Collectors Implementation Findings

## 2026-09-20 — Phase 4 Recovery

- Phase 4 is an ownership migration, not a new transport phase: Direct and Global contracts remain unchanged.
- The frozen Router modes are `legacy`, `shadow`, and `projector`, defaulting to `legacy`; the mode selects the Router input commit owner only.
- `legacy` must preserve current callback/DataStore behavior. `shadow` computes comparison state without affecting routing, sticky, inflight, or KV decisions. `projector` can become authoritative only for an explicitly migrated input family.
- A migrated Router input family cannot retain both its callback writer and Projector writer. The compatibility writer is removed in the same vertical slice that switches the consumer.
- `acquire_server()` and `release_server()` remain synchronous, confirmable Router commands; they are not converted into lossy event delivery.
- Admission work in this phase is observe-only. A future `AdmissionSignalsProjector` may correlate lifecycle/request features and Router capacity-change facts, but Router remains the sole capacity ledger and Grant owner.
- The formal sequence is Router migration first, admission observe integration second; enforce-mode Coordinator, predictors, dynamic throttling, and eviction policy remain outside this phase.
- Router transport/parser/hash translation stays in place. The migration seam is after typed parsing and before the single DataStore commit boundary.
- Router Projector failures on capacity-relevant local input must mark the projected view unhealthy; they cannot be swallowed while later code assumes the state is safe for Grant decisions.
- Capacity-change facts are emitted only after Router state commits. Admission observers can use them as retry notifications, but those events are not authorization and cannot independently allocate capacity.
- Phase 4 admission evidence can cover correlated Gateway lifecycle/generation observations and committed Router capacity notices. It must not claim an authoritative capacity mirror or implement `try_admit`.
- The design acceptance boundary includes epoch/gap/freshness handling, behavior parity, single-writer proof, and no regression to Direct/Global queue isolation.
- The existing callback input path is already typed as `StickyUpdate` and delta `MetricsUpdate`, but `Collector` commits both directly to the singleton-backed `DataStore`.
- Sticky bindings are the smallest complete Router input family for the ownership cutover: `on_acquire` creates/refreshes a request binding and `on_servers_removed` invalidates a replica. They have no exporter side effects, so shadow parity and exact single-writer cutover can be proven without mixing in inflight accounting.
- Inflight counters remain on their existing synchronous callback writer in this slice. The Balancer's own `_inflight` ledger remains the command-side truth and is updated before any capacity observation is published.
- The repository has no AdmissionCoordinator or admission package yet. Phase 4 should add a movable observe-only `AdmissionSignalsProjector` behind Router facts, not invent an enforce path or a second capacity ledger.
- Default legacy mode must not allocate a Router event bus, publisher, Projector, admission observer, thread, Actor, or RPC.

## 2026-09-20 — Retrospective Commit Audit

- `HEAD` is `1fad722` on branch `collector`; no collector phase commit exists yet and the index is clean.
- Phase 1-3 changes overlap in Framework entry, Gateway config/actor/manager, and their tests. Whole-file staging would misattribute later Direct/Global wiring to Phase 1.
- The transport cores and their focused tests are naturally separable: local event/metrics files belong to Phase 1, `events/direct.py` plus Router session shadow belong to Phase 2, and `events/global_bus.py` plus `telemetry/` belong to Phase 3.
- `verl`, `working/`, the automatic root `.planning/`, and unfinished Phase 4 files are outside all retrospective commits.
- The safe method is to construct and validate historical Phase 1/2/3 snapshots in a temporary worktree, then move the `collector` branch forward and leave the current worktree containing only excluded and unfinished changes.
- Blob comparison against the Phase 3 commit confirmed every committed path matched the current worktree except `events/__init__.py` and `events/names.py`, whose only remaining differences are Phase 4 exports.
- A mixed index refresh after moving the branch left no completed Phase 1-3 source or test changes unstaged; this proves the retrospective split captured the implemented phase content without absorbing unrelated files.
- The Phase 4 draft currently defines the sticky ownership and admission observation reducers but has not yet wired them into Collector composition or `KVCAwareBalancer` lifecycle.
- The existing `get_collector(..., balancer_handler=...)` seam can select a sticky handler/observer from the Balancer without changing `CollectorManager` or the provider-factory signature.
- `clear_sticky_cache()` is part of the same sticky ownership family and must update the projected view; otherwise projector mode would regain a compatibility writer and shadow parity would become stale.
- Admission route history and terminal lifecycle state require explicit retention bounds. Status must distinguish total observations from retained observations and remain explicitly non-authoritative.
- In projector mode a reducer fault should fail closed for later acquire calls. The current callback API swallows exceptions, so the Balancer must inspect Projector health after the synchronous callback boundary before returning a selected server.
- The completed Sticky slice keeps exactly one DataStore writer in each mode: legacy writes directly, shadow observes only after that write, and projector bypasses the compatibility writer.
- Router facts are published only after the synchronous command and Sticky commit boundary. Observation failures are counted but cannot roll back or block an already committed Router command.
- `AdmissionSignalsProjector` remains movable and observe-only: it retains bounded route/session evidence and capacity notices, but exposes `authoritative=false` and owns no Grant, Hold, capacity mutation, or scheduling API.
- The Direct Gateway lifecycle endpoint composes the existing Router session shadow first and the admission observer second; it retains the existing applied-ACK/snapshot boundary and does not republish.
- Default legacy mode creates no Router bus, publisher, Projector, admission observer, capacity-version map, replica epoch, thread, Actor, or additional config RPC.
- A same-epoch capacity notice must advance `ledger_version`; a new `replica_epoch` may restart the version sequence. This prevents stale same-instance observations from replacing newer state without treating a restarted replica as stale forever.
- The legacy microbenchmark remained aligned with the Phase 3 parent under same-run measurements after empty-mode short-circuiting; historical P99 remains noisy, so Phase 5 should compare paired runs rather than a single absolute tail sample.

## 2026-09-20 — Phase 3 Recovery

- Phase 3 owns Global telemetry only: asynchronous batching, accepted ACK semantics, independent backpressure, remote local delivery, and isolated reporting.
- The completed Direct source/endpoint contracts remain independent and must not be reused for Global acknowledgement or recovery semantics.
- The implementation must reuse existing Gateway facts and repository Reporter abstractions where available; it must not introduce a parallel event vocabulary merely to exercise the transport.
- The current worktree is intentionally uncommitted across Phase 1/2 and unrelated user areas, so Phase 3 edits must remain narrowly scoped and preserve all existing changes.
- The formal Global contract is `TelemetryBatch(run_id, source_id, source_epoch, batch_id, events)` with `TelemetryAck(batch_id, stage=accepted, status=OK|BUSY)`; accepted means admission to the next bounded queue, never Reporter persistence.
- Source, broker mailbox, subscriber ingress, and pending remote calls all require independent limits. Initial design budgets are 128 events, 256 KiB, and 20 ms, but they must remain configurable rather than hard-coded policy.
- Global Publish and Subscription Endpoints validate and enqueue only. Exporter or Reporter calls belong behind subscriber-local queues and must never execute inside the remote receive RPC.
- Global remote delivery must call the named local `deliver()` ingress so received events cannot re-enter the forwarder and form a loop.
- Observation loss may be explicit under pressure, while final metric fragments require bounded retry or an incomplete outcome. Phase 3 therefore needs event-family loss policy rather than treating every Global event as control state.
- A single slow exporter must not block publishers, the broker, or other subscribers; per-subscriber queues and health counters are part of the acceptance boundary.
- The repository has an RL-Insight facade and trace adapters, but no reusable asynchronous telemetry Reporter contract. Calling those facades inside either Global Endpoint would violate the formal endpoint boundary.
- The existing Gateway event family and `MetricsFragment` models are the available production facts. Phase 3 should add a small generic Reporter boundary behind subscriber-local delivery rather than couple the transport core directly to RL-Insight APIs.
- `LocalEventBus` already provides independently bounded queued subscribers, `deliver(subscription_id, batch)` filter revalidation, drain reports, and isolated handler failure counts. Global Subscription Endpoint can reuse these contracts without adding a second local queue implementation.
- Gateway currently creates one local bus when either Task Metrics or Direct is enabled. Global enablement can share that actor-local ingress while remaining independently gated and independently queued.
- Framework entry is the existing driver-side composition boundary. It can create shared Global broker/subscriber actors only when enabled and inject the publish target into GatewayManager, preserving a zero-Actor default path.
- The Direct bridge is a single blocking worker around an injected callable. Global needs a distinct implementation because accepted batches may be retried but do not carry applied cursors, and batching must wait up to a configurable flush interval.
- `GatewayManager` already owns Gateway actor startup/shutdown and generates a shared run ID for enabled cross-Actor delivery. It can pass a shared externally-created Global publish target while continuing to construct Gateway actors with the exact legacy signature when all collector transports are off.
- Gateway shutdown already centralizes event runtime cleanup, so the Global source subscription/worker should be closed there before the local bus is drained and closed.
- Phase 0 freezes only the independent `collectors.global_telemetry.enabled` switch and transport DTO fields; queue/batch tuning can be additive config while preserving `false` as the no-runtime default.
- Phase 0 requires a new Reporter to use an isolated comparison namespace until ownership changes. Phase 3 must not write existing production series or feed shadow telemetry back into Task/Router state.
- A subscriber endpoint needs an atomic batch ingress before local `deliver()`: directly calling a capacity-limited queued subscription can partially enqueue a batch, making BUSY retry ambiguous. A bounded endpoint batch queue plus one local delivery pump preserves accepted semantics and exposes downstream loss separately.
- Broker fan-out requires one bounded queue and worker per subscription. This makes one slow/hung target consume only its own pending-call budget and prevents it from serializing other subscribers.
- The selected vertical slice forwards existing Gateway `SessionOpened`, `GenerationFinished`, and `SessionClosed` events. It does not add business events or repurpose Task Metrics fragments.
- The first Reporter is a shadow event-count Reporter in a separate Ray actor. It proves end-to-end isolation and exposes counts/health without creating or claiming ownership of existing RL-Insight series.
- Driver-side `GlobalTelemetryRuntime` will own exactly one broker actor and one shadow Reporter actor, register the Reporter before Gateway startup, and shut down in source → broker → Reporter order.
- Global transport core remains Ray-agnostic through injected callables. A small telemetry runtime adapter owns Ray handles, keeping transport tests deterministic and future Reporter actors replaceable.
- Review confirmed that endpoint admission must verify its named local subscription exists before starting the receive pump; later subscription loss is then reported as downstream incompleteness rather than a false startup success.
- Deduplication identity must include `run_id` as well as source ID, source epoch, and batch ID so a retained bounded dedup window cannot collide across runs.
- Runtime configuration must preserve explicit event-family filters through driver → broker/Reporter actor serialization; defaults alone are insufficient for future extension.
- Manager cleanup must drain the Global runtime even if one Gateway shutdown fails, and runtime startup failure must not leave its two newly-created actors behind.
- BUSY observability counts all endpoint rejections, including unknown subscriptions and batches rejected before queue admission for size limits; queue saturation is not the only BUSY cause.
- Phase 3 status snapshots intentionally remain diagnostic rather than transactional: broker and Reporter live in separate actors, so callers poll a convergence predicate instead of comparing one cross-Actor instant.
- The final slice adds no Router/Admission consumer or writer and no production telemetry metric names. Its shadow event-count Reporter is an isolated replaceable sink behind the generic subscription endpoint.
- With the remote call deliberately blocked, Global source enqueue measured 4.156-5.973 us median and 11.811-27.316 us P99 across three 20k-event batches; publisher latency remains decoupled from the broker.

## 2026-09-20 — Phase 2 Recovery

- Phase 2 is the implementation-plan phase named `Direct State Sync`; it is not the completed internal discovery phase from the separate automatic plan.
- The formal architecture requires `EventPublisher -> LocalEventBus -> Direct RayEventBridge -> Direct EventEndpoint -> target LocalEventBus` and forbids a remote endpoint from republishing.
- Direct state synchronization owns a separate bounded outbox, stream epoch, applied cursor, retry and health state; none of these may be shared with the later Global telemetry path.
- Queue overflow, a sequence gap, an unsupported schema, or a source/stream epoch change invalidates the current control view and requires snapshot resynchronization.
- Direct events synchronize state for the next decision; they do not replace synchronous commands with return values or atomic semantics.
- Phase 2 will first identify the smallest existing control-state view that can run in shadow mode without changing its authoritative writer or decision path.
- Direct ordering uses a per-subscription `(subscription_id, stream_epoch, stream_seq)` assigned after filtering; `producer_seq` cannot be used as the transport cursor because unrelated filtered events create valid gaps.
- An applied ACK is returned only after the target state-sync handler has committed the entity reduction and cursor together. A local queued-delivery acknowledgement is not sufficient.
- The source must capture snapshot plus watermark at the authoritative state boundary and retain later matching changes while the receiver installs it. A downstream projector cannot manufacture the recovery snapshot from its potentially incomplete view.
- The current `LocalEventBus.deliver()` already targets exactly one subscription and avoids forwarding loops. Phase 2 should layer transport validation/recovery around this ingress instead of changing ordinary local publish semantics.
- The existing `DeliveryMode.STATE_SYNC` executes synchronously like inline delivery, which gives the endpoint a usable commit boundary once the handler reports success; queueing and retry therefore belong to the source-side Direct Bridge.
- Existing event DTOs are Ray-serializable and immutable in process. Direct transport DTOs can reuse `Event.to_dict()/from_dict()` while enforcing separate transport size and schema limits at the endpoint.
- `KVCAwareBalancer` is a plain class wrapped by veRL as a Ray Actor. Its current callback collectors and `DataStore` remain authoritative, so a Phase 2 Router-side lifecycle view must be comparison-only and must not write inflight, sticky, KV, or routing state.
- Gateway already emits `SessionOpened`, `GenerationFinished`, and `SessionClosed` with monotonic per-session revisions. This is the smallest existing event family suitable for a shadow Router control-state view.
- Gateway owns one process-local bus per Actor only when task metrics is enabled today. Direct state sync must have an independent configuration gate so enabling Direct does not imply task-metric collection, while both features can share the same Actor-local bus and publisher when enabled together.
- A reusable Direct bridge cannot place Ray handles inside event DTOs or subscription specs. The composition layer should inject the target send/snapshot callables; the common event package owns only bounded transport/recovery state.
- veRL's `LLMServerClient` retains the shared Router Actor handle as `_load_balancer`; `build_gateway_manager()` is the existing composition boundary that can extract and inject that handle only when Direct sync is enabled.
- The Router class has no explicit shutdown hook and is serialized by `ray.remote`; the Phase 2 endpoint should therefore be lazy and threadless on the receiver side. Source-side bridge cleanup belongs to the already-existing Gateway shutdown lifecycle.
- Gateway actors are started before sessions can be created. Installing an initial empty authoritative session snapshot during `GatewayActor.start()` provides a race-free first handshake; later lifecycle events can then be streamed by per-source cursor.
- The receiver can remain threadless: `KVCAwareBalancer` lazily creates one local bus, endpoint, and shadow projector when the first snapshot arrives. The source Bridge owns the only transport pump and the Gateway shutdown path closes it.
- The authoritative Gateway snapshot is maintained alongside session lifecycle publication, not reconstructed from the Router shadow. During resync the Bridge opens a new stream before the actor captures the snapshot, so events published while the snapshot RPC is in flight receive post-watermark stream sequences.
- Multiple Gateway actors share one Router subscription ID but retain independent producer/source epochs and stream cursors. The endpoint keys transport state by subscription plus source epoch and the projector keys entities by owner plus session ID.
- Direct DTO parsing failures at the Router boundary must return `RESYNC_REQUIRED` rather than raise across Ray and trigger an unbounded source retry loop.
- The Bridge retains the same event IDs and stream sequence while retrying an unacknowledged batch. Queue overflow clears pending deltas and marks the stream stale; the Gateway maintenance task then opens a new stream and installs a fresh owner snapshot.
- Gateway lifecycle events now carry a source-wide monotonic revision in addition to the per-session revision. This lets Router health report source progress even when many entities each use the same local `1 -> 2` lifecycle revisions.
- Terminal snapshot retention is explicitly bounded and configurable; it protects the agreed retry/recovery window without allowing completed sessions to grow source memory indefinitely.
- Direct disabled mode must preserve the original `GatewayActor.remote(config, backend=...)` construction shape. Extra Direct constructor arguments are now passed only when the feature is enabled, keeping existing Actor fakes and external call seams compatible.
- Broad Gateway results cleanly separate Phase 2 behavior from baseline defects: all non-affected tests pass after the constructor fix; the remaining deselections are the already-recorded veRL continuation API mismatch and stdout-vs-caplog behavior.
- `LocalEventBus.deliver()` now revalidates the named subscription's filters without republishing. This prevents a malformed or mismatched remote batch from advancing an applied cursor merely because it named a valid local subscription.
- A snapshot can make a stream healthy only when the source reports healthy, and its revision cannot move backward within the same source epoch. A new source epoch may start from its recovered owner revision after a fresh handshake.
- The Router shadow preserves `close_reason` separately from the normalized terminal status, so finalized and aborted sessions remain distinguishable without affecting routing decisions.
- Direct source enqueue measured 10.853-11.845 us P99 while a remote call was deliberately blocked, confirming publisher latency is decoupled from the target. The frozen Router benchmark's median improved, but P99 remained noisy above its historical threshold even though Phase 2 does not modify `acquire_server()` or `release_server()`.

## 2026-09-20 — Phase 1 Recovery

- `plannings/collectors/` initially contained only `phase0-baseline-and-contracts.md`; the planning lifecycle files were missing.
- A separate root `.planning/` plan reports a completed local event foundation and Gateway metrics integration. The corresponding source and tests are present as uncommitted work, so Phase 1 begins with an audit rather than duplicate implementation.
- The current worktree contains unrelated `verl`, `working/`, and planning changes. Phase 1 must not restore, delete, or reformat them.
- The formal architecture keeps `EventPublisher -> LocalEventBus` as the common ingress. Direct and Global delivery have separate reliability semantics and are outside Phase 1.
- Remote delivery will eventually use `LocalEventBus.deliver()` rather than republishing, preventing forwarding loops.
- Phase 0 froze explicit producer, episode, session, generation, attempt, event, and fragment identities. Phase 1 must preserve those identities instead of deriving them from process-local accidents.
- The pinned veRL source snapshot is incompatible with current continuation code because it lacks `merge_context_tokens()`. This affects broad regression collection but is not a reason to alter collector contracts.

## Audit Questions

- Does the local event API enforce immutable event data and isolated context overlays?
- Are queued subscriptions independently bounded by event count and bytes?
- Does `_GatewayActor` own lifecycle and cleanup for the bus, publisher, and projector?
- Does every caller-level generation completion emit exactly one fact while owner-only timings remain owner-scoped under request coalescing?
- Is the finalization DTO internal while old Manager callers still receive lists?
- Is the metrics fragment attached only after Framework selection and postprocessing?
- Do empty, failed, cancelled, and capacity-exhausted sessions remain distinguishable without fabricated zeros?

## Phase 1 Audit Notes

- The candidate `uni_agent.events` implementation already provides a frozen envelope, recursive payload copying, ContextVar binding, lock-protected publisher sequencing, and DTO conversion.
- `GatewayTaskMetricsProjector` keeps fixed-size accumulators per session and subscribes locally to `SessionOpened`, `GenerationFinished`, and `SessionClosed`; metric names remain inside the projector.
- Two contract differences require resolution before acceptance: `Event` currently rejects zero for `producer_seq` and `occurred_at_unix_ns` although Phase 0 defines both as non-negative, and `MetricsFragment` exposes `errors` while the frozen fragment contract names `incomplete_reasons`.
- These are narrow model-contract issues; they do not justify expanding Phase 1 into Direct or Global transport work.
- The local bus snapshots matching subscriptions before dispatch, isolates handler failures, bounds each queued subscriber by count and estimated bytes, and implements `deliver()` as named local ingress without republishing.
- The formal design confirms `SessionFinalizationResult(trajectories, metrics_fragment)` as an internal Gateway boundary and requires Framework to attach the fragment after trajectory selection/postprocessing.
- The frozen Phase 0 artifact is the precise field-level source for Phase 1, so the fragment DTO should expose `incomplete_reasons`; a legacy `errors` input/property may be retained only as a compatibility alias.
- `_GatewayActor` currently owns exactly one bus, publisher, and projector, and `GatewaySession` receives only an optional publisher plus immutable base context. The Manager retains the list-returning wrapper and Framework attaches the fragment after selection/postprocessing.
- The candidate integration is currently unconditional: every Gateway actor constructs the event runtime and Framework attaches a fragment whenever the new Manager method is present. This conflicts with the frozen default `collectors.task_metrics.mode=off` and the Phase 1 disabled-path requirement.
- Configuration gating is therefore a Phase 1 correctness gap, not a new feature. The smallest fix is to gate local Task Metrics at the Gateway actor configuration boundary while retaining the old finalization method and a `None` fragment when disabled.
- The existing runtime configuration is assembled in `build_gateway_manager()` from `actor_rollout_ref.rollout.custom.agent_framework`; the frozen key can be represented there as `collectors.task_metrics.mode` and passed to `GatewayActorConfig`.
- `off` can avoid constructing the bus, publisher, or projector entirely. `shadow` and `primary` can share the same local collection path in Phase 1 because the fragment remains an independent envelope and does not replace Framework-owned reward fields.
- Framework's legacy fake managers already exercise the old list-only fallback, while the real Manager can return `SessionFinalizationResult(..., metrics_fragment=None)` in `off` mode without adding an Actor, thread, or extra RPC.
- The existing read-only veRL export remains available at `/tmp/uni-agent-verl-phase0.UgGIt1`; it can supply deleted submodule sources during integration tests without mutating the user's `verl` worktree.
- The active Python is 3.11.14, and its Conda runtime provides the newer `libstdc++` needed by the Ray/SciPy test imports.
- Multi-batch measurement confirmed the no-subscriber P99 budget miss; it was not a one-off. The inline projector was near the limit and passed in later batches, but remained noisy.
- Profiling the no-subscriber path shows avoidable work in `Event.__post_init__`, recursive size estimation, and two `EventContext.overlay()` calls. The safe optimization boundary is internal allocation reduction: cache context field metadata, reuse the immutable empty payload, skip empty overlays, and estimate context directly.
- After that internal optimization, five consecutive batches passed both frozen performance budgets: no-subscriber P99 was 20.598-23.452 us and inline Gateway Projector P99 was 61.177-67.870 us.
- Final source review found no Direct/Global transport implementation in the Phase 1 slice. Existing `Scope.DIRECT`/`Scope.GLOBAL` values and `deliver()` remain local contracts only.
- The default path now constructs no bus, publisher, projector, Actor, RPC, or thread and returns `metrics_fragment=None`; `shadow` or `primary` explicitly enables the local runtime.

## Sources

- `plannings/collectors/phase0-baseline-and-contracts.md`
- `/home/hgq/workspace/aicoder/research/autoresearch/agenticrl/llm-router/04_design/collectors/event-pubsub-detailed-design.md`
- Existing candidate implementation under `uni_agent/events/`, `uni_agent/metrics/`, Gateway, Manager, and Framework

## 2026-09-20 — Phase 5 Recovery

- The formal design assigns final fragment merging to Framework after trajectory selection and postprocessing; it explicitly requires zero-trajectory and failed episodes to reach prompt summary so the failure denominator is retained.
- Phase 5 is a closure slice rather than a new transport or ownership phase. Direct, Global, Router, and Admission boundaries established in Phases 2-4 remain unchanged.
- Compatibility cleanup is evidence-gated: remove an adapter only when in-repository callers and focused mixed-version tests demonstrate parity; preserve public Gateway list-return behavior where it remains a supported boundary.
- The clean phase boundary is the existing Phase 4 commit `79c7abc`; unrelated `verl`, `.planning/`, and `working/` state is excluded from Phase 5.
- Framework currently carries a Gateway fragment only by attaching it to the last retained trajectory. If selection or postprocessing yields an empty list, `_run_agent_episode()` returns no fragment and `_run_prompt_rollouts()` records only an `empty trajectories` string.
- Prompt terminal state is currently a bare TransferQueue tag containing only `status=finished|failure`; batch logs keep aggregate counts but there is no per-prompt summary containing episode outcomes or available fragments.
- The existing `finalize_session_result` feature check is the necessary mixed-version adapter: an older GatewayManager returns `list[Trajectory]`, while the current manager returns `SessionFinalizationResult`. Phase 5 should test both paths and retain the public list-returning Gateway boundary.
- A Framework-internal episode outcome can preserve trajectories, fragment, completion state, and bounded failure information through selection/postprocessing without changing public Gateway DTOs or event transports.
- Gateway already produces complete fragments for zero-request, failed-request, and cancelled-request sessions when they are finalized. The current abort path instead discards projector state, so Framework runner failures cannot retain otherwise available observations.
- Abort collection needs a compatibility-preserving internal result path: keep `abort_session()` behavior for existing callers and let current Framework consume an additive result method when present, with a fallback to the legacy abort method.
- Prompt aggregation must retain one logical contribution per fragment ID at the highest revision and preserve missing/incomplete state. Numeric reduction must first reduce each episode/session, then aggregate sessions, rather than weighting prompts by generation count.
- `PromptMetricsSummary` is gated by `collectors.task_metrics.mode`; `off` keeps the existing prompt terminal tag byte-for-byte, while `shadow` and `primary` record bounded outcome counts, fragment count, completeness, and reduced metrics.
- Prompt reduction uses each fragment metric's scalar `value` as one episode contribution. This preserves equal episode weighting even when fragments contain different raw observation counts.
- Duplicate episode observations and fragment revisions are idempotent; the highest revision replaces an older fragment, conflicting equal revisions or aggregation types make the summary incomplete instead of guessing.
- The additive `abort_session_result()` method returns observations captured before failure, while `abort_session()` continues returning `None` and `finalize_session()` continues returning `list[Trajectory]`.
- Compatibility inventory found no adapter that is both expired and safe to remove: public list-return finalization, Framework feature detection, and legacy `errors` DTO parsing all still have named mixed-version purposes.
- `GatewayTaskMetricsProjector.discard_session()` no longer has an in-repository caller after abort fragments became observable, but the projector class is publicly exported. Remove it only after an external API/deprecation review, not from absence of local references alone.
- veRL replay-buffer synchronization reads the existing `status` key and otherwise preserves or ignores additional tag metadata. The added `agent_metrics_summary` does not enter trajectory tensors or alter terminal-group state transitions.
- The exact compatible Phase 0 suite completed faster after Phase 5 than in the pre-change run. Enabled prompt aggregation for four episodes with two metrics each measured 19.264-19.434 us median and 60.912-87.766 us P99 across isolated 20k-sample batches.
- Phase 5 changes no Router, Direct, Global, or Admission hot path. Their focused recovery/isolation regression passed, and the previously recorded Phase 1-4 performance evidence remains applicable.

## 2026-09-20 — Phase 6 Recovery

- The formal design still names trainer consumption and end-to-end tracking coverage as follow-up work after Framework fragment merging.
- Phase 5 stops at a versioned `agent_metrics_summary` stored in the prompt terminal TransferQueue tag; repository code outside Framework tests does not consume that field yet.
- Phase 6 therefore targets the Framework-to-trainer handoff and export boundary, not a new Local, Direct, Global, Router, or Admission transport.
- The smallest safe slice must keep `collectors.task_metrics.mode=off` byte-for-byte compatible and prevent `shadow` mode from becoming a second production exporter.
- TransferQueue tags are the current prompt-level ownership boundary. The next audit must determine whether the pinned trainer retains terminal tag metadata or needs a narrow versioned adapter.
- The pinned trainer's replay buffer reads all prompt tags but retains only `status` and `global_steps`; prompt-level metrics disappear before `KVBatchMeta` reaches metric computation.
- The replay-buffer `sample()` result already includes an auxiliary metrics mapping that the trainer merges into its step metrics, so Phase 6 does not need to modify the trainer loop or tracking logger.
- veRL exposes `trainer.v1.sampler.custom_sampler.{path,name}` as the supported extension seam. A uni-agent replay-buffer adapter can consume terminal prompt summaries without modifying the user-dirty `verl` submodule.
- Sync and async replay buffers have different selection behavior, so a reusable reducer should remain independent while thin sync/async adapters preserve each upstream sampler implementation.
- `PromptMetricsSummary` serializes a versioned bounded DTO but lacks a parser; Phase 6 needs a strict `from_dict()` boundary before trainer code can safely consume TransferQueue metadata.
- Prompt summaries retain sufficient statistics for correct cross-prompt reduction: `MEAN` combines totals/counts, `SUM` totals, `MIN` minima, `MAX` maxima, and `LAST` follows deterministic selection order.
- Trainer export must report coverage and incomplete/invalid summary counts separately. Missing metrics stay absent from the value reduction rather than contributing a fabricated zero.
- The replay-buffer adapter is optional and must not be imported from `uni_agent.metrics.__init__`; the local CPU environment does not install TransferQueue, while production veRL does.
- A sampler-side `primary` flag alone is insufficient to prove export ownership. Framework primary tags need an explicit trainer-owner marker so shadow summaries can never be exported accidentally.
- The adapter's contract can be verified without mutating or installing into the dirty veRL submodule by loading it against a bounded fake of the documented replay-buffer seam.
- Framework primary mode now places an explicit `trainer` owner marker next to the summary; shadow keeps the same summary for comparison but carries no export ownership.
- Trainer reduction emits a stable set of summary counters, metric values under a separate namespace, and per-metric prompt/observation coverage. Aggregation conflicts omit the value and surface as both a bounded counter and an incomplete reason.
- The public replay-buffer mixin lets deployments compose prompt export with an existing custom sampler instead of installing competing samplers.
- Primary trainer reduction for four prompts with two metrics each measured 60.689-72.245 us median and 167.159-262.754 us P99 across three isolated 20k-sample batches. This runs once per sampled prompt group batch, not on event publication or request hot paths.
- The adapter's `off` and `shadow` contract performs no extra TransferQueue metadata read; production export additionally requires the Framework's explicit `trainer` ownership marker.
- veRL combines sampler metrics again across `parameter_sync_step` using metric-name aggregation rules and trajectory-count weighting. Returning generic flat values would silently distort prompt-level counts and `MEAN`/`LAST` semantics when the sync step contains multiple samples.
- Phase 6 must encode `sum`, `min`, and `max` in tracking keys so the existing trainer preserves them. `MEAN` and `LAST` cannot be exact through this seam because it carries neither a custom weight nor a generic last rule; omit them with explicit unsupported counters/reasons instead of exporting a guessed value.
- Current production Gateway projector metrics all use `AggregationType.SUM`, so the initial end-to-end slice remains complete while preserving a clear extension point for a future weighted trainer contract.
- After trainer-safe key validation, the production-shaped reducer measured 71.766-75.029 us median and 208.050-250.928 us P99 across three isolated 20k-sample batches.
- Final scope review confirms Phase 6 changes only the prompt DTO boundary, Framework terminal-tag ownership, trainer-side metrics adapters, focused tests, documentation, and `plannings/collectors/`; the dirty `verl`, `.planning/`, and `working/` state remains untouched.

## 2026-09-20 — Phase 7 Recovery

- The formal design's remaining implementation gaps after Phase 6 are: Router migration for the inflight and KV/polled-metrics input families (design 7.2 / delivery step 3), and admission enforce (design 8.3-8.5), which is explicitly gated behind the separate capacity-admission acceptance threshold and must not start from this plan.
- The inflight input family is the `inflight_stat` collector: Balancer `on_acquire`/`on_release` callbacks → `CallbackTransport` `StatisticEvent` → `InflightParser` delta `MetricsUpdate` → `Collector._write_metrics_update` commits to the store.
- `Collector._write_metrics_update` is the family's legacy writer and does more than a store write: it folds per-request turn and prompt-length rows into the same batched `incr_metrics`, forwards insight `WriteEvent`s (ACQUIRE/RELEASE kinds), and triggers the throttled `router-dispatch` log line.
- The store behind both the Balancer's and the Collector's `DataStore` wrappers consists of process-wide singletons (`PerReplicaStore`/`KVCacheStore`/`PerRequestStore`), so projector parity checks can compare across the two wrappers; tests must reset the singletons per case (Phase 4 test convention).
- The Balancer's `_inflight` dict is command-path state updated synchronously inside `acquire_server()`/`release_server()`; it is not a callback input family and remains the capacity-fact source for `REPLICA_CAPACITY_CHANGED` payloads.
- Release-side `INFLIGHT_TOKENS` and `INFLIGHT_TURN_SUM` deltas are folded from acquire-time per-request rows because verl #7115 releases carry only `request_id`; the rows persist after release, so a post-commit observer can recompute the folded deltas read-only for parity.
- `MetricsUpdate` also serves absolute polled updates from the `vllm_metrics` collector; only the `inflight_stat` collector may receive an inflight handler/observer, and the dispatch guard should keep absolute updates on the legacy write path.
- Phase 4's sticky wiring precedent: `get_collector` reads `_router_sticky_update_handler`/`_observer` attributes off the Balancer, the Collector dispatches handler-before-observer, and the handler's presence removes the Collector's own write for that family in the same slice.
- `test_legacy_mode_allocates_no_router_projection_runtime` asserts the exact legacy `get_router_state_status()` dict; adding an inflight status key requires updating that Phase 4 assertion.
- `COLLECTOR_NAMES` includes `inflight_stat`, so the inflight collector runs in every Balancer construction; legacy mode must keep constructing it with unchanged behavior.

## 2026-09-20 — Phase 7 Implementation

- The inflight family's commit semantics live in one shared `commit_inflight_delta(store, update)` (fold → batched `incr_metrics` → insight `WriteEvent`); both the legacy Collector delta branch and `RouterInflightStateProjector.apply` call it, so owner cutover cannot change store state, telemetry output, or log cadence.
- The throttled `router-dispatch` line is likewise shared (`log_dispatch_stats(store, last_log) -> float`); each owner keeps its own throttle timestamp, which preserves the ≥5 s cadence per owner instead of per process.
- Per-request row keys are shared constants (`TURN_ROW_KEY`, `PROMPT_LEN_ROW_KEY`); the projector's comparison fold reads the same rows post-commit (release never deletes them), so effective-delta reconstruction is exact rather than approximated.
- Parity in projector mode is tautological after the projector's own commit and only fires when another writer touches the family — mirroring the sticky projector, which also checks parity in both modes.
- The dispatch guard `result.is_delta and handler is not None` keeps absolute polled `MetricsUpdate`s on the legacy write path even if a handler were mis-wired to a polling collector.
- The Balancer fail-closed check generalized to `_unhealthy_router_projector()` iterates sticky then inflight and only acts in projector mode; the legacy hot path keeps a single mode comparison, and the paired benchmark shows no repeatable legacy regression.
- The correct local test environment is the `py311` conda env (`/home/hgq/software/miniconda3/envs/py311`, Python 3.11.14 with `pytest-asyncio`); `agentic-py31114` lacks `pytest-asyncio` and fails async collection in events/Gateway suites on clean HEAD too.
- Pre-existing-failure triage by stashing the working tree and rerunning the identical command on clean HEAD cleanly separated environment defects from Phase 7 regressions before any fix was attempted.

## 2026-09-21 — Phase 8 Rebase Integration

- The new base `092fdb0` replaces the old repeated rollout-config reads with one `_fetch_rollout_config()` call and adds parallel HTTP discovery of vLLM KV-event sources.
- Phase 4's Router mode resolution does not require a second Ray RPC. Passing the already fetched rollout configuration into `_resolve_router_state_mode()` preserves both the new baseline and the ownership-mode contract.
- Phase 2's Direct methods are independent of the rollout-config refactor; the conflict was positional rather than semantic.
- Phase 4's Router health and admission behavior composes with the new base's structured `debug` route log. Production facts still publish only after synchronous commits.
- Phase 7's `_unhealthy_router_projector()` is the final health boundary because it covers both Sticky and Inflight projectors; retaining the older Sticky-only check would regress fail-closed behavior.
- The upstream strategy now requires both sequence and token capacity values. The shared test helper must therefore pass the default `2048` token budget when overriding only `max_num_seqs`.
- Rebase conflict validation must use the existing read-only compatible veRL export because the preserved user submodule lacks `verl.utils.rollout_trace`.
