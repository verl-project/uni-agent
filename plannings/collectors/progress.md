# Collectors Implementation Progress

## 2026-09-20 — Phase 5

### Status

- Phase: Final Migration and Cleanup
- State: complete

### Actions

- Recovered the user-selected `plannings/collectors/` plan and confirmed Phases 1-4 are complete in separate commits.
- Preserved unrelated `verl`, `.planning/`, and `working/` state; Phase 5 starts from `79c7abc`.
- Scoped Phase 5 to Framework prompt aggregation, mixed-version and recovery validation, evidence-gated compatibility cleanup, and final performance/regression checks.
- Added explicit exit criteria requiring empty/failed episode retention, idempotent fragment merging, stable public contracts, and unchanged transport/ownership boundaries.
- Audited Framework finalization and prompt persistence. Confirmed that empty postprocessing currently discards an otherwise available Gateway fragment and that prompt terminal tags contain no episode summary.
- Chose the existing prompt TransferQueue terminal tag as the persistence boundary and a Framework-internal episode outcome as the smallest implementation seam; public Gateway return types remain unchanged.
- Audited Gateway failure finalization: normal finalization preserves empty/failed request metrics, while abort currently discards projector state. Phase 5 will add an additive abort-result seam with a legacy fallback.
- Added bounded prompt metric models and equal-episode aggregation with fragment revision replacement, explicit incompleteness, and pure DTO serialization.
- Added Gateway abort-result collection without changing the legacy `None` return contract, then wired Framework to use the additive method with a legacy fallback.
- Added Framework prompt terminal summaries for enabled Task Metrics modes while preserving the exact disabled-path tag.
- Audited compatibility adapters and retained each externally meaningful path. No adapter met the evidence threshold for removal; the unused projector discard method now has an explicit external-review removal criterion.
- Generalized episode observations to hold multiple source fragments so later Runner/Framework projectors can join the same prompt reducer without changing its public summary shape.
- Completed the final implementation review: prompt summaries retain empty/failed denominators, missing fragments remain explicit, and no Phase 2-4 transport or ownership boundary changed.
- Phase 5 is ready for its own commit on top of the four existing phase commits.

### Verification

| Check | Result |
|---|---|
| Selected plan root | `plannings/collectors/` |
| Phase 5 status | in progress |
| Unrelated worktree changes | present; preserved |
| Prompt aggregation and Gateway abort focused tests | 6 passed; only the Ray deprecation warning |
| Framework empty, failed, and mixed-version summary tests | 3 passed; only the Ray deprecation warning |
| Expanded metrics, Framework, and GatewayManager regression | 86 passed; two recorded baseline failures kept separate |
| Compatible metrics, Framework, and GatewayManager regression after multi-source refactor | 84 passed, 6 exact baseline nodes deselected; only known Ray warnings |
| Cross-phase Direct, Global, local bus, Router, Admission, and metrics recovery regression | 77 passed; only known Ray warnings |
| Exact compatible Phase 0 suite after Phase 5 | 31 passed in 26.45 seconds; external wall time 30.47 seconds versus the pre-change 31.02-second pytest run |
| First enabled prompt reducer benchmark | median 17.072-21.871 us for four episodes and two metrics each; post-Ray P95/P99 drifted upward, so an isolated rerun is required |
| Isolated enabled prompt reducer benchmark | median 19.264-19.434 us, P95 34.785-47.587 us, P99 60.912-87.766 us across three 20k samples |
| Framework prompt summary tests including legacy abort fallback | 4 passed; only the known Ray deprecation warning |
| Full Gateway Actor regression | 67 passed; six known `merge_context_tokens` failures and two known stdout-vs-`caplog` failures |
| Final scoped Ruff, format, compileall, and diff checks | passed |
| Final scope audit | only Phase 5 metrics, Framework, Gateway, tests, and `plannings/collectors/` changed; unrelated dirty state preserved |

### Errors

| Error | Resolution |
|---|---|
| The planning hook reported multiple automatic plans without `PLAN_ID` | Pin this task operationally to the user-selected `plannings/collectors/`; do not recover or modify another plan |
| The first Phase 5 compatibility baseline stopped during collection because the user-modified veRL tree lacks `verl.workers.rollout.replica` | Keep the submodule untouched and rerun with the existing read-only compatible veRL export on `PYTHONPATH` |
| The compatible baseline ran 32 tests and reproduced the known `test_generate_sequences_reports_unfinished_episode_count` stdout-vs-`caplog` failure | Treat this exact node as a baseline defect; 31 other tests passed in 31.02 seconds |
| The first Framework test patch addressed the same file in two separate patch operations and was rejected atomically | Merge the edits into one file operation before retrying; no partial test change was written |
| Initial focused command passed 3 Framework summary tests but Ruff reported import ordering in the two new metrics modules; the shared `-k` expression also deselected metrics/Gateway nodes | Apply scoped import formatting, then run each focused group without a cross-file `-k` filter |
| A later combined command repeated the cross-file `-k` filtering mistake after the multi-fragment refactor | Stop combining filtered Framework selection with other paths; run pure metrics/Gateway tests in a separate invocation |
| Importing TransferQueue helpers directly from the user-modified `verl` tree failed because it also lacks `verl.utils.tensordict_utils` | Preserve the submodule and inspect its replay-buffer source statically; runtime tests continue against the compatible read-only export |
| `check-complete.sh` did not treat `PWF_PLAN_ROOT=plannings/collectors` as a named-plan root | Run the completion check from inside the user-selected legacy plan directory instead of allowing fallback to another plan |
| Expanded metrics/Framework/Manager regression reproduced the second known stdout-vs-`caplog` failure and the known veRL continuation-token HTTP 500 | Keep both exact baseline nodes separate; 86 compatible tests passed |

## 2026-09-20 — Phase 4

### Status

- Phase: Router and Admission Consumers
- State: complete

### Actions

- Recovered the user-selected `plannings/collectors/` plan and confirmed Phase 3 is complete.
- Recorded the dirty worktree and preserved unrelated `verl`, `.planning/`, `working/`, and completed Phase 1-3 changes.
- Started Phase 4 as an ownership-focused migration slice; implementation will follow the formal Router/admission contracts rather than add consumers package by package.
- Froze the formal Phase 4 boundary: preserve Router input protocols and command APIs, migrate one typed input family at a time, and keep admission observe-only.
- Selected sticky bindings as the first Router ownership slice; inflight and KV/metrics inputs remain on their existing writers.
- Selected committed Router route/capacity facts as the admission observation slice; no Coordinator, Grant, Hold, or capacity mirror will be introduced.
- Paused Phase 4 expansion at the user's request to create separate retrospective commits for completed Phases 1-3.
- Audited the current branch, index, untracked files, and overlapping integration diffs; no collector commit or staged change existed.
- Reconstructed each completed phase from `1fad722` in an isolated temporary worktree so shared Gateway/Framework files retained the correct historical boundary.
- Created `3584d2d` (`feat(collectors): add local event metrics slice`) after 23 local tests, 17 focused integration tests, lint, format, compile, and diff checks passed.
- Created `12474f6` (`feat(collectors): add direct router state sync`) after 37 local tests, 9 Direct/config tests, lint, format, compile, and diff checks passed.
- Created `db2f758` (`feat(collectors): add global telemetry pipeline`) after 55 Global/Direct/local tests, 8 config/cleanup tests, lint, format, compile, and diff checks passed.
- Fast-forwarded `collector` to the three commits and refreshed the index without changing worktree files; only Phase 4, planning logs, and explicitly excluded user changes remain uncommitted.
- Removed the temporary worktree and helper branch after the phase commits were attached to `collector`; no temporary project files remain.
- Resumed Phase 4 and audited the draft reducers against Collector callback composition, sticky control commands, Direct snapshot delivery, and bounded admission retention.
- Added focused ownership and failure-isolation coverage for shadow writes, projector commit failure, and admission observer failure.
- Verified the integrated Phase 4 slice together with the local event, metrics, Direct, Global, legacy Router, and callback paths.
- Removed disabled-mode capacity metadata and empty observation calls from the legacy hot path after performance review; focused compatibility tests remained green.
- Completed the final scope review: Phase 4 migrates only Sticky ownership, leaves Inflight/KV/Metrics writers unchanged, and adds no admission enforcement path.
- Completed the final integration, formatting, lint, compile, and diff checks; Phase 4 is ready for its own commit.

### Verification

| Check | Result |
|---|---|
| Selected plan root | `plannings/collectors/` |
| Phase 4 status | complete |
| Unrelated worktree changes | present; preserved |
| Phase 1 commit | `3584d2d` |
| Phase 2 commit | `12474f6` |
| Phase 3 commit | `db2f758` |
| Focused Phase 4 Router/admission tests | 17 passed; only the Ray deprecation warning |
| Final Phase 1-4 integration regression | 158 passed; only known Ray and pytest-mark warnings |
| Focused tests after legacy fast-path cleanup | 61 passed; only the Ray deprecation warning |
| Legacy Router microbenchmark | median 37.285-38.282 us; P99 108.214-116.544 us across three 20k samples, aligned with the Phase 3 parent's noisy 38.041-44.080 us median and 113.032-155.259 us P99 |
| Final Ruff, format, compileall, and diff checks | passed |

### Errors

| Error | Resolution |
|---|---|
| The planning hook reported multiple automatic plans without `PLAN_ID` | Use the user-selected `plannings/collectors/` directory explicitly for this task; do not recover or modify another plan |
| First post-test Router benchmark was distorted by the just-finished Ray workload | Reran isolated legacy batches, compared against the exact Phase 3 parent, then removed disabled-mode work and repeated the focused suite |
| Initial Phase 4 static checks found import order and formatting differences in the new tests | Applied the repository formatter only to those tests and reran the full scoped static checks |

## 2026-09-20 — Phase 3

### Status

- Phase: Global Telemetry
- State: complete

### Actions

- Recovered the user-selected `plannings/collectors/` plan and confirmed Phase 2 is complete.
- Recorded the dirty worktree and preserved unrelated `verl`, `.planning/`, `working/`, and completed Phase 1/2 changes.
- Fixed Phase 3 scope to one default-off Global telemetry vertical slice; Router/Admission ownership migration remains deferred.
- Read the formal Global deployment, lifecycle, batching, pending-call, endpoint, accepted-ACK, loss, and acceptance-test sections.
- Audited existing event, metrics, RL-Insight, and Gateway files; no asynchronous Global Reporter abstraction currently exists.
- Audited LocalEventBus, Gateway runtime construction, and Framework composition; the existing named `deliver()` and driver composition seams can host the Global slice without changing ordinary publish semantics.
- Audited Direct bridge patterns and GatewayManager ownership. Phase 3 will use separate Global DTOs/queues while reusing only the injection and lifecycle seams.
- Re-read Phase 0 Global/config/ownership contracts and identified atomic endpoint ingress as necessary to avoid partial-batch retry ambiguity.
- Chose the Phase 3 vertical slice: existing Gateway lifecycle/generation facts → source forwarder → shared broker → isolated shadow event-count Reporter actor.
- Added the Ray-agnostic Global transport core: versioned DTOs, source batching/retry, bounded broker fan-out, atomic subscriber ingress, named local delivery, Reporter isolation, deduplication, and health/loss counters.
- Added six focused Global core tests covering DTOs, batching, stable-ID BUSY retry, non-blocking overflow, duplicate ingress, slow-subscriber isolation, non-republishing delivery, and Reporter failure isolation.
- Added a driver-owned Global telemetry runtime with separate broker and shadow Reporter Ray actors, then wired the default-off Gateway source path and ordered shutdown through the existing composition seam.
- Added Gateway-to-Ray Global integration coverage and strict configuration checks; the first focused combined run passed all 19 selected tests.
- Reviewed the integrated slice and identified four contained hardening changes: run-aware dedup keys, local subscription startup validation, event-filter config preservation, and failure-safe runtime cleanup.
- Applied the hardening changes and added targeted regression coverage for each; the expanded Global/config/cleanup selection passes 26 tests.
- Replaced Global pump `ray.get` calls with ObjectRef future waits; the dedicated Ray integration now passes without the async-actor blocking warning.
- Extended the Ray vertical slice through one real generation; `SessionOpened`, `GenerationFinished`, and `SessionClosed` all reach the isolated Reporter.
- Added a combined Direct + Global Gateway test proving the two paths advance independently while sharing only the local publisher/bus.
- Made the selected Global event family an explicit composition-owned tuple shared by the source, broker, and Reporter, with strict startup validation.
- Completed the loss-accounting audit so broker and subscriber endpoint BUSY counters include pre-queue validation and size-budget rejections.
- Completed final scope review: no Phase 3 changes were added to Router or Admission ownership, no production RL-Insight series were claimed, and unrelated dirty-worktree content remains untouched.

### Verification

| Check | Result |
|---|---|
| Selected plan root | `plannings/collectors/` |
| Phase 3 status | complete |
| Unrelated worktree changes | present; preserved |
| Focused Global core/config/Ray integration | 19 passed, 63 deselected |
| Expanded Global, runtime-config, and cleanup checks | 26 passed, 67 deselected |
| Event, metrics, and Direct shadow regression | 45 passed |
| Dedicated Gateway-to-Ray Global integration/config | 7 passed; only Ray deprecation/environment warnings |
| Final Global core plus full Gateway event-family Ray slice | 16 passed; only Ray deprecation/environment warnings |
| Gateway Global Ray suite including simultaneous Direct + Global | 8 passed; only Ray deprecation/environment warnings |
| Final event, metrics, and Direct shadow regression | 46 passed |
| Final Global config wiring selection | 7 passed, 63 deselected |
| Final format, Ruff, compileall, and diff checks | passed |
| Final Global core/Ray/config selection after event-family wiring | 27 passed, 60 deselected |
| Final Global core after BUSY accounting audit | 10 passed |
| Final Global, local metrics, Direct shadow, and Gateway Ray acceptance | 55 passed; only Ray deprecation/environment warnings |
| Compatible Gateway Direct and Manager regression | 10 passed, 1 known veRL-dependent test deselected |
| Framework regression | 68 passed, 2 known stdout-vs-caplog tests deselected |
| Full Gateway actor regression | 66 passed; 6 known `merge_context_tokens` dependency failures and 2 known stdout-vs-caplog failures |
| Global source enqueue benchmark with blocked remote target | median 4.156-5.973 us; P99 11.811-27.316 us across three 20k-event batches |

### Errors

| Error | Resolution |
|---|---|
| Initial scoped Ruff check found one import-order error in `uni_agent/events/__init__.py` | Reordered the new Global exports without changing behavior, then rerun scoped checks |
| First Global core lint run found one 123-character test assertion | Split the assertion into named ACK construction and status verification |
| First Global core format check requested formatting for the new transport module | Applied the repository formatter; no manual style-only rewrites |
| Initial telemetry runtime lint found one overlong status assignment | Split the two actor status awaits into self-documenting statements |
| First integrated Phase 3 lint found one Gateway import-order difference | Run the repository import organizer only on the affected Gateway module |
| First combined Gateway Global/Direct run outlived the initial 30-second tool yield and its session handle was not retained | Waited for clean process exit, then rerun the same suite while retaining the session handle and final result |
| Combined Gateway regression reported the already-recorded missing `ContinuousTokenBuilder.merge_context_tokens()` failure | Keep the veRL dependency defect separate; 17 compatible Global/Direct/Manager tests passed in that run |
| Ray warned about `ray.get` from the Global source callback even though it ran in the communication thread | Wait on the ObjectRef's thread-safe future in both Global source and broker subscriber pumps; keep async actor event loops unblocked |
| Full Framework file reproduced the two recorded stdout-vs-caplog failures | Reran with only those exact nodes deselected; all 68 compatible tests pass |
| Full Gateway actor file reproduced six continuation failures from missing `merge_context_tokens()` and two stdout-vs-caplog failures | Classify all eight under the recorded dependency/logging baseline; the remaining 66 tests passed |
| Global enqueue benchmark exposed an `IndexError` when non-draining close cleared a batch still owned by the send pump | Track the active batch under the same condition lock; let the pump retire it before dropping the remaining queue, and add a close-race regression test |
| Extended Ray integration twice observed a later Reporter snapshot paired with earlier broker/subscriber counters | Treat runtime status as two sequential actor snapshots and wait until broker ingress, subscriber acceptance, and Reporter predicates all converge |
| Event-family configuration cleanup left one obsolete Gateway import | Remove the unused constant and rerun scoped lint |
| One resumable final-suite wait call had malformed tool syntax | Reissued the wait for the same session; the suite completed with 55 passing tests |

## 2026-09-20 — Phase 2

### Status

- Phase: Direct State Sync
- State: complete

### Actions

- Recovered `plannings/collectors/` and confirmed Phase 1 is complete.
- Confirmed Phase 2 scope from the formal design: isolated Direct transport, applied ACK/cursor/epoch recovery, bounded queues, snapshot resync, and one shadow control-state consumer.
- Recorded the current dirty worktree and preserved unrelated `verl`, `.planning/`, `working/`, and Phase 1 changes.
- Read the formal Direct deployment, ordering, backpressure, applied-ACK, recovery, and Router migration sections.
- Audited the existing event envelope and local delivery ingress; confirmed `deliver()` can remain the non-forwarding target boundary.
- Audited the Router actor shell, current callback-owned state, Gateway event facts, and actor-local runtime ownership.
- Selected Gateway session lifecycle as the first Direct shadow slice because it exercises real cross-Actor state sync without changing Router decisions or duplicating existing inflight ownership.
- Confirmed the existing driver composition layer can inject the Router Actor handle from `LLMServerClient` into Gateway actors only when the independent Direct feature flag is enabled.
- Added versioned Direct batch/snapshot/ACK DTOs, endpoint validation, a bounded source outbox, retry, applied cursors, epoch isolation, and explicit stale/resync health.
- Added the Router's lazy Gateway-session shadow projector and non-forwarding endpoint methods without changing route selection or existing DataStore writers.
- Wired the default-off Direct flag through Framework composition, GatewayManager, and Gateway actors; enabled actors share their existing local event runtime and install an authoritative snapshot before accepting sessions.
- Corrected first-pass edge cases for snapshot watermark capture, oversized single events, ACK identity validation, terminal snapshot status, and invalid transport DTO handling.
- Applied repository formatting/import organization; scoped lint and compile checks pass for the Phase 2 source files.
- Added focused Direct tests for DTO round trips, contiguous/duplicate delivery, sequence gaps, epoch replacement, subscriber failure, ACK-loss retry, bounded overflow/recovery, and non-blocking publish.
- Added Router shadow tests proving session lifecycle projection does not mutate routing status and malformed schemas fail with `RESYNC_REQUIRED`.
- Added a Ray integration test covering GatewayManager startup snapshot, session open/close delivery, applied cursor advancement, and source health against a remote shadow endpoint.
- Extended configuration tests for Direct default-off behavior, strict boolean validation, and Router handle injection at the existing Framework composition boundary.
- Added source-wide revision propagation, configurable terminal tombstone bounds, and fail-fast Direct handshake ordering before the Gateway HTTP server becomes ready.
- Added a bounded consecutive retry budget; exhaustion marks the Direct view stale so snapshot recovery replaces indefinite BUSY/error retries.
- Fixed the default-path actor construction regression and reran its exact test together with the Direct Ray integration.
- Hardened receiver validation for subscription filters, event schemas, source health, and same-epoch snapshot revision rollback.
- Preserved terminal close reasons in the shadow view and added explicit unsupported event-schema coverage.
- Verified two Gateway actors maintain independent source epochs/cursors while sharing one Router shadow subscription.
- Completed final scope review: no Global transport/telemetry implementation was introduced and unrelated worktree changes remain untouched.

### Verification

| Check | Result |
|---|---|
| Selected plan root | `plannings/collectors/` |
| Phase 2 status | complete |
| Global telemetry scope | explicitly deferred to Phase 3 |
| Direct core and Router shadow tests | covered by the final 37-test local suite |
| Gateway-to-Ray Direct integration | 3 passed using the existing read-only veRL export, including two independent Gateway streams |
| Focused default-off/config compatibility checks | 15 passed |
| Combined Phase 2 focused suite | 30 passed |
| Complete event package plus Router Direct shadow | 28 passed |
| Events, metrics, and focused Router legacy regression | 57 passed |
| Gateway compatible regression after excluding documented baseline failures | 74 passed, 9 deselected |
| Router balancer non-Ray suite | 39 passed; Ray integration collection blocked by missing veRL router module |
| Final Direct validation plus Phase 1 event/metrics regression | 37 passed |
| Direct hot-path enqueue benchmark | three batches, median 3.219-3.280 us and P99 10.853-11.845 us with the target blocked |
| Frozen Router acquire/release rerun | median 28.383-29.804 us; P99 84.677-99.069 us, above the 72.990 us historical baseline despite no Phase 2 hot-path code change |
| Focused Phase 1 integration rerun | 15 passed, 135 deselected |
| Final format, Ruff, compileall, and diff checks | passed |

## 2026-09-20 — Phase 1

### Status

- Phase: Minimal Local Vertical Slice
- State: complete

### Actions

- Recovered the user-selected planning location and confirmed its lifecycle files were missing.
- Read the Phase 0 contract and the separate completed-plan record.
- Recorded the existing uncommitted event, metrics, Gateway, Manager, Framework, and test changes as candidate Phase 1 implementation.
- Started a contract-first audit to avoid duplicate or scattershot changes.
- Audited the event envelope, context binding, publisher, metric DTOs, and Gateway metrics projector.
- Recorded two candidate contract mismatches for validation against the formal design before editing code.
- Audited local bus delivery, queue isolation, lifecycle tests, and the formal Task Metrics integration boundary.
- Confirmed that the fragment field mismatch must be corrected to the frozen `incomplete_reasons` contract.
- Audited Gateway, Manager, and Framework diffs; the vertical path is present but local Task Metrics is always enabled.
- Identified the missing disabled-path configuration gate as a Phase 1 acceptance gap.
- Chose a narrow configuration fix: validate `off|shadow|primary` in `GatewayActorConfig`, build no event runtime for `off`, and wire the frozen nested key through the existing Framework entry point.
- Implemented the three audited contract fixes: default-off runtime gating, canonical `incomplete_reasons`, and non-negative event sequence/timestamp boundaries.
- Updated only the focused configuration, local event, projector, Gateway integration, Manager, and Framework wiring tests.
- Reduced allocation in the event hot path while preserving immutable envelope and DTO behavior.
- Completed final scope review; no Direct or Global transport implementation was introduced.

### Verification

| Check | Result |
|---|---|
| Selected plan root | `plannings/collectors/` |
| Phase 0 artifact | present |
| Phase 1 source/tests | present, uncommitted, and validated |
| Unrelated worktree changes | present; preserved |
| Event and metrics focused tests | 22 passed |
| Scoped Ruff lint | passed |
| Scoped Ruff format check | passed after formatting `uni_agent/gateway/config.py` |
| Focused Phase 1 integration tests | 17 passed; includes default-off, config wiring, Gateway facts/fragments, Manager list compatibility, and Framework post-selection attachment |
| Local/Router collector regression plus Ray DTO | 72 passed |
| Phase 0 compatibility suite | 31 passed, 1 pre-existing logging-capture failure |
| Initial Phase 1 performance run | no-subscriber P99 122.512 us; inline Gateway Projector P99 111.350 us; both above the frozen 25/100 us budgets |
| Five-batch performance rerun | no-subscriber P99 42.466-80.895 us (repeatable failure); inline Projector P99 86.015-116.400 us (mixed) |
| Performance after allocation reduction | five batches passed: no-subscriber P99 20.598-23.452 us; inline Projector P99 61.177-67.870 us |
| Final focused Phase 1 suite | 39 passed |
| Format, Ruff, compileall, and diff checks | passed |

### Errors

| Error | Resolution |
|---|---|
| Initial Phase 2 scoped format/lint found two formatting differences and three import-order errors | Apply the repository formatter/import organizer after correcting the first-pass transport edge cases |
| Gateway Direct integration test could not import `verl.workers.rollout.replica` from the user-modified pinned submodule | Reuse the existing read-only veRL export and Conda runtime library path already recorded during Phase 1; do not mutate `verl` |
| Broad Gateway regression: 10 failures among 83 tests | Seven are the known missing `merge_context_tokens` dependency defect, two are the existing stdout-vs-caplog issue, and one exposed a Phase 2 default-path actor-constructor compatibility regression; preserve the old constructor call when Direct is disabled |
| Full Router balancer suite could not collect `get_router_handle` from the user-modified veRL source | Retry with the same existing read-only veRL export; keep the submodule untouched |
| Read-only veRL export also lacks the newer `verl.workers.rollout.router` module required by Router Ray integration | Run all non-Ray balancer modules and retain the dedicated new Gateway-to-Ray Direct integration as Phase 2 cross-Actor evidence |
| First Phase 2 Router microbenchmark had P99 89.069-93.056 us versus the frozen 72.990 us baseline | Rerun with the baseline's explicit Loguru handler removal before treating the increase as a regression; Direct enqueue itself measured 10.853-11.845 us P99 |
| Resolver did not find lifecycle files under the selected plan root | Created `task_plan.md`, `findings.md`, and `progress.md` in the user-specified directory before continuing |
| `ruff format --check` reported `uni_agent/gateway/config.py` | Apply the repository formatter to that file, then rerun the scoped checks |
| Compatibility test `test_generate_sequences_reports_unfinished_episode_count` saw the expected log on stdout but `caplog.text` was empty | Record as the existing stdout-vs-caplog baseline issue; production output contains the asserted values and no collector code is involved |
| Initial 100k-event performance run exceeded both Phase 1 P99 budgets despite acceptable medians | Rerun multiple isolated batches as required by the baseline; optimize only if the result is repeatable |
| Hot-path optimization left `uni_agent/events/model.py` outside formatter output | Format that file, then rerun lint/tests and the exact same benchmark shape |

## 2026-09-20 — Phase 6

### Status

- Phase: Trainer Metrics Consumption and Export
- State: complete

### Actions

- Recovered the user-selected `plannings/collectors/` plan and confirmed Phases 0-5 are complete.
- Added Phase 6 according to the planning skill's continuation rule for a completed plan.
- Chose the remaining formal-design gap as the Phase 6 vertical slice: consume the Phase 5 prompt summary at the trainer boundary and export it exactly once.
- Confirmed the repository currently writes `agent_metrics_summary` only in Framework and has no downstream production consumer.
- Audited the pinned replay buffer and trainer loop: prompt metadata is visible at sampling time, and auxiliary sampler metrics are already logged by the existing trainer path.
- Selected the supported custom-sampler seam so Phase 6 can remain outside the dirty `verl` submodule.
- Audited the Phase 5 summary DTO and identified the missing trainer-boundary parser and cross-prompt reduction contract.
- Implemented the versioned prompt-summary parser, pure trainer reduction, optional sync/async replay-buffer adapters, and focused DTO/reduction tests.
- Verified 12 focused metrics and adapter-contract tests; scoped lint passes after import organization, and repository formatting was applied only to the two new adapter files and their test.
- Added explicit Framework export ownership: primary claims `trainer`, while shadow never claims or emits production tracking metrics.
- Added stable summary counters, aggregation-conflict diagnostics, public sampler composition, and concise configuration documentation.
- Verified all 13 metrics/replay-adapter tests and 5 focused Framework prompt-metrics tests after splitting the filtered invocations correctly.
- Ran the compatible full metrics/Framework regression: 86 passed, with only the five previously recorded parameterized/log-capture baseline nodes deselected.
- Ran the event, Direct, Global, Router ownership, Admission observation, and metrics cross-phase regression: 73 passed.
- Benchmarked the primary trainer reducer across three isolated 20k-sample batches: median 60.689-72.245 us and P99 167.159-262.754 us for four prompts with two metrics each.
- Audited veRL's second-stage `MetricsAggregator` and found that generic sampler metric names would corrupt prompt counts and `MEAN`/`LAST` semantics across `parameter_sync_step > 1`.
- Tightened the export contract to trainer-safe `sum|min|max` keys; unsupported weighted/last reductions remain explicit rather than approximate.
- Verified the final metrics/Framework compatibility suite: 87 passed, with only five recorded logging-capture parameterizations deselected.
- Verified the final cross-phase event, Direct, Global, Router ownership, Admission observation, and metrics suite: 74 passed.
- Re-ran the production-shaped reducer benchmark after trainer-safe key validation: median 71.766-75.029 us and P99 208.050-250.928 us across three isolated 20k-sample batches.
- Completed final static and scope review; Phase 6 is ready for its own commit.

### Verification

| Check | Result |
|---|---|
| Selected plan root | `plannings/collectors/` |
| Prompt DTO, trainer reduction, and replay adapter tests | 17 passed |
| Focused Framework prompt-summary/owner tests | 5 passed; only the known Ray deprecation warning |
| Compatible metrics and Framework regression | 87 passed, 5 recorded logging-capture parameterizations deselected |
| Cross-phase event, Direct, Global, Router, Admission, and metrics regression | 74 passed; only the known Ray warning |
| Optional replay-buffer dependency | real TransferQueue unavailable locally; adapter contract passed against the documented fake upstream seam |
| Production-shaped trainer reducer benchmark | median 71.766-75.029 us; P99 208.050-250.928 us across three 20k-sample batches |
| Ruff, format, compileall, Markdown fence, and diff checks | passed |
| Unrelated worktree state | `verl`, `.planning/`, and `working/` preserved |

### Errors

| Error | Resolution |
|---|---|
| The first Phase 6 tail command ran from the repository root and used unqualified planning filenames | Reissued the read with explicit `plannings/collectors/` paths; no files were changed by the failed read |
| Initial scoped lint found one import-order difference in `uni_agent/metrics/prompt.py` | Apply the repository import organizer only to the Phase 6 files and rerun the scoped checks |
| Scoped format check reported the two new trainer adapter files | Format only those Phase 6 files and rerun the check |
| Optional replay-buffer adapter import could not find `transfer_queue` in the current CPU environment | Do not add a mandatory dependency; exercise the adapter against a fake upstream contract and record real TransferQueue runtime coverage as unavailable locally |
| The new replay-buffer adapter contract test needed repository formatting | Format that test only and rerun the focused suite and static checks |
| Framework prompt-metrics collection hit the known missing `verl.workers.rollout.replica` module | Rerun against the existing compatible read-only veRL export while preserving the dirty submodule |
| The owner-marker change left `uni_agent/metrics/trainer.py` outside formatter output | Format that Phase 6 file and rerun scoped checks |
| A shared `-k prompt_metrics` filter deselected trainer-export and adapter tests in a combined invocation | Split metrics modules and Framework selection into separate pytest commands so every intended test runs |
| The first trainer-safe metric-key patch missed a formatter-adjusted context block | Re-read only the reducer and focused test sections; the failed patch made no partial changes |
| Ruff was accidentally pointed at `gateway-and-trajectories.md`, producing irrelevant Python parse errors | Rerun Ruff only on Phase 6 Python paths and validate the documentation separately |
| The trainer-safe reducer rewrite needed repository formatting | Format only `uni_agent/metrics/trainer.py` and rerun the focused checks |

## 2026-09-20 — Phase 7

### Status

- Phase: Router Inflight Ownership Migration
- State: complete

### Actions

- Recovered the user-selected `plannings/collectors/` plan; `check-complete.sh` from inside the legacy plan root confirmed all seven phases (0-6) complete before continuation.
- Selected the Phase 7 slice from the formal design's remaining gaps: admission enforce is gated behind the separate capacity-admission threshold (design 8.5), so the next design-faithful slice is completing Router migration for the inflight input family.
- Audited the inflight write path: Balancer callbacks → `CallbackTransport` → `InflightParser` delta `MetricsUpdate` → `Collector._write_metrics_update` (per-request turn/prompt-length folding, batched `incr_metrics`, insight WriteEvents, throttled dispatch logs).
- Confirmed the store backing is process-wide singletons, so the Balancer-owned `DataStore` and the Collector-owned `DataStore` observe the same state — the same property the sticky parity check relies on.
- Confirmed the Balancer's `_inflight` command-side ledger is direct command-path state, not a callback input; it stays untouched as the capacity-fact source.
- Froze the Phase 7 exit criteria: single-writer cutover for the inflight delta family, identical commit semantics regardless of owner, fail-closed projector health, and unchanged Sticky/KV/polled/Direct/Global/admission ownership.
- Extracted the delta-commit core from `Collector._write_metrics_update` into the shared `commit_inflight_delta()` and the throttled `router-dispatch` line into `log_dispatch_stats()`, so the legacy Collector writer and the projector produce identical store state, insight WriteEvents, and log cadence.
- Promoted the per-request row keys to shared `TURN_ROW_KEY`/`PROMPT_LEN_ROW_KEY` constants so the projector's read-only fold cannot drift from the commit fold.
- Added `RouterInflightStateProjector`: projector mode commits through the shared function and owns the family's dispatch-log throttle; shadow mode reconstructs the effective deltas read-only from post-commit rows and compares a per-replica, fixed-key ledger against the store for parity in both modes.
- Wired `inflight_update_handler`/`inflight_update_observer` through `Collector` and `get_collector("inflight_stat")` with an `is_delta` guard that keeps absolute polled updates on the legacy write path; the Collector's own delta write is bypassed exactly when the handler is present, matching the sticky precedent.
- Extended `_unhealthy_router_projector()` so projector-mode failures of either the sticky or the inflight projector block route expansion before routing and after the acquire commit; `get_router_state_status()` exposes the inflight projector in every mode.
- Added nine focused tests: legacy writer behavior, shadow parity and drift detection, shadow write prohibition, projector single-writer with folded deltas, commit-failure fail-closed, non-delta rejection, and Collector-level dispatch routing for delta and absolute updates.
- Updated the Phase 4 legacy-runtime assertion for the added `inflight` status key; legacy mode still allocates no bus, publisher, projector, or observer.

### Verification

| Check | Result |
|---|---|
| Selected plan root | `plannings/collectors/` |
| Phase 0-6 status before continuation | complete (7/7 per `check-complete.sh`) |
| Unrelated worktree changes | `verl`, `.planning/`, `working/`; preserved |
| Focused Phase 7 tests after formatting | 9 passed |
| Phase 7 + router-state-modes + inflight + sticky suites | 22 passed; only the Ray deprecation warning |
| Full router area `-m "cpu and level0"` (ray integration file ignored per recorded blocker) | 245 passed, 2 skipped, 11 deselected |
| Cross-phase events, metrics, admission, Direct, Global suites | 73 passed |
| Ruff check and format on changed files | passed after formatting `router_state.py` and the new test |
| `compileall` over `agent_aware_router` | passed |
| Paired legacy acquire/release benchmark (same host, 20k samples ×3 batches) | Phase 7 median 61.775-62.350 us versus clean parent 67.270-69.221 us; P99 197.477-213.418 us versus parent 175.216-191.348 us — median improved, tail inside this host's cross-run variance; absolute values remain far above the Phase 0 baseline host, consistent with the recorded Phase 2/4 variance findings |
| Final scope audit | only Router inflight ownership files, focused tests, and `plannings/collectors/` changed; unrelated dirty state preserved |

### Errors

| Error | Resolution |
|---|---|
| `check-complete.sh` with `PWF_PLAN_ROOT` pinned to the custom plan root refused resolution | Reproduced the recorded behavior; run the check from inside the legacy plan directory without a pin, as previously established |
| An early grep ran from `plannings/collectors` and warned the package path did not exist | Reissued from the repository root with absolute paths; no state was affected |
| The first test run used the `agentic-py31114` conda env, which lacks `pytest-asyncio`, producing seven collection/async failures across events and Gateway Direct/Global suites | Verified the same seven failures reproduce identically on stashed clean HEAD, then switched to the `py311` env (Python 3.11.14 with `pytest-asyncio`); all 73 cross-phase tests pass there with Phase 7 applied |
| `test_shadow_mode_reports_inflight_parity` asserted `INFLIGHT_TOKENS == 5` from a one-element prompt list (`prompt_len` is `len(prompt_ids)`) | Corrected the test data to a five-element prompt; production code was right |
| The inflight projector's `_validate` ran outside the try block, so a rejected non-delta update raised without marking the projector unhealthy | Moved validation inside the try in both `apply` and `observe`, matching the sticky projector's fail-closed semantics |
| Ruff reported one overlong line in the new projector code | Applied the repository formatter to the two affected files and reran lint, format, and the focused suites |
| A sed-based row-key rename invalidated a pending in-context edit, which was then rejected atomically | Re-read the file and re-applied the same design against the current text |
| Router ray-integration collection still fails on the export's missing `verl.workers.rollout.router` | Recorded Phase 2 baseline blocker; the dedicated Gateway-to-Ray Direct/Global integrations remain the cross-Actor evidence |

## 2026-09-21 — Phase 8

### Status

- Phase: Rebase Integration
- State: complete

### Actions

- Recovered the selected plan from `ORIG_HEAD` while the early rebase commits did not yet contain `plannings/collectors/`.
- Confirmed the rebase target is `092fdb0` and the seven collector commits retain their original order.
- Resolved the Phase 2 conflict by keeping the new `_fetch_rollout_config()` baseline together with the Direct Endpoint, snapshot, and ACK methods.
- Resolved the Phase 4 conflicts by fetching rollout configuration once and reusing it for Router overrides, capacity, and `router_state_mode` selection.
- Preserved the new base's parallel vLLM KV-event endpoint discovery and structured Router logging.
- Updated the Router test helper for the two-argument `set_capacity(max_num_seqs, max_num_batched_tokens)` contract.
- Resolved the Phase 7 conflict by retaining the generalized Sticky/Inflight fail-closed health check.
- Completed the interactive rebase; Phases 3, 5, and 6 replayed without conflicts.
- Confirmed the seven phase commits remain ordered and based directly on `092fdb0`.
- Completed the final worktree audit; only the three Phase 8 planning files plus preserved unrelated `verl`, `.planning/`, and `working/` state remained before the Phase 8 commit.

### Verification

| Check | Result |
|---|---|
| Phase 2 Direct shadow conflict check | 3 passed |
| Phase 4 Router/Direct/new-baseline focused suite | 46 passed with the read-only compatible veRL export |
| Phase 7 Router/Direct/new-baseline focused suite | 55 passed with the read-only compatible veRL export |
| Full Router CPU/Level0 suite | 238 passed, 11 deselected; Ray integration file excluded per the recorded dependency blocker |
| Cross-phase Events, Metrics, Admission, Direct, and Global suite | 73 passed |
| Compatible Framework suite | 70 passed, 5 recorded logging-capture nodes deselected |
| Conflict-file Ruff and compile checks | passed |
| All 51 changed Python files | Ruff check and format check passed; `compileall` passed |
| History and worktree checks | seven ordered collector commits over `092fdb0`; no conflict markers or rebase metadata; diff checks passed |
| Interactive rebase | complete; branch restored to `collector` |

### Errors

| Error | Resolution |
|---|---|
| Selected planning files were not present at the Phase 2 stop | Read them from `ORIG_HEAD` and delayed plan updates until their introducing commit replayed |
| First combined Phase 4 patch did not match one conflict block | The patch made no changes; split the resolution into bounded patches against the current text |
| Direct Phase 4 test invocation failed during collection on missing `verl.utils.rollout_trace` | Re-ran unchanged tests with `/tmp/uni-agent-verl-phase0.UgGIt1` on `PYTHONPATH`; 46 passed |
| Initial parallel final-test orchestration returned before two nested commands finished and did not expose their session ids | Waited for both processes to finish, then reran the commands with resumable session capture; 73 and 70 tests passed respectively |
