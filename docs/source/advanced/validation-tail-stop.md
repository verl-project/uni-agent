# Stop validation tails with unmodified verl

Uni-agent owns validation settlement and request cancellation through its V1 integration.
The verl source and submodule pointer remain unchanged. This workflow does not require
verl PR #8468 or runtime replacement of verl classes or functions.

## Training entry point

Use the uni-agent entry point with the normal verl configuration and overrides:

```bash
python -m uni_agent.integrations.verl_v1.main \
  --config-name=ppo_megatron_trainer \
  trainer.use_v1=True \
  trainer.v1.trainer_mode=separate_async \
  ...
```

The Mini-SWE-Agent training launcher uses this entry point. Other recipes can opt in by changing
only the launcher module while retaining their existing configuration. The native verl entry
continues to use verl's original validation behavior.

The integration passes its own TaskRunner to `verl.trainer.main_ppo.run_ppo`. Its trainer subclasses
reuse native initialization, `fit`, optimizer updates, weight synchronization, checkpointing, and
mode-transition hooks. Uni-agent owns the validation collector and selects a cancellable client
through `LLMServerManager.get_client(client_cls=...)`. No default trainer registry entries are replaced.

The collector follows the V1 contract at the pinned verl revision `2bc6ac46`; upgrades require
compatibility checks. Wrappers exist for sync, colocate_async, and separate_async. The real GPU
experiment for this change uses separate_async; the other modes are not claimed GPU-validated.

## Stop one validation run

Before starting training, configure a framework actor name unique to the job:

```yaml
actor_rollout_ref:
  rollout:
    custom:
      agent_framework:
        control_actor_name: validation-control
```

The log identifies each invocation with `Validation started: validation_id=... global_steps=...`.
Use that ID and the active job's Ray address and namespace:

```bash
python -m uni_agent.framework.control \
  --address "$RAY_ADDRESS" \
  --namespace "$RAY_NAMESPACE" \
  --actor validation-control \
  --validation-id "$VALIDATION_ID" \
  --timeout 30
```

Stop affects only unfinished sessions belonging to that invocation, including work waiting for
admission and later batches. Completed trajectories remain in TransferQueue. Repeating a stop
is idempotent; stopping an old invocation does not affect the next one. A CLI wait timeout does
not revoke the stop request.

The CLI waits for the trainer's committed metrics. `phase=finished` and `pending_sessions=0`
indicate settlement. With framework `log_dir`, the result is written atomically to
`validation/<validation_id>.json`. Generation cleanup that cannot be confirmed produces an
`unsafe` result and prevents training from resuming.

## Scoring policy

This entry explicitly treats unfinished validation observations as zero for canonical reward,
accuracy, and scorer score. It does not change the policy used by other verl entry points.
The original expected count is retained even when a batch has no trajectory records.
Padding happens only in metric inputs: no synthetic training trajectories, token IDs, or
log-probabilities are written to TransferQueue.

Observed-only statistics retain the `*_success_only` names. They include completed zero-score
results; they do not mean only solved tasks. Sources that never define accuracy or scorer score
do not acquire synthetic versions of those metrics. Other metrics and native bootstrap/majority
calculations are reused, without inventing predictions for missing observations.

The result also reports `val-control/expected_sessions`, `scored_sessions`, `failed_sessions`, and
`scored_fraction`. A scored session is a materialized final result, not necessarily a solved task
or a natural agent submission. These counts make a zero score from missing results visible.

## Request and sandbox ownership

The integration client owns each dispatched Ray ObjectRef. Cancelling the local call requests
remote cancellation and waits for acknowledgement. Unconfirmed cancellation raises an error and
retains the routing reservation. The default confirmation budget is five seconds, configurable
as `actor_rollout_ref.rollout.custom.agent_framework.cancellation_timeout_seconds`.
Late routing allocations are also released when acquisition is cancelled. Cleanup is protected
against repeated cancellation of the caller. Full-async response budgets and partial resumption
continue to use verl's implementation.

OpenYuanRong command execution requires `openyuanrong-sandbox==0.10.2rc9`. Blocking SDK command
waits run in a separate process; the parent retains the sandbox and tunnel. Confirmed instance
termination reaps the command process and raises `SandboxTerminatedError`. Normal nonzero exits,
stdout/stderr, env/cwd, and the image's PATH are preserved. A health-query transport failure alone
does not establish sandbox death. The two-second default `command_wait_interval` is not a latency
guarantee when control-plane RPCs themselves stall.

## Validation scope

The unmodified-dependency acceptance run passed automatically on the first stop attempt:
scikit-learn-14141, django-15741, and astropy-12907 submitted naturally and each scored 1.0;
django-11179 and astropy-13033 were then cancelled. The expected/scored/failed counts were
5/3/2, and canonical reward, accuracy, and score were all 0.6.

Metrics were recorded 0.955 seconds after the framework received stop, or 5.540 seconds after
the automatic trigger. The complete CLI round trip took 6.830 seconds. Both target tasks,
Gateway sessions, model requests, and sandbox instances were terminal before metric publication.
The next native training update then completed. The job exited successfully and all 17 recorded
sandboxes, job processes, and GPU allocations were cleaned up.

The dependency remained a clean checkout of `2bc6ac46043e31855cea194236ff7548c91817c8`:
source hashes matched before and after, and native function identities were unchanged.
Read-only tracing observed native `fit()` invoking the uni-agent validation implementation;
no verl methods were replaced by the experiment. There was no explicit control retry.

The exact production snapshot also passed 282 integration/framework/Gateway/sandbox regressions,
full pre-commit, a strict documentation build, and the required live sandbox service check.

The acceptance workload uses Qwen3.5-9B, mini-swe-agent, five real SWE-bench tasks, actual SWE
grading, and a native training update before and after validation. The controller waits for three
natural submissions with published scores before stopping the two remaining tasks; it does not
shorten the validation step limit. Acceptance must verify score publication before the next
training update, target task/session/request termination, sandbox cleanup, and unchanged verl
source hashes and method identities.

The integration regressions cover partial and total failure, multiple validation batches, repeated
validation at one training step, observed-only metrics, prediction alignment, native partial
resumption, late routing release, unconfirmed cancellation, and isolation between real Ray requests.

Earlier cross-repository experiments are historical evidence, not acceptance of this integration:
confirmed-deletion-to-score latency improved from 135.310 to 2.740 seconds, and a public stop recorded
metrics 1.444 seconds after receipt. The latter required an explicit retry after an injector address
lookup error, so it was not a successful first automatic injection. Those runs used a patched verl;
they cannot establish compatibility with an unmodified dependency.

The separate command-worker probe measured roughly 47 MiB peak helper RSS and a short-command
median increase from 0.166 to 0.451 seconds. Five real tasks do not establish 500-sandbox capacity.
The external rl-insight UI is not covered when its optional package is absent.

Gateway cleanup overlaps with [PR #256](https://github.com/verl-project/uni-agent/pull/256).
That overlap remains a separate review concern; it does not require a verl fork.
