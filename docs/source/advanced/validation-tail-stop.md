# Stop validation tails and publish scores

A validation run can stop only its unfinished sessions while preserving scores that have already completed.
Unfinished observations contribute zero to the original sample denominator. Completed trajectories still enter
TransferQueue; failures do not create synthetic training trajectories. The trainer commits validation metrics
before proceeding to its next training step.

## Dependencies and usage

The complete workflow requires the companion [V1 trainer changes in verl #8468](https://github.com/verl-project/verl/pull/8468):
validation lifecycle hooks, empty-batch handling, the original score denominator, and remote generation cancellation.
This change preserves the existing verl submodule pointer. The PR remains a draft until the companion changes
are merged and the dependency is updated. Applying only the uni-agent changes does not provide the complete
stop-and-score protocol.

Before starting training, configure a framework actor name that is unique to this job:

```yaml
actor_rollout_ref:
  rollout:
    custom:
      agent_framework:
        control_actor_name: validation-control
```

The training log identifies each run with `Validation started: validation_id=... global_steps=...`.
Run the following command from an environment that can access the same Ray cluster and namespace:

```bash
python -m uni_agent.framework.control \
  --address "$RAY_ADDRESS" \
  --namespace "$RAY_NAMESPACE" \
  --actor validation-control \
  --validation-id "$VALIDATION_ID" \
  --timeout 30
```

Use the address, namespace, and validation ID of the active training job. The CLI returns committed metrics as
JSON. `phase=finished` and `pending_sessions=0` indicate completed settlement. Repeating a stop for the same run
is idempotent; stopping an old run does not affect the next one. A CLI wait timeout does not revoke the stop request.

When the framework has a `log_dir`, results are written atomically to `validation/<validation_id>.json`.
If backend generation cleanup fails, the computed scores are saved with phase `unsafe`, and the trainer raises
instead of resuming training.

## Sandbox command lifecycle

OpenYuanRong command execution requires the validated `openyuanrong-sandbox==0.10.2rc9` SDK.
After external sandbox deletion, a synchronous command request can remain blocked until a route timeout.
Cancelling its Python await does not terminate the blocked thread. SDK command calls therefore run in a separate
process while the parent task retains ownership of the sandbox and tunnel and periodically checks instance state.
Confirmed deletion or termination stops and reaps the command process and propagates `SandboxTerminatedError`.
Ordinary nonzero exit codes, stdout, stderr, command environment, and working directory retain their semantics.

Cancellation and command timeout first request termination of that remote command. The parent task still owns
sandbox context cleanup. A health-query transport error alone does not establish that the sandbox has died.
The default `command_wait_interval=2.0` seconds can be changed through sandbox kwargs; it is not a response-time
guarantee when network RPCs themselves stall.

## Validation results and limits

The real experiments used Qwen3.5-9B, mini-swe-agent, five independent SWE-bench tasks, real SWE grading, and the
native V1 PPO workflow. The validation step limit was not shortened. Three tasks submitted naturally and were
graded before the two remaining tasks were stopped.

| Scenario | Observation |
|---|---|
| Original external deletion | Metrics appeared 135.310 seconds after both instances were confirmed deleted; acc was 0.4 but reward/score were 0.667. |
| Fixed automatic three-completed/two-deleted run | Metrics appeared 2.740 seconds after confirmed deletion; acc/reward/score were all 0.4. |
| Actual public stop call | Metrics of 0.6 were recorded 1.444 seconds after the framework received stop; the two tasks exited after approximately 0.45/0.50 seconds and their instances were confirmed absent after 0.56/0.52 seconds. |

Both fixed runs recorded validation metrics before completing the next real training update. Each run's 17
sandboxes and job resources were cleaned up. The original run did not actually skip metric publication.
The public-stop experiment's automatic injector initially failed before calling the CLI because an address
environment variable was missing. An explicit retry was made after confirming that the same batch still had
three completed and two active tasks. The first trigger to metric publication took 259.172 seconds; the actual
CLI start to publication took 5.986 seconds and the complete CLI round trip took 7.284 seconds. This run does
not establish a successful first automatic injection.

These GPU experiments used uni-agent `6b0015a3` and the verl archive identified as `2bc6ac46`, plus the candidate
fixes. Targeted checks were rerun after preparing the PR on newer main revisions; the earlier GPU results are
not presented as a full GPU rerun of the new commits. The experiment environment did not contain the separate
rl-insight package: trace-finish calls were checked, but an external telemetry UI was not validated.

In an isolated deletion race, the previous poll implementation waited 60.116 seconds, while the command-process
implementation took approximately 2.043 seconds and reaped its helper. Median short-command latency increased
from approximately 0.166 to 0.451 seconds; the helper's peak RSS in the lightweight probe was approximately
47 MiB. The cost of 500 concurrent real sandboxes was not measured. Earlier controlled tests with 500 logical
samples are not evidence of real-model or PPO validation.

The Gateway cleanup changes overlap with [existing PR #256](https://github.com/verl-project/uni-agent/pull/256),
which additionally handles repeated cancellation and preparation cleanup. Reconcile the overlapping changes
after that PR is merged.
