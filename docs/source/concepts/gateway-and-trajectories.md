# Gateway and Trajectories

The Uni-Agent Gateway connects Agent runtimes to a verl-managed rollout engine. It exposes familiar model APIs to Agents while preserving the token IDs, masks, log probabilities, and rewards required by training.

The Gateway is not an inference engine. vLLM or SGLang generates tokens; the Gateway owns session routing, protocol conversion, and trajectory materialization.

## Two Inference Paths

External API inference bypasses the Gateway:

```text
Task -> Agent -> external model API
```

It is useful for evaluation, but it does not produce Gateway token-level training trajectories.

verl-managed inference and training use the full path:

```text
verl LLMServerManager
    -> AgentFrameworkRolloutAdapter
    -> Uni-Agent Gateway session
    -> Task Runner
    -> Task / Agent / Sandbox
    -> TransferQueue
```

Use this path when inference must match training or when trajectories are needed as training data.

## Session Lifecycle

For each rollout session, the Agent Framework:

1. Chooses a Gateway actor and creates a session.
2. Receives a session-scoped model `base_url` and reward endpoint.
3. Launches the configured Agent Runner, such as `run_task`.
4. The runner injects the session endpoint into `agent.model`.
5. The Agent sends OpenAI Chat Completions or Anthropic Messages requests to the session URL.
6. The Gateway forwards tokenized requests to the verl rollout engine.
7. The Task returns a `TaskResult` with `reward` / `accuracy` / `finished` / `metrics`.
8. The framework finalizes the session, scores from `TaskResult`, and writes trajectories to TransferQueue.

The model-facing endpoints are:

```text
POST /sessions/{session_id}/v1/chat/completions
POST /sessions/{session_id}/v1/messages
POST /sessions/{session_id}/reward_info
```

Sessions are held in Gateway memory until they are finalized or aborted.

## Token-Level Trajectories

A finalized trajectory contains:

- `prompt_ids`: encoded initial prompt.
- `response_ids`: generated tokens plus inter-turn continuation tokens.
- `response_mask`: `1` for model-generated tokens and `0` for Tool results or other continuation context.
- Optional rollout log probabilities.
- Session metadata, reward information, and materialization reason.

Before writing to TransferQueue, the Agent Framework derives the full training record, including `input_ids`, attention masks, position IDs, loss masks, and sparse `rm_scores`.

An optional trajectory postprocessor can filter, reorder, or replace the
finalized trajectories before reward scoring and artifact logging. The order is:

```text
Gateway finalization
    -> Framework copies TaskResult reward/acc/finished onto every trajectory as reward_info
    -> Runner trajectory_selection (all or longest)
    -> trajectory postprocessor
    -> score from TaskResult; merge TaskResult.metrics onto the last trajectory
    -> trajectory logs and TransferQueue
```

Configure a postprocessor at the Framework level:

```yaml
actor_rollout_ref:
  rollout:
    custom:
      agent_framework:
        trajectory_postprocessor_fqn: my_recipe.trajectory.process_trajectories
        trajectory_postprocessor_kwargs:
          max_total_tokens: 262144  # 256K prompt + response tokens
```

The FQN is imported when the Agent Framework starts, so it must be available in
the AgentFrameworkWorker environment. A configured FQN must be a non-empty
string, and `trajectory_postprocessor_kwargs` must be a mapping. Invalid
explicit configuration, import failures, and non-callable targets fail during
Framework initialization. The callable must follow this contract:

```python
from uni_agent.gateway.session import Trajectory


def process_trajectories(
    trajectories: tuple[Trajectory, ...],
    *,
    max_total_tokens: int = 262_144,
) -> list[Trajectory]:
    return [
        trajectory
        for trajectory in trajectories
        if len(trajectory.prompt_ids) + len(trajectory.response_ids) <= max_total_tokens
    ]
```

`trajectory_postprocessor_kwargs` is passed as keyword arguments. The processor
receives a tuple and must return `list[Trajectory]`; returning an empty list
filters the session out. An `async def` processor is also supported and is
awaited. Returned trajectories must preserve the finalized session
`reward_info` and keep their token arrays aligned because reward scoring,
unfinished masking, and TransferQueue materialization run later. Use
`dataclasses.replace` when constructing transformed trajectories so unrelated
fields remain intact.

`max_total_tokens` above is defined by the example processor.
This compact example drops oversized trajectories; a processor that
crops them must preserve token-array alignment and valid turn boundaries. A
processor may define different keyword arguments or none at all.

The hook is disabled when `trajectory_postprocessor_fqn` is omitted or `null`.
In that case no extension is imported or called, and finalized trajectories
continue through the original scoring, logging, and TransferQueue path.

The Gateway uses a `MessageCodec` to:

- Apply the model chat template.
- Incrementally encode Tool observations between turns.
- Decode model tokens into text and Tool calls.
- Handle OpenAI and Anthropic wire formats.
- Extract multimodal inputs when a processor is configured.

The configured Tool parser must match the model's chat template.

## Multiple Turns and Chains

One session may contain multiple model turns. Tool observations are encoded as continuation tokens and marked with `response_mask=0`, while model completions are marked with `response_mask=1`.

Concurrent requests may create multiple chains within one session. Chains sharing a message prefix reuse the same encoded context where possible, then materialize as separate trajectories during finalization.

When a client rewrites only the most recent Assistant message, the Gateway rolls
the matching chain back to the start of that Assistant turn and re-encodes the
replacement suffix. This preserves token, mask, and rollout-log-probability
alignment without materializing a redundant trajectory. The behavior is enabled
by default and can be disabled with
`actor_rollout_ref.rollout.custom.agent_framework.enable_last_assistant_rollback=false`.

## Reward Flow

The built-in Task Runner returns a `TaskResult`. Reward does not enter the
Gateway. After finalization the Agent Framework copies `reward` / `acc` /
`finished` onto every trajectory as session `reward_info` (dump /
`mask_unfinished_episode` metadata only):

```json
{
  "reward": 1.0,
  "acc": 1.0,
  "finished": true
}
```

Scoring and metrics stay on `TaskResult`. After `trajectory_selection` /
`trajectory_postprocessor`, the Framework writes `TaskResult.reward` onto a
sparse token-level `rm_scores` tensor and merges `TaskResult.metrics` onto the
last trajectory's `agent_metrics`.

Not returning a `TaskResult` is the same as a Gateway session that never
received a reward. Session `reward_info` is scored if the Gateway has one. A
configured `RewardLoopWorker` is the custom reward path for sessions with no
Task-provided reward (judge / RewardManager), not a last-resort fallback.
Without a session reward and without a RewardLoopWorker score, `rm_scores`
remains zero and the framework emits a warning.

Agent completion is factual session metadata; the Framework, not the Task, decides how training consumes it. When the training configuration enables `mask_unfinished_episode`, a session with `finished=false` is still written and tagged as successful, but its TransferQueue `response_mask` and `loss_mask` are all zero so it does not contribute policy gradients, loss-normalization counts, or auxiliary losses.

Masking stops at the loss. The trajectory keeps its reward in `rm_scores`, so a group-relative estimator such as GRPO or RLOO still folds that reward into the group mean and standard deviation, shifting the advantages of the sibling rollouts sharing its `uid`. The masked trajectory itself gets a zero advantage, and it still costs a full forward and backward pass. Treat unfinished episodes as evidence that keeps the baseline honest, not as samples removed from the batch.

## TransferQueue

TransferQueue decouples asynchronous rollout generation from the trainer.

Prompt-level records use:

```text
{uid}
```

Trajectory records use:

```text
{uid}_{session_index}_{trajectory_index}
```

A prompt begins as `pending` and ends as:

- `finished` when at least one session succeeds.
- `failure` when all sessions fail.

Individual trajectory records contain token tensors, masks, log probabilities, rewards, Task metadata, Agent name, session ID, and rollout status.

The trainer's ReplayBuffer consumes completed records independently of rollout timing.

## Failure Handling

- A Runner exception aborts its Gateway session.
- One failed session does not discard successful sibling sessions.
- A prompt is marked `finished` when any of its sessions succeeds.
- A prompt is marked `failure` when all sessions fail.
- A batch raises only when every rollout fails.

This isolation is important for long-horizon workloads, where session latency and failure modes vary widely.

## Configuration

The main Agent Framework settings live under:

```text
actor_rollout_ref.rollout.custom.agent_framework
```

Important knobs include:

- `gateway_count`: Gateway actor pool size.
- `mask_unfinished_episode`: zeroes training masks for sessions that report
  `finished=false`. Sessions without completion metadata remain trainable.
  Defaults to `false`.
- `enable_last_assistant_rollback`: reuses a chain when only its latest Assistant
  message is rewritten. Defaults to `true`; set it to `false` to preserve the
  previous split-on-rewrite behavior.
- `trajectory_postprocessor_fqn`: optional import path for a sync or async
  callable that postprocesses finalized trajectories before reward scoring.
- `trajectory_postprocessor_kwargs`: optional keyword arguments passed to the
  postprocessor.
- `agent_runners`: Runner import paths and arguments.
- `dispatch_mode`: inline async execution or Ray tasks.
- `max_concurrent_sessions`: per-Runner concurrency limit.
- `log_dir`: runtime log root. Sessions with a global step write `framework.log`, `task.log`, and trajectory artifacts under `step_<global_step>/<log_id>/`; Sessions whose `global_steps` is `None` write directly under `<log_id>/`.
- `rollout.n`: sessions per prompt.
- `rollout.multi_turn.format`: model-specific Tool parser.
- `transfer_queue.enable`: enables asynchronous trajectory storage.

## Extension Boundaries

Customize the layer that owns the behavior:

- Implement an Agent Runner to launch a different workload against a Gateway session.
- Add a Gateway adapter for a new model API wire format.
- Customize Task, Agent, Tool, and Sandbox behavior through their registries.
- Implement a trajectory postprocessor for use-case-specific filtering or
  transformation after finalization and before scoring.
- Customize reward scoring in the Task or a verl Reward Loop Worker.

Do not put Task logic inside Gateway routes or bypass the Gateway token buffers when training-format trajectories are required.
