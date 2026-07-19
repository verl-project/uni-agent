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
7. The Task posts its final reward to the session.
8. The framework finalizes the session and writes trajectories to TransferQueue.

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

## Reward Flow

The built-in Task Runner posts:

```json
{
  "reward_info": {
    "reward": 1.0,
    "acc": 1.0
  }
}
```

The Agent Framework reads the session reward, applies it to finalized trajectories, and writes a sparse token-level `rm_scores` tensor with the reward on the final token.

If no Task reward is reported, an optional verl Reward Loop Worker can score the final trajectory. Without either source, `rm_scores` remains zero and the framework emits a warning.

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
- Reward reporting is best-effort and logs failures.

This isolation is important for long-horizon workloads, where session latency and failure modes vary widely.

## Configuration

The main Agent Framework settings live under:

```text
actor_rollout_ref.rollout.custom.agent_framework
```

Important knobs include:

- `gateway_count`: Gateway actor pool size.
- `agent_runners`: Runner import paths and arguments.
- `dispatch_mode`: inline async execution or Ray tasks.
- `max_concurrent_sessions`: per-Runner concurrency limit.
- `log_dir`: optional per-session trajectory and execution logs.
- `rollout.n`: sessions per prompt.
- `rollout.multi_turn.format`: model-specific Tool parser.
- `transfer_queue.enable`: enables asynchronous trajectory storage.

## Extension Boundaries

Customize the layer that owns the behavior:

- Implement an Agent Runner to launch a different workload against a Gateway session.
- Add a Gateway adapter for a new model API wire format.
- Customize Task, Agent, Tool, and Sandbox behavior through their registries.
- Customize reward scoring in the Task or a verl Reward Loop Worker.

Do not put Task logic inside Gateway routes or bypass the Gateway token buffers when training-format trajectories are required.
