# Trajectory selectors

Sample `custom_select_function` implementations for
[`trajectory_selection`](../../../docs/source/concepts/gateway-and-trajectories.md#trajectory-selection).

A session can split into several trajectories: a title request, a compaction
replay, a subagent shell, plus the chain that did the real work. `all` and
`longest` cannot tell them apart, so a runner can point at a function here instead
and decide for itself what to keep.

## Use one

```yaml
agent_runners:
  my_agent:
    trajectory_selection:
      custom_select_function:
        path: /abs/path/to/examples/quickstart/trajectory_selectors/selectors.py
        name: drop_non_work_segments
```

Use a file path, not `pkg://examples.*`: `examples/` has no `__init__.py`, so
`pkg://` resolution follows `sys.path` order and a second checkout earlier on the
path wins silently — the run then uses a selector other than the one you edited.

## What is here

| Function | Keeps |
|---|---|
| `drop_title_segments` | everything but title generation |
| `drop_compaction_segments` | everything but compaction replays |
| `drop_subagent_segments` | everything but subagent shells |
| `drop_non_work_segments` | all three dropped |
| `keep_first_work_segment` | only the earliest non-title chain (`chain_id` order) |

The predicates behind them — `is_title_segment`, `is_compaction_segment`,
`is_subagent_segment` — are exported too, so a recipe can compose its own policy
instead of copying marker lists.

## The markers are examples, not a library

Each marker list (`TITLE_MARKERS`, `COMPACTION_MARKERS`, `SUBAGENT_MARKERS`)
matches text a particular agent happens to write. Check them against the agent you
are filtering before trusting a selector:

- **Compaction is the easy one to get backwards.** Match the *summarisation
  request* prompt, which is its own chain. The summary is later injected into the
  working chain, so a marker that also appears there deletes the task rollout right
  after its first compaction — the opposite of the intent.
- **Title generation** only shows up if something upstream asks for it. If the
  agent never names sessions, `drop_title_segments` is a no-op.
- **Subagent shells** differ per harness (a system prompt, a tool call, a protocol
  frame). Widen or narrow the tuple to fit.

## What the framework guarantees

- Called once per session, after finalization and before the postprocessor and
  reward scoring.
- `trajectories` is the session's full list, in the gateway's close order.
  Reward scoring dispatches the *last* entry you return — in close order that is
  the chain active most recently, normally the one holding the task result.
- Each entry carries `chain_id`, the creation order (from 1) — use it when you
  want "which chain came first" rather than "which one is last in this list".
- Each trajectory carries `extra_fields["message_history"]`: a shallow copy of
  the chain's chat text, shared with the session, so read it but do not mutate it.
- An empty return drops the session (`status="empty"`, no training data). A raised
  exception keeps everything.

Message text is released as soon as your function returns, before scoring and
before anything is written to the TransferQueue, so it never reaches training data
and no later stage can read it.
