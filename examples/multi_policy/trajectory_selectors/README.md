# Trajectory selector

Selector for
`actor_rollout_ref.rollout.custom.agent_framework.agent_runners.<name>.trajectory_selection.custom_select_function`.

The multi_policy framework wraps every custom select function: it splits the
finalized session by `role` and calls the function **once per role** with that
role's trajectories. Write the rule as if it receives one agent's list, exactly
like a single-agent selector.

## Use it

```yaml
agent_runners:
  multi_agent:
    trajectory_selection:
      custom_select_function:
        path: /abs/path/to/examples/multi_policy/trajectory_selectors/selectors.py
        name: longest
```

With this configuration every role keeps its longest trajectory.

Use an absolute file path, not `pkg://examples.*`: `examples/` resolution
follows `sys.path` order, so a second checkout earlier on the path can win
silently.

## Function

| Function | Keeps |
| --- | --- |
| `longest` | the longest trajectory in the list it receives |

"Longest" compares, in order: `sum(response_mask)`, `len(response_ids)`,
`num_turns`, and the trajectory's position in the list (later wins) — the same
tie-breakers as the framework's built-in `longest`.

## Notes

- The wrapper is applied by `MultiAgentGatewayFramework`, so every
  `custom_select_function` in this recipe is called per role. A plain function
  like `longest` is therefore a per-role rule.
- A role with no trajectories is not called.
- A rule that raises for one role keeps that role's trajectories unchanged; the
  other roles still run their rules.
- Dropped trajectories do not train. With `log_dir` set they are written as
  `dropped_trajectory.*` next to the kept `trajectory.*` under
  `agents/<role>/`.
- `extra_fields["message_history"]` is available while the rule runs and is
  released before reward scoring and TransferQueue writes.
- To add a rule, implement `def my_rule(session_id: str, trajectories: list) ->
  list` in a module and point `custom_select_function` at `{path, name}`. Sync
  and async functions are both supported.
