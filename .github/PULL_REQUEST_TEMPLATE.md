## Summary

<!--
Explain the problem, the chosen solution, and why this belongs in Uni-Agent.
Link related work with "Fixes #123" or "Related to #123" when applicable.
-->

## Changes

<!--
List the important changes. For non-trivial work, identify the owning layer
(Task, Agent, Tool, Sandbox, Gateway, or Framework) and note key trade-offs.
-->

-

## Validation

<!--
List exact commands and results, including focused unit tests and relevant
end-to-end runs. If a check was not run, write "Not run" and explain why.
Include benchmark methodology for performance-sensitive changes.
-->

-

## Compatibility

<!--
Describe impacts on public Python APIs, Task/Sample Config, datasets, model
protocols, trajectories, checkpoints, and existing recipes. Write "None" when
there is no compatibility impact. For a breaking change, include migration steps.
-->

None.

## PR title

Use `[area] type: summary`.

- Areas: `agents`, `framework`, `gateway`, `logging`, `sandbox`, `tasks`, `tools`, `training`, `app`, `docs`, `examples`, `ci`, `build`, `deps`, `misc`
- Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `revert`
- Separate multiple areas with comma-space: `[agents, sandbox] feat: add isolated harness execution`
- Prefix compatibility-breaking work with `[BREAKING]`: `[BREAKING][tasks, docs] refactor: replace task config schema`
- A stacked series may start with `[1/N]`: `[1/N][gateway] refactor: split protocol adapters`

## Checklist

- [ ] The PR is focused and linked to an issue or explains why no issue is needed.
- [ ] The title follows the format above and names the layer that owns the change.
- [ ] Tests cover the behavior, or the Validation section explains why they are not practical.
- [ ] User-facing API, config, and workflow changes include documentation or runnable examples.
- [ ] Compatibility impact and migration steps are documented.
- [ ] Logs, fixtures, and examples contain no credentials or private data.
- [ ] `pre-commit run --all-files --show-diff-on-failure` passes.
