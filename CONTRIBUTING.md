# Contributing to Uni-Agent

Thank you for contributing to Uni-Agent. Bug fixes, new Agent and Task integrations, Sandbox backends, Tools, training improvements, tests, examples, and documentation are all welcome.

## Getting Started

1. Fork and clone the repository.
2. Install Uni-Agent in editable mode:

   ```bash
   python -m pip install -e .
   ```

3. Install the optional dependencies for the Task, Sandbox provider, inference path, or training workflow you plan to change. See the [installation guide](https://uni-agent.readthedocs.io/en/latest/quickstart/installation.html).
4. Install the development hooks:

   ```bash
   python -m pip install pre-commit
   pre-commit install
   ```

## Choose the Owning Layer

Keep behavior in the abstraction that owns it:

- **Task** owns the episode lifecycle, sample metadata, configuration, reward, and verification.
- **Agent** owns the solving strategy and either implements or launches the agent loop.
- **Tool** owns one model-visible action; **Toolbox** binds Tool instances to a Sandbox.
- **Sandbox** owns the execution environment, lifecycle, filesystem, and command data plane.
- **Gateway** owns model-protocol adaptation, session routing, and token-level trajectory capture.
- **Framework** owns rollout orchestration, failure isolation, and training-record delivery.
- **Logging** owns shared context, handlers, artifacts, and sensitive-data redaction.

For example, reward policy belongs in a Task, provider failures belong in a Sandbox, and protocol encoding belongs in the Gateway. Avoid adding a cross-layer shortcut when the owning layer can expose a narrow interface instead.

## Development Workflow

- Keep each change focused on one problem.
- Add regression tests for fixes and boundary tests for public contracts.
- Cover lifecycle cleanup and failure paths for asynchronous or remote resources.
- Update documentation and runnable examples when APIs, configuration, or workflows change.
- Include reproducible measurements for scheduling, throughput, memory, or training-performance claims.
- Do not commit credentials, private model responses, proprietary data, or unredacted logs.

Before opening a pull request, run:

```bash
pre-commit run --all-files --show-diff-on-failure
python -m pytest tests/uni_agent/framework/test_task_runner.py  # replace with relevant tests
```

Use the smallest relevant test selection locally. CI may run broader checks.

## Pull Request Title

PR titles are checked in CI and must use:

```text
[area] type: summary
```

Allowed areas:

```text
agents, framework, gateway, logging, sandbox, tasks, tools, training,
app, docs, examples, ci, build, deps, misc
```

Allowed types:

```text
feat, fix, refactor, perf, test, docs, chore, revert
```

Use lowercase area and type names. Separate multiple areas with comma-space:

```text
[agents, sandbox] feat: add isolated harness execution
```

Prefix changes that break a public API, Task or Sample Config, dataset, model protocol, trajectory contract, checkpoint, or documented workflow with `[BREAKING]`:

```text
[BREAKING][tasks, docs] refactor: replace task config schema
```

A stacked series may use `[1/N]` or a numeric total:

```text
[1/N][gateway] refactor: split protocol adapters
```

## Submitting a Pull Request

- Fill in the Summary, Changes, Validation, and Compatibility sections.
- Link the issue with `Fixes #123` when the PR resolves it.
- List exact validation commands and results. If a check was not run, explain why.
- Call out the owning layer and important design trade-offs for non-trivial changes.
- Document migration steps for every breaking change.

Reviewers will focus on correctness, ownership boundaries, compatibility, failure handling, test coverage, and reproducibility.
