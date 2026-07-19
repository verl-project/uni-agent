# Core Abstractions

Uni-Agent separates an agent workload into small, replaceable abstractions. A `Task` defines what should happen and how success is measured; an `Agent` decides how to solve it; a `Toolbox` exposes actions; and a `Sandbox` provides the execution environment.

This section introduces each abstraction together with the interface used to customize it.

## Episode Lifecycle

A normal episode runs from the top down:

1. A runner resolves the task configuration for one dataset sample.
2. The `Task` starts its `Sandbox` and builds the configured `Agent`.
3. The `Agent` interacts with the sandbox directly or through a `Toolbox`.
4. The `Task` evaluates the resulting sandbox state.
5. The task returns a `TaskResult` containing the reward, metrics, and evaluation details.

```text
Task
├── Sandbox
├── Agent
│   └── Tool and Toolbox
└── Reward / Verification
```

When inference or training uses the verl-managed rollout path, the Uni-Agent Gateway additionally connects the agent's model requests to the rollout engine and materializes token-level trajectories.

## Ownership

- **Task** owns the episode lifecycle, task metadata, prompt, and reward computation.
- **Sandbox** owns the execution environment, process lifecycle, filesystem, and command data plane.
- **Agent** owns the solving strategy. A white-box agent owns its loop; a black-box agent delegates the loop to an external harness.
- **Tool** owns one model-visible action and any host-side state required by that action.
- **Toolbox** owns a set of tool instances bound to one sandbox.
- **Gateway** owns session-scoped model routing and token-level trajectory capture.

Clear ownership is important: Tasks start and stop sandboxes, Tools never manage sandbox lifecycle, and runtime model endpoints are injected by the runner instead of being stored in datasets.

## Configuration Model

The same configuration shape is used by standalone inference and training:

```yaml
- name: swe_bench
  sandbox:
    provider: modal
  agent:
    name: react
    tools:
      - name: stateful_shell
      - name: str_replace_editor
      - name: submit
    model:
      temperature: 0.8
      top_p: 0.9
      max_total_tokens: 65536
```

Dataset rows carry the sample-specific Task configuration under `extra_info.tools_kwargs.task`. Run-level YAML overrides are merged on top, and the live model endpoint is injected last.

## Customize Bottom-Up

The overview is top-down, but customization is easier in dependency order:

1. [Sandbox](sandbox.md) — add an execution backend.
2. [Tool and Toolbox](tool-and-toolbox.md) — expose new actions.
3. [Agent](agent.md) — implement a white-box loop or integrate a black-box harness.
4. [Task and Reward](task-and-reward.md) — compose the lower layers into a scored workload.
5. [Gateway and Trajectories](gateway-and-trajectories.md) — understand the training rollout path.
