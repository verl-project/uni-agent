# Core Abstractions

Uni-Agent separates an agent workload into small, replaceable abstractions.
This section introduces each abstraction together with the interface used to customize it.

![Uni-Agent architecture overview](../assets/uni-agent.png){ width="800" }

A `Task` defines what should happen and how success is measured; an `Agent` decides how to solve it; a `Toolbox` exposes actions; and a `Sandbox` provides the execution environment.

## Execution Lifecycle

A normal episode runs from the top down:

1. A runner resolves the task configuration for one dataset sample.
2. The `Task` starts its `Sandbox` and builds the configured `Agent`.
3. The `Agent` interacts with the sandbox directly (black-box agent) or through a `Toolbox` (white-box agent).
4. The `Task` evaluates the resulting sandbox state via `reward` module.
5. The task returns a `TaskResult` containing the reward, metrics, and evaluation details.

```text
Task
├── Sandbox
├── Agent
│   └── Tool and Toolbox (Optional)
└── Reward / Verification
```

When inference or training uses the verl-managed rollout path, the Uni-Agent Gateway additionally connects the agent's model requests to the rollout engine and materializes token-level trajectories.

## Ownership

- **Gateway** owns session-scoped model routing and token-level trajectory capture for the training pipeline.
- **Task** owns the Task execution lifecycle, task metadata, prompt, and reward computation.
- **Agent** owns the solving strategy. A white-box agent owns its loop; a black-box agent delegates the loop to an external agent harness, such as Claude Code.
- **Tool** owns one model-visible action and any host-side state required by that action. **Toolbox** owns a set of tool instances bound to one sandbox.
- **Sandbox** owns the execution environment, filesystem, and command data plane.

## Task Configuration

Uni-Agent uses two Task Config layers for both standalone inference and training:

1. **Task Config:** run-level defaults shared by the workload.
2. **Sample Config:** sample-wise values, merged on top of the Task Config.

The run-level Task Config defines common behavior such as the Agent, Tools, Sandbox provider, and sampling settings:

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

Each sample can then provide or override fields such as its prompt, metadata, sandbox image, budgets, or other nested Task settings:

```yaml
name: swe_bench
sandbox:
  image: swebench/sweb.eval.x86_64.example
agent:
  max_steps: 300
prompt:
  - role: user
    content: Fix the issue in /testbed.
metadata:
  instance_id: example
```

Nested dictionaries are merged recursively, while lists and scalar values from the Sample Config replace Task Config defaults. This allows one batch to contain heterogeneous tasks or sandbox environments while still sharing a common Agent and rollout configuration.

The live model endpoint, API key, and served model name are injected after both layers. They are runtime state rather than a third user-configurable layer, so Sample Config cannot replace the active policy endpoint.

`TaskConfigResolver` is the shared entry point used by standalone inference and the Agent Framework:

```python
from uni_agent.tasks import TaskConfigResolver

resolver = TaskConfigResolver.from_file("task_config.yaml")
resolved = resolver.resolve(
    sample_config,
    runtime_model={
        "base_url": model_endpoint,
        "api_key": api_key,
        "model_name": model_name,
    },
)
```

The resolver loads every named Task Config, routes each Sample by `name`, applies the merge order, and validates missing or duplicate routes consistently.

## Customize Bottom-Up

The overview is top-down, but customization is easier in dependency order:

1. [Sandbox](sandbox.md) — add an execution backend.
2. [Tool and Toolbox](tool-and-toolbox.md) — expose new actions.
3. [Agent](agent.md) — implement a white-box loop or integrate a black-box harness.
4. [Task and Reward](task-and-reward.md) — compose the lower layers into a scored workload.
5. [Gateway and Trajectories](gateway-and-trajectories.md) — understand the training rollout path.
