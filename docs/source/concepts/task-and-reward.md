# Task and Reward

A Task is the top-level unit executed by inference and training. It combines one sample's prompt and metadata with an Agent, a Sandbox, and reward logic.

The Task owns the complete episode:

```text
start logging
    -> start sandbox
    -> run agent
    -> evaluate sandbox state
    -> return TaskResult
    -> stop sandbox
```

## Task Configuration

Every Task configuration inherits from `TaskConfig`:

- `name`: registered Task family.
- `sandbox`: `SandboxConfig`.
- `agent`: concrete Agent configuration.
- `prompt`: OpenAI-style messages.
- `metadata`: sample-specific data used by execution and scoring.

Task-specific configs can add validated fields:

```python
from pydantic import Field

from uni_agent.tasks.base import TaskConfig


class MyTaskConfig(TaskConfig):
    name: str = "my_task"
    eval_timeout: float = Field(default=300)
```

Unknown fields are rejected. Agent mappings are resolved through the Agent registry into the correct AgentConfig subclass.

## Episode Implementation

A Task implements `run()` without arguments because all sample state lives on its config:

```python
from uni_agent.tasks.base import Task, TaskResult
from uni_agent.tasks.registry import register_task


@register_task("my_task")
class MyTask(Task):
    config_model = MyTaskConfig

    async def run(self) -> TaskResult:
        config: MyTaskConfig = self.config

        async with self.build_sandbox() as sandbox:
            agent = self.build_agent()
            agent_result = await agent.run(
                sandbox=sandbox,
                messages=config.prompt,
            )

            score = await compute_reward(
                config.metadata,
                sandbox,
                agent_result,
            )

        return TaskResult(
            reward=score,
            accuracy=score,
            info={"score": score},
        )
```

`build_sandbox()` and `build_agent()` dispatch through their registries. Logging is provided by the runtime that invokes the Task; the Task only emits normal log records.

## Reward Design

Uni-Agent does not impose a Reward base class. Reward logic belongs to the Task because different workloads evaluate different artifacts.

SWE tasks use an async function:

```python
async def compute_reward(
    metadata: dict,
    sandbox,
    eval_timeout: float = 300,
) -> dict:
    ...
```

The built-in SWE-Bench tasks:

1. Write an evaluation script into the Sandbox.
2. Execute tests against `/testbed`.
3. Parse the test output.
4. Return `resolved`, evaluation status, timing, and a detailed report.

The Task converts that payload into `TaskResult`:

```python
TaskResult(
    reward=float(result["resolved"]),
    accuracy=float(result["resolved"]),
    info=result,
)
```

Custom Tasks may return scalar, dense, rubric-based, or multi-component rewards. The framework consumes `TaskResult.reward`; additional metrics belong in `accuracy` and `info`.

## Dataset Contract

Preprocessing should serialize the sample-specific Task configuration into each dataset row:

```python
{
    "prompt": prompt,
    "extra_info": {
        "tools_kwargs": {
            "task": {
                "name": "my_task",
                "sandbox": {"image": "..."},
                "prompt": prompt,
                "metadata": {...},
            }
        }
    },
}
```

Keep datasets provider-agnostic when possible. For example, SWE-Bench rows store canonical image references; the selected Sandbox provider maps them to its registry at runtime.

## Runtime Configuration

Task configuration has two user-defined layers:

1. Run-level Task Config provides shared defaults.
2. The sample's serialized `tools_kwargs.task` is merged on top and wins on conflicts.

Nested dictionaries are deep-merged. Lists and scalar values from the Sample Config replace Task Config defaults.

The runtime injects `agent.model.base_url`, API key, and served model name after the two layers. Endpoint information is not sample-overridable because it belongs to the live policy service.

`TaskConfigResolver` implements this routing and merge order for both standalone inference and Framework-managed rollouts.

This allows one dataset batch to customize prompts, metadata, Sandbox images, Agents, or budgets sample by sample while retaining shared defaults.

## Register a Task

Register the class and lazy module:

```python
@register_task("my_task")
class MyTask(Task):
    ...
```

```python
TASK_MODULES["my_task"] = "my_package.task"
```

`get_task()` accepts either a typed `TaskConfig` or a serialized mapping and validates it through the registered Task's `config_model`.

## Implementation Rules

- Keep the Task responsible for the Sandbox and Task execution lifecycle.
- Keep model-serving endpoints out of preprocessed datasets.
- Put sample-specific evaluation data in `metadata`.
- Emit normal log records and let the invoking runtime bind their `LogContext`.
- Return a `TaskResult` for every successful episode.
- Let infrastructure failures propagate instead of silently converting them to zero reward.
- Keep reward implementation close to the Task; do not force unrelated tasks into one reward schema.
- Add preprocessing, a runnable Task Config, and tests for both successful and failed evaluations.
