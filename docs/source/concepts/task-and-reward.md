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
- `prompt`: source OpenAI-style messages supplied by the dataset or caller.
- `prompt_template`: optional recipe-owned messages rendered before the Agent starts.
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

### Source and Effective Prompts

Datasets should keep `prompt` agent-neutral. SWE preprocessors, for example, emit one user message containing the issue text. A Task recipe can turn that source into the complete Agent-specific input with `prompt_template`:

```yaml
- name: swe_bench
  prompt_template:
    - role: system
      content: You are a software engineer working in an existing repository.
    - role: user
      content: |-
        Resolve this issue in /testbed:

        {prompt}
  agent:
    name: claude_code
```

The runtime binds the top-level source `prompt` before Task Config resolution. A recipe-file `prompt_template` owns the complete template and cannot be replaced by a same-named value serialized in a dataset row. Other Task Config fields retain their normal merge behavior.

The first template contract is intentionally strict:

- The source must be exactly one user message.
- Template output is a list of messages with string `role` and string `content`.
- Exactly one `{prompt}` field is required. Other fields, conversions, and format specifications are rejected.
- Content equal to `{prompt}` preserves the source content object, including structured multimedia blocks.
- Embedding `{prompt}` in surrounding text requires string source content.
- Use standard Python formatting escapes, `{{` and `}}`, for literal braces.

Without `prompt_template`, messages pass through unchanged, so existing already-rendered datasets remain compatible with template-free recipes. Combining a template with legacy multi-message rendered input fails validation instead of rendering a prompt twice.

After rendering, `TaskConfig.prompt` is the effective message list passed to the Agent. `prompt_template` is an input-only rendering directive: serialized Task configs contain the effective prompt and omit the template body, preventing a later config round-trip from rendering it again. Framework-managed execution keeps the dataset value as `source_prompt` and exposes the effective messages as downstream `raw_prompt` to RewardLoop and TransferQueue consumers. This provenance copy is local to each rollout session; concurrent sessions never mutate the shared sample mapping. Reward-info telemetry remains limited to reward, accuracy, and completion state.

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
            finished=agent_result.finished,
            extra_info={"score": score},
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
    finished=agent_result.finished,
    extra_info=result,
)
```

Custom Tasks may return scalar, dense, rubric-based, or multi-component rewards. The framework consumes
`TaskResult.reward`; additional metrics belong in `accuracy` and `extra_info`.

`TaskResult.finished` is factual episode metadata copied from
`AgentResult.finished`; it does not decide whether the trajectory contributes to
training. The Agent Framework owns that policy through
`mask_unfinished_episode`, so the same Task Config can be reused for
inference, evaluation, and different training runs without embedding optimizer
behavior in the Task or dataset.

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
