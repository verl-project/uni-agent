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
- `prompt`: an Agent-neutral dataset/source message list on input; after optional template rendering, the Agent-facing messages held by the resolved Task Config.
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

### Source Prompts and Runtime Templates

When a recipe supplies `prompt_template`, datasets should keep the source `prompt` agent-neutral. SWE preprocessors, for example, emit one user message whose content is the problem statement and retain `problem_statement` plus evaluator fields in Task `metadata`. A Task recipe can build complete Agent-specific messages by formatting metadata fields:

```yaml
- name: swe_bench
  prompt_template:
    - role: user
      content: |-
        Resolve this issue in /testbed:

        Language: {language}

        {problem_statement}
  agent:
    name: claude_code
```

The runtime treats the top-level dataset `prompt` as the authoritative source message list and binds it before Task Config resolution, overwriting any stale nested `task.prompt`. A recipe-file `prompt_template` owns the complete template and cannot be replaced by a same-named value serialized in a dataset row. Other Task Config fields retain their normal merge behavior.

A Task Config YAML file may contain entries for several task names. The file is parsed and indexed as a whole, so invalid YAML, entries without `name`, and duplicate names still fail at load time. The resolver merges and validates only the entry whose `name` matches the sample Task Config; an unused entry is never rendered or passed to an Agent.

Runtime templates are intentionally text-only:

- Template output is a list of messages with a non-empty string `role` and string `content`.
- Each content string uses Python standard-library brace parsing to replace direct Task `metadata` fields such as `{problem_statement}`. Fields may be repeated or omitted from the template.
- Placeholder names must be simple identifiers. Attribute or index access, conversions such as `!r`, and format specifications such as `:>10` are rejected.
- Missing fields, malformed templates, non-text replacement values, and non-string template content fail validation before the Agent starts.
- Use standard Python formatting escapes, `{{` and `}}`, for literal braces.
- Image, video, audio, and other structured message content are not supported by runtime templates. Multimodal template rendering is deferred.

Without `prompt_template`, the dataset/source messages pass through unchanged. They may therefore already contain complete Agent instructions or structured multimodal content; end-to-end support for that content still depends on the selected Agent, API adapter, and model processor. Template-free pass-through is also the intended path for self-rendering Agents. For example, the planned mini-swe-agent integration will read the source user content as the problem statement and apply its own template inside the Sandbox. The Task still calls every Agent through the uniform `Agent.run(sandbox, messages, workdir=None)` interface and does not pass metadata or branch on Agent name.

After Task rendering, `TaskConfig.prompt` is the message list passed to `Agent.run()`. `prompt_template` is an input-only rendering directive and is omitted when Task configs are serialized. In Framework-managed execution, verl uses the source prompt for loader-time token-length checks when overlong-prompt filtering is enabled, makes it available as `raw_prompt` when a configured RewardLoop or judge scoring path is used, and preserves it as metadata in records written to TransferQueue. Task-rendered messages do not replace that source value. The trajectory token tensors are instead built from the Agent's actual model requests captured by the Gateway. Built-in SWE Tasks evaluate from `TaskConfig.metadata`, independently of `raw_prompt`.

Task-rendered messages are not guaranteed to equal a self-rendering Agent's final internal prompt. Such an Agent may apply its own Sandbox-side template, and the current Task Runner cannot observe its final internal messages. `TaskResult` consequently reports episode results only and does not attempt to carry prompt provenance.

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
                workdir=None,
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
    "prompt": [{"role": "user", "content": problem_statement}],
    "extra_info": {
        "tools_kwargs": {
            "task": {
                "name": "my_task",
                "sandbox": {"image": "..."},
                "metadata": {
                    "problem_statement": problem_statement,
                    ...,
                },
            }
        }
    },
}
```

Keep datasets provider-agnostic when possible. For example, SWE-Bench rows store canonical image references; the selected Sandbox provider maps them to its registry at runtime.

## Runtime Configuration

Task configuration has two user-defined layers:

1. Run-level Task Config provides shared defaults.
2. The sample's serialized `tools_kwargs.task` is merged on top and normally wins on conflicts.

Nested dictionaries are deep-merged. Lists and scalar values from the Sample Config replace Task Config defaults. The exception is a recipe-file `prompt_template`, which remains authoritative so a serialized sample cannot replace the selected Agent recipe.

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
