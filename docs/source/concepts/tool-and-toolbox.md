# Tool and Toolbox

A Tool is one model-visible action. It defines an OpenAI-compatible function schema and executes that action through a `SandboxBackend`. A Toolbox binds multiple Tool instances to one sandbox and keeps stateful tools alive across agent turns.

Tools run on the host side. Commands and file operations cross the Sandbox data plane; Tool code does not enter the task image.

!!! note "White-box agents only"
    Tool and Toolbox are part of the white-box Agent stack, where Uni-Agent owns the interaction loop and exposes explicit Tool schemas to the model. Black-box Agent Harnesses, such as Claude Code, own their own loop and built-in tools inside the Sandbox, so they do not use this module.

## Tool Contract

A Tool declares:

- `name`: model-facing function name.
- `description`: description shown to the model.
- `args_model`: Pydantic model for call arguments.
- `config_model`: optional Pydantic model for construction settings.
- `run()`: async implementation returning `ToolResult`.
- `start()` and `close()`: optional lifecycle hooks for stateful resources.

`ToolResult` contains model-visible `text` and a status:

```text
ok | format_error | error | timeout
```

`ToolResult.to_observation()` converts the result into the next-turn observation and clips oversized output.

## Build a Toolbox

Tool specs are configuration dictionaries. The `name` selects a registry entry; remaining fields are validated by that Tool's `config_model`.

```python
from uni_agent.tools import Toolbox

toolbox = Toolbox.from_specs(
    [
        {
            "name": "stateful_shell",
            "command_timeout": 120,
            "env_vars": {"PAGER": "cat"},
        },
        {"name": "str_replace_editor"},
        {"name": "submit"},
    ],
    sandbox=sandbox,
)
```

Enter the Toolbox before the agent loop:

```python
async with toolbox.entered(retry=3, timeout=60):
    schemas = toolbox.schemas()
    result = await toolbox.call(
        "shell",
        {"command": "python --version"},
    )
```

The Toolbox starts every Tool, dispatches calls by model-facing name, and closes all Tool instances on exit. Close failures are isolated so one Tool cannot prevent the others from releasing resources.

## Registry Name vs. Model Name

The configuration registry key may differ from the function name shown to the model.

The built-in shell is configured as:

```yaml
- name: stateful_shell
```

but appears in the model schema as:

```text
shell
```

This separation allows implementation names to remain stable while presenting concise function names to the model.

## Stateful Tools

Tool state belongs to the Tool instance, not the Sandbox:

- `stateful_shell` owns a persistent tmux channel, so cwd, exports, and background jobs survive between calls.
- `str_replace_editor` owns host-side edit history while files live in the Sandbox.
- `submit` and `finish` are stateless control tools.

All Tool instances in one Toolbox share the same Sandbox filesystem.

## Error Semantics

Use the exception type that matches the failure:

- Raise `ToolCallFormatError` for malformed names or arguments.
- Raise `ToolError` for expected runtime failures that the model can correct.
- Return `ToolResult(status="timeout")` when the Tool handles a timeout.
- Let unexpected exceptions propagate; they represent implementation or infrastructure failures.

`Toolbox.call()` converts format and expected runtime errors into observations. It intentionally does not hide unexpected exceptions.

## Custom Tool

Define argument and optional construction models:

```python
from pydantic import BaseModel, Field

from uni_agent.tools.base import Tool, ToolResult, register_tool


class ReadTextArgs(BaseModel):
    path: str = Field(description="Absolute file path")


@register_tool("read_text")
class ReadTextTool(Tool):
    description = "Read a UTF-8 text file."
    args_model = ReadTextArgs

    async def run(
        self,
        args: dict,
        *,
        timeout: float | None = None,
    ) -> ToolResult:
        data = await self.sandbox.read_file(args["path"])
        return ToolResult(text=data.decode("utf-8"))
```

Make sure the module is imported before `get_tool()` is called. Built-in tools are imported for registration from `uni_agent.tools.__init__`.

## Implementation Rules

- Depend on `SandboxBackend`, not provider-specific classes.
- Do not start or stop the Sandbox from a Tool.
- Put call arguments in `args_model` and construction settings in `config_model`.
- Use `ToolError` only for failures the model should see and recover from.
- Release channels, subprocesses, and temporary state in `close()`.
- Keep observations concise and non-interactive.
- Use the registry key in YAML and the model-facing Tool name in calls.
