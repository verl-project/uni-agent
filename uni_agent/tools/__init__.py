"""Host-side tool layer: the agent runs outside the task image and calls these.

Each :class:`Tool` is a schema plus an async ``run`` that drives the container
through the :class:`~uni_agent.sandbox.SandboxBackend` data plane and returns a
:class:`ToolResult`; :class:`Toolbox` binds a selection to one sandbox::

    from uni_agent.sandbox import LocalSandbox
    from uni_agent.tools import Toolbox

    async with LocalSandbox() as sandbox:
        tools = Toolbox.all(sandbox=sandbox)
        schemas = tools.schemas()                       # hand to the model
        obs = await tools.call("shell", {"command": "ls"})
        print(obs.text)
        await tools.close()                             # release open channels

Importing this package registers the built-ins in :data:`TOOL_REGISTRY`:
``stateful_shell`` (seen by the model as ``shell``), ``str_replace_editor``, and the
control tools ``finish`` / ``submit`` (no side effect; the CodeAct loop ends the
episode when the policy calls one -- see ``_FINISH_TOOLS``).
"""

from __future__ import annotations

from .base import (
    TOOL_REGISTRY,
    Tool,
    ToolCallFormatError,
    ToolError,
    ToolResult,
    ToolStatus,
    Toolbox,
    build_function_schema,
    get_tool,
    register_tool,
)
from .edit_file import EditFileTool
from .finish import FinishTool
from .shell import CommandResult, ShellChannel, ShellTool, ShellToolConfig
from .submit import SubmitTool

__all__ = [
    "Tool",
    "ToolError",
    "ToolCallFormatError",
    "ToolResult",
    "ToolStatus",
    "Toolbox",
    "TOOL_REGISTRY",
    "register_tool",
    "get_tool",
    "build_function_schema",
    "ShellTool",
    "ShellToolConfig",
    "EditFileTool",
    "FinishTool",
    "SubmitTool",
    "ShellChannel",
    "CommandResult",
]
