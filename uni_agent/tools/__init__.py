"""Host-side tool layer: the agent runs outside the task image and calls these.

Each :class:`Tool` is the single agent-facing unit -- a schema plus an async
``run`` that drives the container through the
:class:`~uni_agent.sandbox.SandboxBackend` data plane and returns a normalized
:class:`Observation`. A tool is built with its sandbox and owns whatever state it
needs (e.g. ``shell`` keeps a persistent shell channel); there is no separate
session layer. Bind a selection of tools to a sandbox with :class:`Toolbox`::

    from uni_agent.sandbox import LocalSandbox
    from uni_agent.tools import Toolbox

    async with LocalSandbox() as sandbox:
        tools = Toolbox.all(sandbox=sandbox)
        schemas = tools.schemas()                       # hand to the model
        obs = await tools.call("shell", {"command": "ls"})
        print(obs.text)
        await tools.close()                             # release open channels

Importing this package registers the built-in tools (under registry keys
``stateful_shell`` and ``str_replace_editor``) in :data:`TOOL_REGISTRY`; the
shell tool is surfaced to the model as ``shell``.
"""

from __future__ import annotations

from .base import (
    TOOL_REGISTRY,
    Observation,
    Tool,
    ToolError,
    Toolbox,
    build_function_schema,
    get_tool,
    register_tool,
)
from .edit_file import EditFileTool
from .shell import CommandResult, ShellChannel, ShellTool, ShellToolConfig

__all__ = [
    "Tool",
    "ToolError",
    "Toolbox",
    "Observation",
    "TOOL_REGISTRY",
    "register_tool",
    "get_tool",
    "build_function_schema",
    "ShellTool",
    "ShellToolConfig",
    "EditFileTool",
    "ShellChannel",
    "CommandResult",
]
