"""Host-side tool layer: the agent runs outside the task image and calls these.

Each :class:`Tool` is a schema plus an async ``run`` that drives the container
through the :class:`~uni_agent.sandbox.SandboxBackend` data plane and returns a
:class:`ToolResult`; :class:`Toolbox` binds a selection to one sandbox::

    from uni_agent.sandbox import LocalSandbox
    from uni_agent.tools import Toolbox

    async with LocalSandbox() as sandbox, Toolbox.all(sandbox=sandbox) as tools:
        schemas = tools.schemas()                       # hand to the model
        obs = await tools.call("shell", {"command": "ls"})
        print(obs.text)                                 # `async with` starts + closes the tools

Importing this package registers the built-ins in :data:`TOOL_REGISTRY`:
``stateful_shell`` (seen by the model as ``shell``), ``str_replace_editor``, and the
control tools ``finish`` / ``submit`` (no side effect; the ReAct loop ends the
episode when the policy calls one -- see ``_FINISH_TOOLS``).
"""

from __future__ import annotations

from .base import (
    Tool,
    ToolCallFormatError,
    ToolError,
    ToolResult,
    ToolStatus,
    Toolbox,
)

# Built-in tools self-register on import; keep these imports for that side effect.
from . import edit_file, finish, shell, submit  # noqa: F401

__all__ = [
    "Tool",
    "ToolError",
    "ToolCallFormatError",
    "ToolResult",
    "ToolStatus",
    "Toolbox",
]
