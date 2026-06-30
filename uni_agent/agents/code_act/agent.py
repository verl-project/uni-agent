"""CodeAct: the white-box agent driven by our own framework loop.

Our loop runs *outside* the task image: it serves the policy through the
task-created gateway session (a
:class:`~uni_agent.gateway.session.types.SessionHandle`), exposes host-side tools
(bound to the task's live sandbox), and feeds tool observations back to the policy
until it stops or hits :attr:`CodeActConfig.max_steps`. The actual session-driven
policy step is owned by the framework (:meth:`CodeActAgent._policy_step`, stubbed
here so the layer stays decoupled).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from ...tools import TOOL_REGISTRY, Toolbox
from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from ...gateway.session.types import SessionHandle
    from ...sandbox import Sandbox, SandboxBackend


class CodeActConfig(AgentConfig):
    """White-box launch params: host-side tools + a step budget."""

    name: str = "code_act"
    tools: list[dict] = Field(
        default_factory=lambda: [
            {"name": "stateful_shell", "command_timeout": 120},
            {"name": "str_replace_editor"},
        ],
        description="Host-side tools exposed to the policy (each a {name, ...kwargs} entry).",
    )
    max_steps: int = Field(default=50, description="Max tool-calling turns per episode.")


def _build_toolbox(sandbox: SandboxBackend, tools: list[dict[str, Any]]) -> Toolbox:
    """Build a :class:`Toolbox` over a *live* sandbox from ``{name, ...kwargs}`` entries.

    Each tool auto-parses its kwargs into its ``config_model`` (e.g. the shell's
    ``command_timeout``). Unlike :class:`~uni_agent.environment.Environment`, this
    does not touch the sandbox lifecycle -- the task already started it.
    """
    instances = []
    for entry in tools:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError(f"each tools entry must be a mapping with a 'name': {entry!r}")
        tool_name = entry["name"]
        kwargs = {k: v for k, v in entry.items() if k != "name"}
        cls = TOOL_REGISTRY.get(tool_name)
        if cls is None:
            raise KeyError(f"Unknown tool: {tool_name!r}")
        instances.append(cls(sandbox, **kwargs))
    return Toolbox(instances)


@register_agent("code_act")
class CodeActAgent(Agent):
    """White-box solver: framework loop + host-side tools over the gateway."""

    config_model = CodeActConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        sample: dict[str, Any],
        session: SessionHandle,
    ) -> AgentResult:
        cfg: CodeActConfig = self.config  # type: ignore[assignment]
        transcript: list[dict[str, Any]] = []
        toolbox = _build_toolbox(sandbox, cfg.tools)
        try:
            schemas = toolbox.schemas()  # handed to the policy via the session
            for _step in range(cfg.max_steps):
                action = await self._policy_step(session, schemas, transcript)
                if action is None:  # policy decided it is done
                    break
                obs = await toolbox.call(action["name"], action.get("args"))
                transcript.append({"action": action, "observation": obs.text})
        finally:
            await toolbox.close()  # release open channels (the task stops the sandbox)
        return AgentResult(transcript=transcript)

    async def _policy_step(
        self,
        session: SessionHandle,
        tool_schemas: list[dict],
        transcript: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """One session-driven policy step -> a tool call, or ``None`` when done.

        Owned by the framework: call the model at ``session.base_url`` (the
        per-session OpenAI-compatible endpoint) with ``tool_schemas`` and the
        running ``transcript``, then parse a tool call out of the reply. Stubbed
        here so the agent layer stays decoupled.
        """
        raise NotImplementedError("wire the session-driven policy step here")
