"""CodeAct: the white-box agent driven by our own framework loop.

Our loop runs *outside* the task image: it serves the policy at the task-supplied
``base_url`` / ``api_key`` (an OpenAI-compatible endpoint), exposes host-side tools
(bound to the task's live sandbox), and feeds tool observations back to the policy
until it stops or hits :attr:`CodeActConfig.max_steps`. The actual policy step is
owned by the framework (:meth:`CodeActAgent._policy_step`, stubbed here so the
layer stays decoupled).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from ...tools import Toolbox
from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from ...sandbox import Sandbox


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


@register_agent("code_act")
class CodeActAgent(Agent):
    """White-box solver: framework loop + host-side tools over an OpenAI endpoint."""

    config_model = CodeActConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        base_url: str,
        api_key: str,
        messages: list[dict[str, Any]],
    ) -> AgentResult:
        cfg: CodeActConfig = self.config  # type: ignore[assignment]
        transcript: list[dict[str, Any]] = list(messages)  # seed the conversation
        toolbox = Toolbox.from_specs(cfg.tools, sandbox=sandbox)
        try:
            schemas = toolbox.schemas()  # handed to the policy at base_url
            for _step in range(cfg.max_steps):
                action = await self._policy_step(base_url, api_key, schemas, transcript)
                if action is None:  # policy decided it is done
                    break
                obs = await toolbox.call(action["name"], action.get("args"))
                transcript.append({"action": action, "observation": obs.text})
        finally:
            await toolbox.close()  # release open channels (the task stops the sandbox)
        return AgentResult(transcript=transcript)

    async def _policy_step(
        self,
        base_url: str,
        api_key: str,
        tool_schemas: list[dict],
        transcript: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """One policy step -> a tool call, or ``None`` when done.

        Owned by the framework: call the model at ``base_url`` (an OpenAI-compatible
        endpoint, authed with ``api_key``) with ``tool_schemas`` and the running
        ``transcript``, then parse a tool call out of the reply. Stubbed here so the
        agent layer stays decoupled.
        """
        raise NotImplementedError("wire the policy step here")
