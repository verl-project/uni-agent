"""ReAct: the white-box agent driven by our own framework loop (reason + tool-call act)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import Field

from uni_agent.tools import Toolbox

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent
from .model import OpenAICompatibleChatModel

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

#: Tool names that end the episode when the policy calls them.
_FINISH_TOOLS = {"submit", "finish"}


class ReActConfig(AgentConfig):
    """White-box launch params: host-side tools + step / timeout budgets."""

    name: str = "react"
    tools: list[dict] = Field(
        default_factory=lambda: [
            {"name": "stateful_shell", "command_timeout": 120},
            {"name": "str_replace_editor"},
            {"name": "submit"},
        ],
        description="Host-side tools exposed to the policy (each a {name, ...kwargs} entry). "
        "Include a finish tool (submit/finish) so the policy can end the episode explicitly.",
    )
    max_steps: int = Field(default=50, description="Max tool-calling turns per episode.")
    action_timeout: float | None = Field(
        default=None,
        description="Per-call timeout (s) forwarded to each tool call, overriding the tool's own "
        "default (e.g. the shell's command_timeout); None defers to the tool's own timeout.",
    )
    timeout_budget: int = Field(
        default=3,
        description="Tool-call timeouts tolerated per episode before it stops (exit_reason='timeout_limit').",
    )


@register_agent("react")
class ReActAgent(Agent):
    """White-box solver: framework loop + host-side tools over an OpenAI endpoint."""

    config_model = ReActConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
    ) -> AgentResult:
        cfg: ReActConfig = self.config  # type: ignore[assignment]
        if cfg.model.base_url is None:
            raise ValueError("react: config.model.base_url is not set (the endpoint the policy calls)")

        toolbox = Toolbox.from_specs(cfg.tools, sandbox=sandbox)
        model = OpenAICompatibleChatModel(
            base_url=cfg.model.base_url,
            api_key=cfg.model.api_key,
            model_name=cfg.model.model_name,
            sampling_params={
                "temperature": cfg.model.temperature,
                "top_p": cfg.model.top_p,
                "top_k": cfg.model.top_k,
            },
            tools_schemas=toolbox.schemas(),
        )

        transcript: list[dict[str, Any]] = list(messages)
        for message in messages:
            logger.info(f"{str(message.get('role', '')).upper()} PROMPT:\n{message.get('content', '')}")
        trajectory_info: dict[str, Any] = {
            "steps": 0,
            "num_tool_calls": 0,
            "timeouts": 0,
            "errors": 0,
            "total_tokens": 0,
            "exit_reason": "unknown",
        }
        try:
            async with toolbox.entered(retry=3, timeout=60):
                for step_idx in range(1, cfg.max_steps + 1):
                    trajectory_info["steps"] = step_idx
                    stop_reason = await self.step(cfg, model, toolbox, transcript, trajectory_info)
                    if stop_reason != "completed":
                        trajectory_info["exit_reason"] = stop_reason
                        break
                else:  # loop ran the full step budget without an early stop
                    trajectory_info["exit_reason"] = "max_steps"
                    logger.warning(f"Reached max steps ({cfg.max_steps}) without finishing.")
        except Exception as exc:  # keep the partial transcript; the task buckets the failure
            logger.exception("react loop failed at step %s", trajectory_info["steps"])
            trajectory_info["exit_reason"] = "unknown_error"
            trajectory_info["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            await model.aclose()  # release the reused HTTP session

        logger.info(
            f"Episode done: exit_reason={trajectory_info['exit_reason']} steps={trajectory_info['steps']} "
            f"tool_calls={trajectory_info['num_tool_calls']} timeouts={trajectory_info['timeouts']} "
            f"errors={trajectory_info['errors']} total_tokens={trajectory_info['total_tokens']}"
        )
        return AgentResult(transcript=transcript, info=trajectory_info)

    async def step(
        self,
        cfg: ReActConfig,
        model: OpenAICompatibleChatModel,
        toolbox: Toolbox,
        transcript: list[dict[str, Any]],
        info: dict[str, Any],
    ) -> str:
        """Run one turn: query the policy, record its message, run its tool calls."""
        logger.info(f"{'=' * 25} STEP {info['steps']} {'=' * 25}")

        # step 1: query the model
        max_tokens = cfg.model.max_tokens_per_turn or cfg.model.max_total_tokens
        if cfg.model.max_total_tokens is not None:
            remaining = cfg.model.max_total_tokens - info["total_tokens"]
            if remaining <= 0:
                logger.info(f"Exit: token budget spent ({info['total_tokens']}/{cfg.model.max_total_tokens}).")
                return "token_limit"
            max_tokens = min(max_tokens, remaining)

        sampling_params: dict[str, Any] = {
            "temperature": cfg.model.temperature,
            "top_p": cfg.model.top_p,
            "top_k": cfg.model.top_k,
        }
        if max_tokens is not None:  # both budgets unset -> let the server run to EOS
            sampling_params["max_tokens"] = max_tokens
        content, tool_calls, gen_info = await model.query(transcript, sampling_params=sampling_params)
        info["total_tokens"] = gen_info["prompt_tokens"] + gen_info["completion_tokens"]
        logger.info(
            f"Prompt Tokens: {gen_info['prompt_tokens']}, Completion Tokens: {gen_info['completion_tokens']} "
            f"(total {info['total_tokens']})"
        )
        logger.info(f"💭 THOUGHT:\n{content}")

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        transcript.append(assistant_msg)

        if cfg.model.max_total_tokens is not None and info["total_tokens"] >= cfg.model.max_total_tokens:
            logger.info(f"Exit: token budget reached ({info['total_tokens']}/{cfg.model.max_total_tokens}).")
            return "token_limit"

        if not tool_calls:  # policy answered with plain text -> done
            logger.info("💬 FINISHED: policy replied with plain text (no tool call).")
            return "finished"

        # step 2: dispatch the tool calls
        saw_finish = False
        for idx, tool_call in enumerate(tool_calls):
            fn = tool_call.get("function", {})
            name = fn.get("name", "")
            logger.info(f"🎬 ACTION ({name}):\n{fn.get('arguments')}")
            tool_result = await toolbox.call(name, fn.get("arguments"), timeout=cfg.action_timeout)
            observation = tool_result.to_observation()
            info["num_tool_calls"] += 1
            if tool_result.status == "timeout":  # a tool hit its own timeout (e.g. shell command_timeout)
                info["timeouts"] += 1
                logger.warning(
                    f"⏳ TIMEOUT ({name}): {info['timeouts']}/{cfg.timeout_budget} budget used\n{observation}"
                )
            elif tool_result.status == "error":  # a tool raised ToolError, skipped by Toolbox.call
                info["errors"] += 1
                logger.error(f"❌ TOOL ERROR ({name}):\n{observation}")
            else:
                logger.info(f"👀 OBSERVATION ({name}):\n{observation}")

            transcript.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "name": name,
                    "content": observation,
                }
            )

            if info["timeouts"] > cfg.timeout_budget:
                logger.warning("Exit: timeout budget exhausted; skipping remaining tool calls this turn.")
                for tool_call in tool_calls[idx + 1 :]:
                    transcript.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id"),
                            "name": tool_call.get("function", {}).get("name", ""),
                            "content": "Skipped: timeout budget exhausted.",
                        }
                    )
                return "timeout_limit"
            if name in _FINISH_TOOLS:
                saw_finish = True
        if saw_finish:
            logger.info("💬 FINISHED: policy called a finish tool.")
        return "finished" if saw_finish else "completed"
