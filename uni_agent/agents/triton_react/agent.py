"""Five-tool Triton ReAct loop with compaction and segment-boundary rewards.

The loop is the stock ReAct step. ATRL adds three behaviors on top:

* nudge the policy to create an implementation, then to call ``run_verify``
* compact the transcript after a verify call once the context crosses the threshold
* record the workspace reward at each compaction boundary

The KernelBench task turns those boundary rewards plus the final reward into
the mixed segment rewards consumed by ``grpo_segment``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from pydantic import Field

from uni_agent.atrl.reward_formula import reward_breakdown
from uni_agent.compaction import CompactionConfig, ConversationCompactor
from uni_agent.tools import Toolbox

from ..base import AgentResult
from ..react.agent import ReActAgent, ReActConfig
from ..react.model import OpenAICompatibleChatModel
from ..registry import register_agent

logger = logging.getLogger(__name__)

_RESUME_NOTE = (
    "[Context was compacted. The checkpoint above is your own record of all "
    "work so far on this task. Do NOT restart the task or rewrite the "
    "implementation from scratch. Continue from the checkpoint's Current State "
    "and execute the Next Step now.\n\nNext Step from checkpoint:\n{next_step}"
)


class TritonReactConfig(ReActConfig):
    name: str = "triton_react"
    tools: list[dict] = Field(
        default_factory=lambda: [
            {"name": "str_replace_editor"},
            {"name": "list_skills"},
            {"name": "read_skill"},
            {"name": "run_verify"},
            {"name": "submit"},
        ]
    )
    workspace_dir: str = "/workspace"
    impl_nudge_margin: int = 12
    verify_nudge_margin: int = 8


def _next_step_section(checkpoint_text: str) -> str:
    marker = "## Next Step"
    index = checkpoint_text.find(marker)
    if index < 0:
        return ""
    rest = checkpoint_text[index + len(marker) :]
    nxt = rest.find("\n## ")
    return (rest if nxt < 0 else rest[:nxt]).strip()


@register_agent("triton_react")
class TritonReactAgent(ReActAgent):
    """ReAct agent for ATRL: five tools, nudges, and structured compaction."""

    config_model = TritonReactConfig

    async def run(
        self,
        *,
        sandbox,
        messages: list[dict[str, Any]],
        workdir: str | None = None,
    ) -> AgentResult:
        cfg: TritonReactConfig = self.config  # type: ignore[assignment]
        if cfg.model.base_url is None:
            raise ValueError("triton_react: config.model.base_url is not set")

        toolbox = Toolbox.from_specs(cfg.tools, sandbox=sandbox)
        model = OpenAICompatibleChatModel(
            base_url=cfg.model.base_url,
            api_key=cfg.model.api_key,
            model_name=cfg.model.model_name,
            sampling_params=cfg.sampling_params_override.sampling_params(),
            tools_schemas=toolbox.schemas(),
        )
        compactor = ConversationCompactor(CompactionConfig.from_env())
        min_remaining_turns = int(os.environ.get("TRITON_COMPACTION_MIN_REMAINING_TURNS", "3"))
        transcript: list[dict[str, Any]] = list(messages)
        info: dict[str, Any] = {
            "steps": 0,
            "num_tool_calls": 0,
            "timeouts": 0,
            "errors": 0,
            "total_tokens": 0,
            "segment_boundary_rewards": [],
        }
        has_impl = False
        has_verified = False
        termination_reason = "unknown"
        workspace = workdir or cfg.workspace_dir
        try:
            async with toolbox.entered(retry=3, timeout=60):
                for step_idx in range(1, cfg.max_steps + 1):
                    info["steps"] = step_idx
                    turns_left = cfg.max_steps - step_idx + 1
                    self._maybe_nudge(cfg, transcript, turns_left, has_impl, has_verified, workspace)
                    stop_reason = await self.step(cfg, model, toolbox, transcript, info)
                    has_impl, has_verified, ran_verify = self._observe_last_turn(transcript, has_impl, has_verified)
                    if stop_reason != "completed":
                        termination_reason = stop_reason
                        break
                    if ran_verify:
                        await self._maybe_compact(
                            cfg,
                            compactor,
                            model,
                            sandbox,
                            transcript,
                            info,
                            step_idx,
                            turns_left,
                            min_remaining_turns,
                            workspace,
                        )
                else:
                    termination_reason = "max_steps"
        except Exception as exc:
            logger.exception("triton_react failed at step %s", info["steps"])
            termination_reason = "unknown_error"
            info["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            await model.aclose()

        info["termination_reason"] = termination_reason
        info["has_impl"] = has_impl
        info["has_verified"] = has_verified
        return AgentResult(transcript=transcript, info=info, finished=termination_reason == "finished")

    @staticmethod
    def _maybe_nudge(
        cfg: TritonReactConfig,
        transcript: list[dict[str, Any]],
        turns_left: int,
        has_impl: bool,
        has_verified: bool,
        workspace: str,
    ) -> None:
        if not has_impl and turns_left <= cfg.impl_nudge_margin:
            transcript.append(
                {
                    "role": "user",
                    "content": (
                        f"IMPORTANT: {turns_left} turns left. Create the implementation with "
                        f"str_replace_editor at {workspace}/src/<op>_triton_ascend_impl.py, then call run_verify."
                    ),
                }
            )
        elif has_impl and not has_verified and turns_left <= cfg.verify_nudge_margin:
            transcript.append(
                {
                    "role": "user",
                    "content": (
                        f"IMPORTANT: code is written but not verified. Call run_verify now. "
                        f"({turns_left} turns left.)"
                    ),
                }
            )

    @staticmethod
    def _observe_last_turn(
        transcript: list[dict[str, Any]],
        has_impl: bool,
        has_verified: bool,
    ) -> tuple[bool, bool, bool]:
        ran_verify = False
        for message in reversed(transcript):
            if message.get("role") == "assistant":
                break
            if message.get("role") == "tool" and message.get("name") == "run_verify":
                has_verified = True
                ran_verify = True
            elif message.get("role") == "tool" and message.get("name") == "str_replace_editor":
                has_impl = True
        return has_impl, has_verified, ran_verify

    async def _maybe_compact(
        self,
        cfg: TritonReactConfig,
        compactor: ConversationCompactor,
        model: OpenAICompatibleChatModel,
        sandbox,
        transcript: list[dict[str, Any]],
        info: dict[str, Any],
        step_idx: int,
        turns_left: int,
        min_remaining_turns: int,
        workspace: str,
    ) -> None:
        compactor.notify_step()
        usage = (info.get("total_tokens"), None)
        if not compactor.should_compact(transcript, usage):
            return
        if turns_left < min_remaining_turns:
            logger.info("skip compaction at step %s: turns_left=%s", step_idx, turns_left)
            return
        rewritten = await compactor.compact(transcript, model, step=step_idx, last_usage=usage)
        if rewritten is None:
            return
        transcript[:] = rewritten
        checkpoint = rewritten[-1]["content"] if rewritten else ""
        transcript.append(
            {
                "role": "user",
                "content": _RESUME_NOTE.format(next_step=_next_step_section(checkpoint) or "(see Next Step)"),
            }
        )
        info["segment_boundary_rewards"].append(await _boundary_reward(sandbox, workspace))


async def _boundary_reward(sandbox, workspace: str) -> float:
    """Score the workspace at a compaction boundary. Missing artifacts score 0."""
    try:
        verify_raw = await sandbox.read_file(f"{workspace}/output/verify/verify_result.json")
        perf_raw = await sandbox.read_file(f"{workspace}/output/perf_result.json")
    except Exception:
        logger.info("segment boundary has no verify artifacts yet")
        return 0.0
    try:
        verify = json.loads(verify_raw.decode() if isinstance(verify_raw, bytes) else verify_raw)
        perf = json.loads(perf_raw.decode() if isinstance(perf_raw, bytes) else perf_raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return 0.0
    if not isinstance(verify, dict):
        return 0.0
    metrics = dict(verify)
    if isinstance(perf, dict):
        metrics["perf_data"] = perf
    return float(reward_breakdown(metrics)["total"])
