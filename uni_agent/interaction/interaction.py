import time
from typing import Literal

import orjson
from pydantic import BaseModel, Field

from uni_agent.async_logging import get_logger
from uni_agent.skills.manager import SkillsManager
from uni_agent.utils import auto_await, simple_timer

from .env import ActionIncorrectSyntaxError, ActionTimeoutError, AgentEnv, TerminalNotAliveError
from .model import AgentChatModel, MaxTokenExceededError
from .tool_parser import FunctionCallFormatError
from .tool_schemas import OpenAIFunctionToolCall
from .tools_manager import ToolsManager

ToolStatus = Literal["ok", "timeout", "syntax_error", "skipped"]


class ToolResult(BaseModel):
    """Per-tool-call result inside a single step. ``observation`` is the
    string sent back to the model as the ``role="tool"`` content (error
    text included).
    """

    tool_call_id: str
    name: str
    action: str = ""
    observation: str = ""
    status: ToolStatus
    execution_time: float | None = None


class StepOutput(BaseModel):
    step_idx: int

    response: str = ""
    thought: str = ""
    tool_results: list[ToolResult] = Field(default_factory=list)
    done: bool = False
    exit_reason: str = ""


def fast_deepcopy(obj):
    return orjson.loads(orjson.dumps(obj))


class AgentInteraction:
    def __init__(
        self,
        run_id: str,
        env: AgentEnv,
        model: AgentChatModel,
        tools_manager: ToolsManager,
        messages: list[dict[str, str]],
        action_timeout: int = 60,
        timeout_budget: int = 3,
        max_turns: int = 50,
        skills_manager: SkillsManager | None = None,
        chat_mode: bool = False,
    ):
        """:param chat_mode: how to treat an assistant message with no
        tool calls. ``False`` (default, training / code-eval) raises
        ``format_error`` so the loop continues. ``True`` (long-running
        chat) marks the step ``turn_done`` so the caller can wait for
        the next user message.
        """
        self.env = env
        self.model = model
        self.tools_manager = tools_manager
        self.skills_manager = skills_manager
        self.messages = messages
        self.action_timeout = action_timeout
        self.timeout_budget = timeout_budget
        self.max_turns = max_turns
        self.chat_mode = chat_mode
        self.logger = get_logger("interaction", run_id)

    def inject_skills_manifest(self) -> None:
        """Append the skills manifest to the first system message.

        The manifest lists each discovered skill (name + description +
        path to its SKILL.md) so the model knows what is available and
        how to load it on demand. Skill *bodies* are not in the prompt --
        they live as real files on disk (read lazily, progressive
        disclosure).

        Call this exactly once, after ``AgentEnv.install_skills`` has
        populated ``runtime_paths``. The method is **not** idempotent --
        calling it twice will append the manifest twice. The single
        in-tree caller (``UniAgentLoop.run``) already enforces this.
        """
        if self.skills_manager is None:
            return
        manifest = self.skills_manager.build_manifest()
        if not manifest:
            return

        block = "\n\n" + manifest
        for msg in self.messages:
            if msg.get("role") == "system":
                content = msg.get("content") or ""
                msg["content"] = content + block
                return
        self.messages.insert(0, {"role": "system", "content": manifest})

    async def step(self, step_idx: int):
        """Run one model-call + tool-execution cycle.

        Outcome is reported at two levels:

        * **Tool**: per-call :class:`ToolResult` on ``step_output.tool_results``
          with ``status`` in ``{ok, timeout, syntax_error, skipped}``.
        * **Step**: ``step_output.exit_reason`` + ``done``:

          - terminal (``done=True``): ``finished``, ``turn_done``,
            ``token_limit``, ``terminal_dead``, ``timeout_budget_exhausted``.
          - non-terminal (``done=False``): ``completed``,
            ``completed_with_tool_errors``, ``format_error``.
          - set by :meth:`run`: ``max_step_limit``, ``unknown_error``.

        ``turn_done`` is gated on ``self.chat_mode`` (see ``__init__``).
        """
        # step index start from 1
        step_output = StepOutput(step_idx=step_idx)
        self.logger.info(f"{'=' * 25} STEP {step_idx} {'=' * 25}")

        # step 1: prepare template
        self.logger.info(f"🤖 MODEL INPUT\n{self.messages[-1]['content']}")

        # step 2: generate response and update rollout cache
        try:
            model_output, tool_calls, rollout_cache, generation_info = await self.model.query(
                messages=self.messages,
                rollout_cache=self.rollout_cache,
            )
            step_output.response = model_output
            self.logger.info(
                f"Prompt Tokens: {generation_info['prompt_tokens']}, "
                f"Completion Tokens: {generation_info['completion_tokens']}"
            )
            self.logger.debug(f"Model Output:\n{model_output}")
        except MaxTokenExceededError as e:
            self.logger.error(str(e))
            step_output.exit_reason = "token_limit"
            step_output.done = True
            return step_output

        # step 3: parse model response to actions
        self.rollout_cache = rollout_cache

        # Persist the assistant message in api-shape (with tool_calls)
        # so replay preserves the assistant<->tool linkage.
        assistant_msg: dict[str, object] = {"role": "assistant", "content": model_output}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        self.messages.append(assistant_msg)

        try:
            if tool_calls:
                content, tool_calls = await self.tools_manager.parse_structured_action(
                    content=model_output,
                    tool_calls_data=tool_calls,
                )
            else:
                content, tool_calls = await self.tools_manager.parse_action(model_output=model_output)
            if not tool_calls and not self.chat_mode:
                raise FunctionCallFormatError("No function call found in the response.")
        except FunctionCallFormatError as e:
            if tool_calls:
                error_msgs: list[dict[str, object]] = [
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": tc["function"]["name"],
                        "content": str(e),
                    }
                    for tc in tool_calls
                ]
            else:
                error_msgs = [{"role": "tool", "content": str(e)}]
            self.messages.extend(error_msgs)
            self.rollout_cache = await self.model.append_messages_to_rollout_cache(error_msgs, self.rollout_cache)
            step_output.exit_reason = "format_error"
            model_output_preview = "\n".join(model_output.splitlines()[:20])
            self.logger.error(
                f"Fail to parse thought and action from model output.\n"
                f"Error Message: {str(e)}\n"
                f"Model Output (first 20 lines): {model_output_preview}"
            )
            return step_output

        step_output.thought = content

        # step 4: chat_mode-only end-of-turn (single-shot already raised above).
        if not tool_calls:
            step_output.done = True
            step_output.exit_reason = "turn_done"
            self.logger.info(f"💬 TURN DONE (no tool call): {model_output}")
            return step_output

        # step 5: run each tool call sequentially in the shared bash session
        tool_results: list[ToolResult] = []
        tool_messages: list[dict[str, object]] = []
        saw_finish = False
        terminal_dead = False

        with simple_timer("tool_calls", self.rollout_cache["metrics"]):
            for idx, tool_call in enumerate(tool_calls):
                tool_call: OpenAIFunctionToolCall  # type: ignore[no-redef]
                action_cmd = self.tools_manager.get_tool_bash_command(tool_call)
                self.logger.info(f"🎬 ACTION ({tool_call.function.name}):\n{action_cmd}")

                tool_t0 = time.perf_counter()
                status: ToolStatus
                try:
                    observation = await self.env.run_action(action_cmd, action_timeout=self.action_timeout)
                    status = "ok"
                    if tool_call.function.name in ("finish", "submit"):
                        saw_finish = True
                except ActionTimeoutError as e:
                    observation = str(e)
                    status = "timeout"
                    self.timeout_budget -= 1
                    self.logger.error(f"{observation} (timeout_budget left: {self.timeout_budget})")
                except ActionIncorrectSyntaxError as e:
                    observation = str(e)
                    status = "syntax_error"
                    self.logger.error(observation)
                except TerminalNotAliveError as e:
                    observation = str(e)
                    status = "skipped"
                    terminal_dead = True
                    self.logger.error(observation)
                elapsed = time.perf_counter() - tool_t0

                tool_results.append(
                    ToolResult(
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name,
                        action=action_cmd,
                        observation=observation,
                        status=status,
                        execution_time=elapsed,
                    )
                )

                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": observation,
                    }
                )

                # On hard failure (dead session / budget out), synthesize
                # `skipped` results for remaining tool calls to keep the
                # assistant<->tool N:N invariant, then break.
                budget_exhausted = self.timeout_budget < 0
                if terminal_dead or budget_exhausted:
                    skipped_reason = (
                        "Skipped: the bash session died mid-step; no further tool calls ran."
                        if terminal_dead
                        else "Skipped: timeout budget exhausted mid-step; no further tool calls ran."
                    )
                    for remaining in tool_calls[idx + 1 :]:
                        tool_results.append(
                            ToolResult(
                                tool_call_id=remaining.id,
                                name=remaining.function.name,
                                action="",
                                observation=skipped_reason,
                                status="skipped",
                                execution_time=None,
                            )
                        )
                        tool_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": remaining.id,
                                "name": remaining.function.name,
                                "content": skipped_reason,
                            }
                        )
                    break

        # step 6: commit collected tool messages
        self.messages.extend(tool_messages)
        self.rollout_cache = await self.model.append_messages_to_rollout_cache(tool_messages, self.rollout_cache)
        step_output.tool_results = tool_results

        # step 7: step-level exit_reason (precedence: terminal_dead >
        # timeout_budget_exhausted > finished > completed_with_tool_errors > completed)
        if terminal_dead:
            step_output.done = True
            step_output.exit_reason = "terminal_dead"
            return step_output
        if self.timeout_budget < 0:
            step_output.done = True
            step_output.exit_reason = "timeout_budget_exhausted"
            self.logger.info("Exit step: timeout budget exhausted.")
            return step_output
        if saw_finish:
            step_output.done = True
            step_output.exit_reason = "finished"
            return step_output
        if any(tr.status in ("timeout", "syntax_error") for tr in tool_results):
            step_output.done = False
            step_output.exit_reason = "completed_with_tool_errors"
            return step_output
        step_output.done = False
        step_output.exit_reason = "completed"
        return step_output

    @auto_await
    async def run(self):
        self.trajectory: list[StepOutput] = []

        self.logger.info("Inital Prompt:")
        for message in self.messages:
            self.logger.info(f"{message['role'].upper()} PROMPT:\n{message['content']}")

        rollout_cache = await self.model.prepare_rollout_cache(self.messages)
        self.rollout_cache: dict[str, str] = rollout_cache

        done = False
        step_idx = 0
        execution_time = time.perf_counter()
        while not done:
            # we start from 1
            step_idx += 1
            try:
                step_output = await self.step(step_idx=step_idx)
                self.trajectory.append(step_output)
                done = step_output.done
                if step_idx >= self.max_turns:
                    self.logger.error(f"Exit due to max step limit: {self.max_turns}")
                    step_output = StepOutput(step_idx=step_idx, exit_reason="max_step_limit")
                    self.trajectory.append(step_output)
                    break
            except Exception as e:
                # this should not happen, if it happens, we should fix the code
                self.logger.critical(f"Exit due to unknown error: {str(e)}")
                step_output = StepOutput(step_idx=step_idx, exit_reason="unknown_error")
                self.trajectory.append(step_output)
                break

        execution_time = time.perf_counter() - execution_time
        result = {
            "trajectory": self.trajectory,
            "rollout_cache": self.rollout_cache,
            "execution_time": execution_time,
            "messages": self.messages,
        }
        return result
