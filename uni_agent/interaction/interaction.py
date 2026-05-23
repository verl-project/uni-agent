import time

import orjson
from pydantic import BaseModel

from uni_agent.async_logging import get_logger
from uni_agent.skills.manager import SkillsManager
from uni_agent.utils import auto_await, simple_timer

from .env import ActionIncorrectSyntaxError, ActionTimeoutError, AgentEnv, TerminalNotAliveError
from .model import AgentChatModel, MaxTokenExceededError
from .tool_parser import FunctionCallFormatError
from .tool_schemas import OpenAIFunctionToolCall
from .tools_manager import ToolsManager


class StepOutput(BaseModel):
    step_idx: int

    response: str = ""
    thought: str = ""
    action: str = ""
    observation: str = ""
    execution_time: float | None = None
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
    ):
        self.env = env
        self.model = model
        self.tools_manager = tools_manager
        self.skills_manager = skills_manager
        self.messages = messages
        self.action_timeout = action_timeout
        self.timeout_budget = timeout_budget
        self.max_turns = max_turns
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
        # step index start from 1
        step_output = StepOutput(step_idx=step_idx)
        self.logger.info(f"{'=' * 25} STEP {step_idx} {'=' * 25}")

        # step 1: prepare template
        self.logger.info(f"🤖 MODEL INPUT\n{self.messages[-1]['content']}")

        # step 2: generate response and update rollout cache
        try:
            model_output, rollout_cache, generation_info = await self.model.query(
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
        self.messages.append({"role": "assistant", "content": model_output})  # tool call message
        try:
            structured_tool_calls = self.rollout_cache.get("extra_fields", {}).get("last_tool_calls", [])
            if structured_tool_calls:
                content, tool_calls = await self.tools_manager.parse_structured_action(
                    content=model_output,
                    tool_calls_data=structured_tool_calls,
                )
            else:
                content, tool_calls = await self.tools_manager.parse_action(model_output=model_output)
        except FunctionCallFormatError as e:
            user_message = {"role": "tool", "content": str(e)}
            self.messages.append(user_message)  # error message
            self.rollout_cache = await self.model.append_messages_to_rollout_cache([user_message], self.rollout_cache)
            step_output.exit_reason = "format_error"
            model_output_preview = "\n".join(model_output.splitlines()[:20])
            self.logger.error(
                f"Fail to parse thought and action from model output.\n"
                f"Error Message: {str(e)}\n"
                f"Model Output (first 20 lines): {model_output_preview}"
            )
            return step_output

        # step 4: run action in the environment
        tool_call: OpenAIFunctionToolCall = tool_calls[0]
        action_cmd = self.tools_manager.get_tool_bash_command(tool_call)
        step_output.thought = content
        step_output.action = action_cmd
        self.logger.info(f"💭 THOUGHT:\n{content}")
        self.logger.info(f"🎬 ACTION:\n{action_cmd}")
        execution_t0 = time.perf_counter()
        with simple_timer("tool_calls", self.rollout_cache["metrics"]):
            try:
                observation = await self.env.run_action(action_cmd, action_timeout=self.action_timeout)
                tool_message = {"role": "tool", "content": observation}
                self.messages.append(tool_message)  # tool response message
                self.rollout_cache = await self.model.append_messages_to_rollout_cache(
                    [tool_message], self.rollout_cache
                )
                step_output.observation = observation
            except ActionTimeoutError as e:
                self.logger.error(str(e))
                user_message = {"role": "tool", "content": str(e)}
                self.messages.append(user_message)
                self.rollout_cache = await self.model.append_messages_to_rollout_cache(
                    [user_message], self.rollout_cache
                )
                step_output.exit_reason = "timeout_error"
                self.logger.info(f"Existing timeout budget: {self.timeout_budget}")
                if self.timeout_budget > 0:
                    self.timeout_budget -= 1
                    return step_output
                else:
                    step_output.done = True
                    return step_output
            except ActionIncorrectSyntaxError as e:
                self.logger.error(str(e))
                user_message = {"role": "tool", "content": str(e)}
                self.messages.append(user_message)
                self.rollout_cache = await self.model.append_messages_to_rollout_cache(
                    [user_message], self.rollout_cache
                )
                step_output.exit_reason = "syntax_error"
                return step_output
            except TerminalNotAliveError as e:
                self.logger.error(str(e))
                user_message = {"role": "tool", "content": str(e)}
                self.messages.append(user_message)
                self.rollout_cache = await self.model.append_messages_to_rollout_cache(
                    [user_message], self.rollout_cache
                )
                step_output.exit_reason = "terminal_not_alive"
                step_output.done = True
                return step_output

        # step 5: finalize step output
        execution_time = time.perf_counter() - execution_t0
        step_output.execution_time = execution_time
        if tool_call.function.name in ["finish", "submit"]:
            step_output.done = True
            step_output.exit_reason = "finished"
        else:
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
