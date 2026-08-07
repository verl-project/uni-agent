"""Self-contained MemAgent implementation with explicit context management."""

from __future__ import annotations

import copy
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator

from pydantic import BaseModel, Field

from uni_agent.agents.react.model import OpenAICompatibleChatModel
from uni_agent.agents.registry import register_agent

from ..base import Agent, AgentConfig, AgentResult

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

TEMPLATE = (
    "You are presented with a problem, a section of an article that may contain the answer to the problem, and a "
    "previous memory. Please read the provided section carefully and update the memory with the new information that "
    "helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any "
    "new, useful information.\n\n"
    "<problem>\n{prompt}\n</problem>\n\n"
    "<memory>\n{memory}\n</memory>\n\n"
    "<section>\n{chunk}\n</section>\n\n"
    "Updated memory:\n"
)

TEMPLATE_FINAL_BOXED = (
    "You are presented with a problem and a previous memory. Please answer the problem based on the previous memory "
    "and put the answer in \\boxed{{}}.\n\n"
    "<problem>\n{prompt}\n</problem>\n\n"
    "<memory>\n{memory}\n</memory>\n\n"
    "Your answer:\n"
)


class ContextTurnOutput(BaseModel):
    """One model turn inside a MemAgent context segment."""

    step_idx: int
    response: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ContextStepOutput(BaseModel):
    """One independently materialized MemAgent context segment."""

    prompt_messages: list[dict[str, Any]] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[ContextTurnOutput] = Field(default_factory=list)
    reward: float = 0.0
    execution_time: float = 0.0

    def set_prompt_messages(self, messages: list[dict[str, Any]]) -> None:
        self.prompt_messages = copy.deepcopy(messages)

    def set_messages(self, messages: list[dict[str, Any]]) -> None:
        self.messages = copy.deepcopy(messages)

    def set_execution_time(self, execution_time: float) -> None:
        self.execution_time = execution_time

    def add_step(self, step_output: ContextTurnOutput) -> None:
        self.steps.append(step_output)

    def set_reward(self, reward: float) -> None:
        self.reward = reward

    def get_reward(self) -> float:
        return self.reward


class ContextManagerResult(BaseModel):
    """Aggregate result of a context-managed MemAgent execution."""

    run_id: str
    execution_time: float = 0.0
    trajectory: list[ContextStepOutput] = Field(default_factory=list)
    final_state: ContextStepOutput = Field(default_factory=ContextStepOutput)
    total_steps: int = 0

    def set_reward(self, reward: float) -> None:
        """Assign the final task reward to every context segment."""

        for step in self.trajectory:
            step.set_reward(reward)


class MemAgentConfig(AgentConfig):
    """Configuration for chunked-context memory updates."""

    name: str = "mem_agent"
    max_steps: int = Field(default=50, gt=0, description="Maximum model calls across all context segments.")
    max_memorization_length: int = Field(default=1024, gt=0)
    max_chunks: int = Field(default=8, gt=0)
    max_final_response_length: int = Field(default=1024, gt=0)


@register_agent("mem_agent")
class MemAgent(Agent):
    """Read long input in chunks and carry only a compact memory between contexts."""

    config_model = MemAgentConfig

    def __init__(self, config: MemAgentConfig | None = None) -> None:
        super().__init__(config)
        self._model: OpenAICompatibleChatModel | None = None
        self._trajectory: list[ContextStepOutput] = []
        self._current_context_step: ContextStepOutput | None = None
        self._global_step_idx = 0
        self._step_idx = 0
        self._total_completion_tokens = 0
        self._interaction_start = 0.0
        self._context_session_active = False
        self._context_manager_result: ContextManagerResult | None = None
        self.messages: list[dict[str, Any]] = []

    @asynccontextmanager
    async def context_session(self) -> AsyncIterator[None]:
        """Initialize and close the model runtime used by MemAgent."""

        if self._context_session_active:
            raise RuntimeError("MemAgent does not support nested context_session() calls")

        cfg = self._mem_agent_config
        if cfg.model.base_url is None:
            raise ValueError(f"{cfg.name}: config.model.base_url is not set (the endpoint the policy calls)")

        self._trajectory = []
        self._current_context_step = None
        self._global_step_idx = 0
        self._step_idx = 0
        self._total_completion_tokens = 0
        self._context_manager_result = None
        self.messages = []

        self._model = OpenAICompatibleChatModel(
            base_url=cfg.model.base_url,
            api_key=cfg.model.api_key,
            model_name=cfg.model.model_name,
            sampling_params=self._default_sampling_params(),
        )

        started_at = time.perf_counter()
        self._context_session_active = True
        try:
            yield
        finally:
            self._collect_context_step()
            try:
                await self._model.aclose()
            finally:
                self._context_session_active = False
                self._context_manager_result = ContextManagerResult(
                    run_id=str(uuid.uuid4()),
                    execution_time=time.perf_counter() - started_at,
                    trajectory=self._trajectory,
                    final_state=self._trajectory[-1] if self._trajectory else ContextStepOutput(),
                    total_steps=self._global_step_idx,
                )

    def build_agent_result(self) -> AgentResult:
        """Convert the completed MemAgent context session into an Agent result."""

        if self._context_session_active:
            raise RuntimeError("build_agent_result() must be called after context_session() exits")
        if self._context_manager_result is None:
            raise RuntimeError("No completed context session; call context_session() from run() first")

        result = self._context_manager_result
        final_response = result.final_state.steps[-1].response if result.final_state.steps else ""
        transcript = [message for segment in result.trajectory for message in segment.messages]
        return AgentResult(
            output={"response": final_response, "context_manager_result": result},
            transcript=transcript,
            info={
                "execution_time": result.execution_time,
                "total_steps": result.total_steps,
                "num_contexts": len(result.trajectory),
            },
        )

    async def update_context(self, messages: list[dict[str, Any]]) -> None:
        """Finalize the current segment and continue from a newly built context."""

        if not self._context_session_active:
            raise RuntimeError("update_context() must be called inside context_session()")
        self._collect_context_step()
        self._track_context_step()
        assert self._current_context_step is not None
        self._current_context_step.set_prompt_messages(messages)
        self.messages = copy.deepcopy(messages)
        for message in self.messages:
            logger.info("%s PROMPT:\n%s", str(message.get("role", "")).upper(), message.get("content", ""))

    async def step(self, sampling_params: dict[str, Any] | None = None) -> ContextTurnOutput:
        """Run one model call in the active context segment."""

        if self._current_context_step is None:
            raise RuntimeError("Please call update_context() before calling the first step()")
        if self._model is None:
            raise RuntimeError("MemAgent runtime is not initialized; call run() through a Task")

        cfg = self._mem_agent_config
        if self._global_step_idx >= cfg.max_steps:
            raise RuntimeError(f"MemAgent reached max_steps={cfg.max_steps}")

        self._global_step_idx += 1
        self._step_idx += 1
        params = self._sampling_params_for_step(sampling_params)
        content, _, generation_info = await self._model.query(
            self.messages,
            sampling_params=params,
        )
        self._total_completion_tokens += generation_info["completion_tokens"]

        self.messages.append({"role": "assistant", "content": content})

        output = ContextTurnOutput(
            step_idx=self._step_idx,
            response=content,
            prompt_tokens=generation_info["prompt_tokens"],
            completion_tokens=generation_info["completion_tokens"],
        )

        self._current_context_step.add_step(output)
        return output

    def get_global_step_idx(self) -> int:
        """Return the number of model calls across all context segments."""

        return self._global_step_idx

    def get_current_step_idx(self) -> int:
        """Return the number of model calls in the current context segment."""

        return self._step_idx

    def get_current_context_step(self) -> ContextStepOutput:
        """Return the segment currently being constructed."""

        if self._current_context_step is None:
            raise RuntimeError("No context is active; call update_context() first")
        return self._current_context_step

    @property
    def _mem_agent_config(self) -> MemAgentConfig:
        config = self.config
        if not isinstance(config, MemAgentConfig):
            raise TypeError("MemAgent requires MemAgentConfig")
        return config

    def _track_context_step(self) -> None:
        self._current_context_step = ContextStepOutput()
        self._step_idx = 0
        self._interaction_start = time.perf_counter()

    def _collect_context_step(self) -> None:
        if self._current_context_step is None or not self._current_context_step.steps:
            return
        self._current_context_step.set_messages(self.messages)
        self._current_context_step.set_execution_time(time.perf_counter() - self._interaction_start)
        self._trajectory.append(self._current_context_step)
        self._current_context_step = None

    def _default_sampling_params(self) -> dict[str, Any]:
        return self._mem_agent_config.model.sampling_params()

    def _sampling_params_for_step(self, overrides: dict[str, Any] | None) -> dict[str, Any]:
        cfg = self._mem_agent_config
        params = self._default_sampling_params()
        params.update(overrides or {})

        max_tokens = params.get("max_tokens", cfg.model.max_tokens_per_turn)
        if cfg.model.max_total_tokens is not None:
            remaining = cfg.model.max_total_tokens - self._total_completion_tokens
            if remaining <= 0:
                raise RuntimeError(f"MemAgent reached max_total_tokens={cfg.model.max_total_tokens}")
            max_tokens = min(max_tokens, remaining) if max_tokens is not None else remaining
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        return params

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
        raw_data: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Run the MemAgent policy using explicit context-management calls."""

        cfg: MemAgentConfig = self.config  # type: ignore[assignment]
        if not messages:
            raise ValueError("mem_agent requires a non-empty prompt message list")
        prompt = str(messages[0].get("content", ""))
        chunks = (raw_data or {}).get("chunks")
        if not isinstance(chunks, list) or not all(isinstance(chunk, str) for chunk in chunks):
            raise ValueError("mem_agent requires pre-split text chunks in raw_data['chunks']")

        async with self.context_session():
            memory: str | None = None
            for chunk in chunks:
                if self.get_global_step_idx() >= cfg.max_chunks:
                    break
                conversation = [
                    {
                        "role": "user",
                        "content": TEMPLATE.format(
                            prompt=prompt,
                            memory=memory if memory else "No previous memory",
                            chunk=chunk,
                        ),
                    }
                ]
                await self.update_context(conversation)
                step_output = await self.step(sampling_params={"max_tokens": cfg.max_memorization_length})
                memory = step_output.response

            conversation = [
                {
                    "role": "user",
                    "content": TEMPLATE_FINAL_BOXED.format(
                        prompt=prompt,
                        memory=memory if memory else "No previous memory",
                    ),
                }
            ]
            await self.update_context(conversation)
            await self.step(sampling_params={"max_tokens": cfg.max_final_response_length})

        return self.build_agent_result()
