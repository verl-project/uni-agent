# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GUI Agent Loop for VLM GUI Agent Training.

Sandbox as environment: model outputs raw COT + actions -> sandbox executes -> returns screenshot.
Loop continues until DONE/FAIL/max_steps.
"""

import asyncio
import copy
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import numpy as np
from oagi.utils.output_parser import parse_raw_output
from PIL import Image

from uni_agent.gui_utils import PyautoguiActionConvertor, apply_sliding_window_to_images
from uni_agent.tools.os_sandbox_tool import (
    DummySandboxTool,
    OSSandboxTool,
    SystemUnavailableError,
    TaskUnrecoverableError,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Fixed token length for environment response (image-only message)
ENV_RESPONSE_TOKEN_LEN = 1135
VISUAL_WINDOW_SIZE = 3


class TruncatedError(Exception):
    """Exception raised when sequence is truncated due to length limit.

    This error should NOT trigger retry. The sample will be discarded (remove_sample=True)
    since truncation occurs before sandbox verification and final_score would be meaningless.
    """

    def __init__(self, message: str, agent_data: "GUIAgentData"):
        super().__init__(message)
        self.agent_data = agent_data


class GUIAgentState(Enum):
    """States for GUI Agent loop."""

    INITIALIZING = "initializing"  # Creating sandbox, getting initial screenshot
    GENERATING = "generating"  # Model generating action
    EXECUTING = "executing"  # Executing action in sandbox
    TERMINATED = "terminated"  # Done or max steps reached


@dataclass
class GUIAgentData:
    """State container for GUI Agent loop."""

    # Core state
    image_data: list[Any] = field(default_factory=list)
    """Screenshots (PIL Images) accumulated during the episode."""

    image_uuids: list[str] = field(default_factory=list)
    """UUIDs for vLLM image caching."""

    image_urls: list[str] = field(default_factory=list)
    """Screenshot URLs for lazy loading (skips Ray transfer)."""

    def add_image(self, image: Any, image_url: Optional[str] = None) -> None:
        """Add an image with auto-generated UUID."""
        self.image_data.append(image)
        self.image_uuids.append(uuid4().hex)
        if image_url is not None:
            assert image_url, f"image_url must be non-empty when provided, got {image_url!r}"
            self.image_urls.append(image_url)

    request_id: str = ""
    """Unique ID for this episode (used for both sandbox and vLLM requests)."""

    task_config: dict = field(default_factory=dict)
    """Task configuration from dataset."""

    # Generation state
    prompt_ids: list[int] = field(default_factory=list)
    """Current prompt token IDs."""

    response_mask: list[int] = field(default_factory=list)
    """Response mask (1=model generated, 0=environment)."""

    response_logprobs: list[float] = field(default_factory=list)
    """Log probabilities."""

    routed_experts: list = field(default_factory=list)
    """Routed experts indices for MoE models. Each element is a (layer_num, topk_num) array."""

    # Turn state
    last_turn_output: str = ""
    """Model output from last generation (to be executed in sandbox)."""

    # Counters
    step_count: int = 0
    """Number of actions executed."""

    generation_token_counts: list[int] = field(default_factory=list)
    """Token counts for each generation (for per-generation statistics)."""

    # Heterogeneous context support (TITOUT)
    env_token_ranges: list[tuple[int, int]] = field(default_factory=list)
    """Image token ranges (start, end) for sliding window. Index 0 = initial image."""

    turn_metadata: list[dict] = field(default_factory=list)
    """Per-turn metadata for trajectory splitting (response_start, len, visible_images)."""

    # Results
    done: bool = False
    """Whether episode is complete."""

    final_score: Optional[float] = None
    """Final score from sandbox."""

    verification_logs: Optional[str] = None
    """Verification process logs."""

    # Truncation and removal flags
    truncated: bool = False
    """Whether the sequence was truncated due to length limit."""

    remove_sample: bool = False
    """Whether to exclude from loss (loss_mask=0)."""

    # Per-task configuration
    max_steps: Optional[int] = None
    """Per-task max_steps from dataset, or None to use global default."""

    # Metrics
    metrics: dict[str, Any] = field(default_factory=dict)
    """Performance metrics."""


@register("gui_agent")
class GUIAgentLoop(AgentLoopBase):
    """Agent loop for GUI Agent VLM training.

    Treats sandbox as RL environment: generate action -> execute -> get screenshot -> repeat.
    """

    _class_initialized: bool = False
    _sandbox_tool: Optional[OSSandboxTool] = None
    _action_convertor: Optional[PyautoguiActionConvertor] = None

    def __init__(self, trainer_config, server_manager, tokenizer, processor, dataset_cls, dataset_config, **kwargs):
        super().__init__(trainer_config, server_manager, tokenizer, processor, dataset_cls, dataset_config, **kwargs)
        self.init_class(trainer_config.config, tokenizer, processor, **kwargs)

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True

        cls.tokenizer = tokenizer
        cls.processor = processor

        multi_turn_config = config.actor_rollout_ref.rollout.multi_turn
        cls.max_steps = multi_turn_config.max_assistant_turns
        cls.trajectory_splitting = getattr(multi_turn_config, "trajectory_splitting", False)
        cls.dummy_image_width = 1260
        cls.dummy_image_height = 700
        # FlexAttention heterogeneous context mode: keep all tokens but control visibility via attention mask
        cls.use_flex_attention_hetero = getattr(multi_turn_config, "use_flex_attention_hetero", False)
        assert cls.max_steps is not None, "max_assistant_turns must be set in multi_turn config"

        # Parse dummy_mode with backward compatibility for bool type
        raw_dummy_mode = getattr(multi_turn_config, "dummy_mode", "disabled")
        if isinstance(raw_dummy_mode, bool):
            cls.dummy_mode = "full" if raw_dummy_mode else "disabled"
            logger.warning(f"dummy_mode={raw_dummy_mode} (bool) is deprecated. Use dummy_mode='{cls.dummy_mode}'")
        else:
            cls.dummy_mode = raw_dummy_mode

        # Initialize sandbox tool based on dummy_mode
        if cls.dummy_mode in ("sandbox", "full"):
            cls._sandbox_tool = DummySandboxTool(max_steps=cls.max_steps)
            cls.use_url_mode = False
            logger.info(f"Initialized GUIAgentLoop in {cls.dummy_mode.upper()} DUMMY MODE: max_steps={cls.max_steps}")
        else:
            sandbox_url = os.environ.get("SANDBOX_URL", "http://localhost:9000")
            cls.use_url_mode = os.environ.get("SANDBOX_IMAGE_URL_MODE", "true").lower() == "true"
            cls._sandbox_tool = OSSandboxTool(
                config={
                    "max_steps": cls.max_steps,
                    "sandbox_url": sandbox_url,
                }
            )
            logger.info(f"Initialized GUIAgentLoop: max_steps={cls.max_steps}, sandbox_url={sandbox_url}")

        cls._action_convertor = PyautoguiActionConvertor(logger=logger)

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.calculate_log_probs = getattr(config.actor_rollout_ref.rollout, "calculate_log_probs", False)

        # Base conversation for consistent tokenization (trim prefix when tokenizing new messages)
        cls._base_chat_history = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I am a user."},
        ]
        base_dummy_prompt = cls.processor.apply_chat_template(
            cls._base_chat_history,
            add_generation_prompt=False,
            tokenize=False,
            **cls.apply_chat_template_kwargs,
        )
        cls._base_trim_length = len(cls.tokenizer.encode(base_dummy_prompt, add_special_tokens=False))

    # Retry configuration
    MAX_RETRIES_VALIDATE = 6
    MAX_RETRIES_TRAIN = 3
    RETRY_DELAY = 2.0  # seconds
    MAX_GENERATION_RETRIES = 2  # Max retries for format validation during generation
    DUMMY_RESPONSE_TOKENS = 128  # Fixed token length for full dummy mode

    def _generate_dummy_tokens(self, step_idx: int) -> TokenOutput:
        """Generate deterministic dummy tokens for full dummy mode."""
        # Use fixed token pattern: [pad_id] * 128 (deterministic, no tokenization needed)
        pad_id = self.tokenizer.pad_token_id or 0
        dummy_ids = [pad_id] * self.DUMMY_RESPONSE_TOKENS

        return TokenOutput(
            token_ids=dummy_ids,
            log_probs=[0.0] * len(dummy_ids) if self.calculate_log_probs else None,
            routed_experts=None,
            stop_reason="completed",
        )

    def _extract_task_config(self, kwargs: dict[str, Any]) -> tuple[dict, str, list]:
        """Extract task_config, task_id, raw_prompt from kwargs."""
        task_config = kwargs.get("gui_agent_kwargs", {}).get("task_config")
        if task_config is None:
            raise ValueError("task_config is required in gui_agent_kwargs.")
        task_id = task_config.get("id", "unknown")
        raw_prompt = kwargs["raw_prompt"]
        return task_config, task_id, raw_prompt

    def _create_agent_data(
        self,
        task_id: str,
        task_config: dict,
        retry_count: int = 0,
    ) -> GUIAgentData:
        """Create a fresh GUIAgentData instance."""
        request_id = f"{task_id}-{uuid4().hex[:8]}"
        agent_data = GUIAgentData(
            request_id=request_id,
            task_config=task_config,
        )
        agent_data.metrics["retry_count"] = retry_count
        return agent_data

    @rollout_trace_op
    async def run(
        self, sampling_params: dict[str, Any], prefetched_agent_data: Optional["GUIAgentData"] = None, **kwargs
    ) -> AgentLoopOutput:
        """Run GUI Agent loop with retry support for sandbox failures."""
        if self._sandbox_tool is None:
            raise RuntimeError("GUIAgentLoop._sandbox_tool is not initialized.")

        sampling_params_template = self._clone_sampling_params(sampling_params)
        early_stop_event = kwargs.pop("early_stop_event", None)
        task_config, task_id, raw_prompt = self._extract_task_config(kwargs)
        is_validate = kwargs.get("validate", False)
        max_retries = self.MAX_RETRIES_VALIDATE if is_validate else self.MAX_RETRIES_TRAIN

        last_error: Optional[Exception] = None
        retry_count = 0
        retry_details = []
        _early_stop_step = None  # tracks step count when early stopped mid-execution
        for attempt in range(max_retries):
            # Skip new retry attempts if early stop triggered
            if early_stop_event is not None and early_stop_event.is_set():
                break

            attempt_start = time.time()
            try:
                attempt_sampling_params = self._clone_sampling_params(sampling_params_template)
                use_prefetch = prefetched_agent_data is not None and attempt == 0

                if use_prefetch:
                    agent_data = prefetched_agent_data
                    request_id = agent_data.request_id
                    state = GUIAgentState.GENERATING
                else:
                    agent_data = self._create_agent_data(task_id, task_config, retry_count)
                    request_id = agent_data.request_id
                    state = GUIAgentState.INITIALIZING

                while state != GUIAgentState.TERMINATED:
                    # Early stop check at step boundary
                    if early_stop_event is not None and early_stop_event.is_set():
                        if request_id and not request_id.endswith("-failed"):
                            await self._release_sandbox_safe(request_id, task_id, is_validate, reason="early_stopped")
                        _early_stop_step = agent_data.step_count
                        break  # exit while → check below exits for loop → falls to failed path

                    if state == GUIAgentState.INITIALIZING:
                        state = await self._handle_initializing(agent_data, is_validate, raw_prompt)
                    elif state == GUIAgentState.GENERATING:
                        state = await self._handle_generating(agent_data, attempt_sampling_params)
                    elif state == GUIAgentState.EXECUTING:
                        state = await self._handle_executing(agent_data)
                    else:
                        state = GUIAgentState.TERMINATED

                # Early stop broke while loop — exit retry loop, fall to failed path
                if _early_stop_step is not None:
                    break

                retry_details.append(
                    {
                        "attempt": attempt + 1,
                        "duration": round(time.time() - attempt_start, 2),
                        "success": True,
                    }
                )
                agent_data.metrics["retry_details"] = retry_details
                return self._build_output(agent_data)

            except TruncatedError as e:
                # Truncation doesn't trigger sandbox verification, so final_score=0.0
                # is a hardcoded placeholder, not a real score. Fall through to
                # _build_failed_agent_data (remove_sample=True, no image_urls).
                retry_details.append(
                    {
                        "attempt": attempt + 1,
                        "duration": round(time.time() - attempt_start, 2),
                        "success": False,
                        "truncated": True,
                    }
                )
                last_error = e
                logger.warning(f"[{task_id}] Sequence truncated: {e}")
                await self._release_sandbox_safe(e.agent_data.request_id, task_id, is_validate, reason="truncated")
                break

            except (TaskUnrecoverableError, SystemUnavailableError) as e:
                retry_details.append(
                    {
                        "attempt": attempt + 1,
                        "duration": round(time.time() - attempt_start, 2),
                        "success": False,
                        "error": str(e),
                    }
                )
                last_error = e
                retry_count += 1
                logger.warning(f"[{task_id}] Attempt {attempt + 1}/{max_retries} failed: {e}")
                await self._release_sandbox_safe(request_id, task_id, is_validate, reason="retry_failed")

                if attempt < max_retries - 1:
                    await asyncio.sleep(getattr(e, "retry_after", self.RETRY_DELAY))

        # Failed, truncated, or early-stopped - return remove_sample=True output
        is_early_stopped = early_stop_event is not None and early_stop_event.is_set()
        is_truncated = isinstance(last_error, TruncatedError)
        if not is_early_stopped:
            if is_truncated:
                logger.warning(f"[{task_id}] Sequence truncated, returning remove_sample=True. Error: {last_error}")
            else:
                logger.error(
                    f"[{task_id}] All {max_retries} retry attempts failed. Last error: {last_error}. "
                    f"Returning remove_sample=True to avoid system crash."
                )

        failed_data = await self._build_failed_agent_data(
            task_id=task_id,
            raw_prompt=raw_prompt,
            task_config=task_config,
            retry_count=retry_count,
            last_error=last_error,
            max_retries=max_retries,
        )
        failed_data.truncated = is_truncated
        if is_early_stopped:
            failed_data.metrics["failure_reason"] = "early_stopped"
            if _early_stop_step is not None:
                failed_data.metrics["early_stopped_at_step"] = _early_stop_step
        failed_data.metrics["retry_details"] = retry_details
        return self._build_output(failed_data)

    async def _handle_initializing(
        self,
        agent_data: GUIAgentData,
        is_validate: bool,
        raw_prompt: list[dict[str, Any]],
    ) -> GUIAgentState:
        """Initialize sandbox and get initial screenshot."""
        # Get per-task max_steps from task_config, fall back to class-level default
        task_max_steps = agent_data.task_config.get("max_steps") or self.max_steps
        agent_data.max_steps = task_max_steps

        with simple_timer("sandbox_create", agent_data.metrics):
            request_id, response = await self._sandbox_tool.create(
                instance_id=agent_data.request_id,
                create_kwargs={
                    "task_config": agent_data.task_config,
                    "max_steps": task_max_steps,
                    "validate": is_validate,
                },
            )

        agent_data.request_id = request_id

        if response.image is None or len(response.image) == 0:
            raise TaskUnrecoverableError(
                task_id=agent_data.request_id,
                error=f"Sandbox initialization failed: no screenshot. Response: {response.text}",
                step_count=0,
            )
        if not isinstance(response.image, list):
            raise TaskUnrecoverableError(
                task_id=agent_data.request_id,
                error=f"Sandbox initialization failed: screenshot is not a list, got {type(response.image)}",
                step_count=0,
            )

        initial_screenshot = response.image[0]
        initial_url = response.image_urls[0] if self.use_url_mode else None
        agent_data.add_image(initial_screenshot, initial_url)

        await self._prepare_prompt(agent_data, raw_prompt)

        if self.trajectory_splitting or self.use_flex_attention_hetero:
            await self._record_initial_image_range(agent_data)

        if len(agent_data.prompt_ids) > self.prompt_length:
            original_len = len(agent_data.prompt_ids)
            logger.error(
                f"[{agent_data.request_id}] Initial prompt length {original_len} exceeds "
                f"prompt_length limit {self.prompt_length}. "
                f"Discarding sample (task definition incomplete). Please check prompt_length."
            )
            agent_data.truncated = True
            raise TruncatedError(
                agent_data=agent_data,
                message=f"Initial prompt length {original_len} exceeds prompt_length {self.prompt_length}",
            )

        return GUIAgentState.GENERATING

    async def _handle_generating(self, agent_data: GUIAgentData, sampling_params: dict[str, Any]) -> GUIAgentState:
        """Generate model response with OAGI format validation and retry.

        This method has two modes based on trajectory_splitting config:

        trajectory_splitting=True (new mode):
        - Uses heterogeneous context with sliding window on images
        - Records turn_metadata for trajectory splitting
        - No response_length truncation (handled by trajectory splitting)

        trajectory_splitting=False (legacy mode):
        - Uses agent_data.prompt_ids directly (all images in context)
        - Pre-checks and adjusts max_tokens based on response_length
        - Truncates response if exceeds response_length limit
        """

        # Prepare generation context based on mode
        ctx = self._prepare_generation_context(agent_data, sampling_params)
        prompt_ids = ctx["prompt_ids"]
        image_data_for_gen = ctx["image_data_for_gen"]
        image_uuids = ctx["image_uuids"]
        visible_img_idxs = ctx["visible_img_idxs"]
        hetero_ids = ctx["hetero_ids"]

        for gen_attempt in range(self.MAX_GENERATION_RETRIES):
            with simple_timer("generate_sequences", agent_data.metrics, record_per_call=True):
                if self.dummy_mode == "full":
                    # Full dummy: use fixed tokens, skip real generation
                    output = self._generate_dummy_tokens(agent_data.step_count)
                else:
                    # Sandbox dummy: force temperature=0 for deterministic output
                    if self.dummy_mode == "sandbox":
                        sampling_params["temperature"] = 0.0
                    output = await self.server_manager.generate(
                        request_id=agent_data.request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        image_data=image_data_for_gen,
                        image_uuids=image_uuids,
                    )

            response_ids = output.token_ids
            generated_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # Dummy mode: skip validation (sandbox/full dummy doesn't need action format checks)
            if self.dummy_mode in ("full", "sandbox"):
                pass  # Skip validation
            else:
                try:
                    step = parse_raw_output(generated_text)
                    if not step.actions:
                        raise ValueError("No actions in parsed output")

                    converted = self._action_convertor(step.actions)
                    if not converted:
                        raise ValueError("Action conversion returned empty result")

                except Exception as e:
                    error_msg = f"{e}, output={generated_text}, actual_max_tokens={sampling_params['max_tokens']}"
                    logger.warning(
                        f"[{agent_data.request_id}] Validation failed "
                        f"(attempt {gen_attempt + 1}/{self.MAX_GENERATION_RETRIES}): {error_msg}"
                    )
                    agent_data.metrics["retry_count"] += 1
                    if gen_attempt == self.MAX_GENERATION_RETRIES - 1:
                        raise TaskUnrecoverableError(
                            task_id=agent_data.request_id,
                            error=f"Validation failed after {self.MAX_GENERATION_RETRIES} attempts: {error_msg}",
                            step_count=agent_data.step_count,
                        ) from e
                    continue

            if not self.trajectory_splitting:
                current_response_len = len(agent_data.response_mask)
                new_total_response_len = current_response_len + len(response_ids)

                if new_total_response_len > self.response_length:
                    available_len = self.response_length - current_response_len

                    if available_len <= 0:
                        logger.warning(
                            f"[{agent_data.request_id}] No space for new response at step {agent_data.step_count}, "
                            f"current_response_len={current_response_len}, response_length={self.response_length}"
                        )
                        agent_data.truncated = True
                        raise TruncatedError(
                            agent_data=agent_data,
                            message=f"Response buffer full at step {agent_data.step_count}",
                        )

                    truncated_response_ids = response_ids[:available_len]
                    logger.warning(
                        f"[{agent_data.request_id}] Truncating response from {len(response_ids)} "
                        f"to {available_len} tokens at step {agent_data.step_count} "
                        f"(response_length limit={self.response_length})"
                    )

                    agent_data.prompt_ids += truncated_response_ids
                    agent_data.response_mask += [1] * len(truncated_response_ids)
                    if output.log_probs is not None:
                        agent_data.response_logprobs += output.log_probs[:available_len]

                    agent_data.truncated = True
                    raise TruncatedError(
                        agent_data=agent_data,
                        message=f"Response truncated from {len(response_ids)} to {available_len} tokens",
                    )

            response_start_in_full = len(agent_data.prompt_ids)
            agent_data.prompt_ids += response_ids
            agent_data.response_mask += [1] * len(response_ids)
            if output.log_probs is not None:
                agent_data.response_logprobs += output.log_probs
            if output.routed_experts is not None:
                agent_data.routed_experts.extend(output.routed_experts)
            agent_data.generation_token_counts.append(len(response_ids))

            # Record turn metadata for trajectory splitting and FlexAttention modes
            if self.trajectory_splitting or self.use_flex_attention_hetero:
                self._record_turn_metadata(
                    agent_data=agent_data,
                    turn_idx=agent_data.step_count,
                    response_start_in_full=response_start_in_full,
                    response_len=len(response_ids),
                    visible_image_indices=visible_img_idxs,
                    hetero_prompt_len=len(hetero_ids),
                )

            agent_data.last_turn_output = generated_text
            return GUIAgentState.EXECUTING

    async def _handle_executing(self, agent_data: GUIAgentData) -> GUIAgentState:
        """Execute action in sandbox and get new screenshot."""
        # Pre-execution length check
        if self.trajectory_splitting:
            after_exec_len = self._compute_projected_hetero_length(agent_data)
            will_exceed = after_exec_len > self.prompt_length
            limit_name = "prompt_length (hetero)"
            limit_value = self.prompt_length
            current_total_len = after_exec_len
        else:
            current_response_len = len(agent_data.response_mask)
            after_exec_len = current_response_len + ENV_RESPONSE_TOKEN_LEN
            will_exceed = after_exec_len > self.response_length
            limit_name = "response_length"
            limit_value = self.response_length
            current_total_len = current_response_len

        if will_exceed:
            logger.warning(
                f"[{agent_data.request_id}] Skipping execute at step {agent_data.step_count}: "
                f"current ({current_total_len}) + ENV_RESPONSE_TOKEN_LEN ({ENV_RESPONSE_TOKEN_LEN}) "
                f"= {after_exec_len} > {limit_name} ({limit_value})"
            )
            agent_data.truncated = True
            raise TruncatedError(
                agent_data=agent_data,
                message=f"Pre-execute length check failed: would exceed {limit_name}",
            )

        action = agent_data.last_turn_output

        execute_start_time = time.time()
        with simple_timer("sandbox_execute", agent_data.metrics, record_per_call=True):
            response, final_score, metrics = await self._sandbox_tool.execute(agent_data.request_id, {"action": action})
        execute_end_time = time.time()

        sandbox_receive_time = metrics.get("sandbox_receive_time", 0.0)
        sandbox_send_time = metrics.get("sandbox_send_time", 0.0)

        if sandbox_receive_time > 0 and sandbox_send_time > 0:
            handle_executing_prep_time = sandbox_receive_time - execute_start_time
            handle_executing_return_time = execute_end_time - sandbox_send_time
            agent_data.metrics["handle_executing_prep_time"] = (
                agent_data.metrics.get("handle_executing_prep_time", 0.0) + handle_executing_prep_time
            )
            agent_data.metrics["handle_executing_return_time"] = (
                agent_data.metrics.get("handle_executing_return_time", 0.0) + handle_executing_return_time
            )

        for time_key in ["execute_oagi_time", "execute_finalize_time"]:
            time_value = metrics.get(time_key, 0.0)
            if time_value > 0:
                agent_data.metrics[time_key] = agent_data.metrics.get(time_key, 0.0) + time_value

        error_type = metrics.get("error")
        action_preview = (action or "").replace("\n", " ")
        error_msg = None
        if error_type in ("instance_not_found", "invalid_state"):
            error_msg = response.text or "Unknown error"
        elif error_type is not None and error_type != "max_steps_reached":
            error_msg = f"Sandbox execution error: {error_type}. Response: {response.text}"
        elif response.image is None or len(response.image) == 0:
            error_msg = f"Sandbox execution failed: no screenshot. Response: {response.text}"

        if error_msg:
            raise TaskUnrecoverableError(
                task_id=agent_data.request_id,
                error=f"{error_msg} | action_preview={action_preview}",
                step_count=agent_data.step_count,
            )

        new_screenshot = response.image[0] if isinstance(response.image, list) else response.image

        agent_data.step_count += 1

        is_done = metrics.get("done", False)
        is_max_steps = error_type == "max_steps_reached"
        should_terminate = is_done or is_max_steps

        if is_done:
            pass
        elif is_max_steps:
            assert agent_data.step_count == agent_data.max_steps, (
                f"BUG: Sandbox max_steps_reached but VERL step_count ({agent_data.step_count}) "
                f"!= max_steps ({agent_data.max_steps}). Config mismatch?"
            )
        elif agent_data.step_count >= agent_data.max_steps:
            raise RuntimeError(
                f"BUG: VERL reached max_steps ({agent_data.max_steps}) but sandbox didn't report "
                f"done or max_steps_reached. Config mismatch?"
            )

        # Terminating: don't add final screenshot (saves tokens, not used for training)
        if should_terminate:
            agent_data.done = is_done
            agent_data.final_score = final_score
            agent_data.verification_logs = metrics.get("verification_logs")
            return GUIAgentState.TERMINATED

        new_url = response.image_urls[0] if self.use_url_mode else None
        agent_data.add_image(new_screenshot, new_url)

        env_response_ids = await self._tokenize_env_response(agent_data, response)

        agent_data.prompt_ids += env_response_ids
        agent_data.response_mask += [0] * len(env_response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(env_response_ids)
        if agent_data.routed_experts:
            agent_data.routed_experts.extend([None] * len(env_response_ids))

        expected_images = 1 + agent_data.step_count
        assert len(agent_data.image_data) == expected_images, (
            f"Image count mismatch during execution: {len(agent_data.image_data)} != {expected_images}"
        )
        assert len(agent_data.image_uuids) == expected_images, (
            f"Image UUID count mismatch: {len(agent_data.image_uuids)} != {expected_images}"
        )

        return GUIAgentState.GENERATING

    async def _prepare_prompt(self, agent_data: GUIAgentData, raw_prompt: list[dict[str, Any]]) -> None:
        """Prepare initial prompt tokens for VLM."""
        assert len(raw_prompt) == 1 and raw_prompt[0].get("role") == "user", f"Invalid prompt structure: {raw_prompt}"

        instruction = raw_prompt[0].get("content")
        assert isinstance(instruction, str) and instruction, (
            f"Instruction must be non-empty string, got: {instruction!r}"
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": instruction}, {"type": "image"}]}]

        formatted_prompt = await self.loop.run_in_executor(
            None,
            lambda: self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            ),
        )
        model_inputs = self.processor(text=[formatted_prompt], images=agent_data.image_data, return_tensors="pt")
        agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()

    def _prepare_generation_context(
        self,
        agent_data: GUIAgentData,
        sampling_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Prepare generation context based on trajectory_splitting mode."""

        if self.trajectory_splitting or self.use_flex_attention_hetero:
            # Trajectory splitting mode
            hetero_ids, visible_img_idxs = self._get_heterogeneous_prompt_ids(agent_data)

            vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            assert vision_start_id is not None
            vision_start_count = hetero_ids.count(vision_start_id)

            # Only latest visible image needs actual data; others use UUID cache
            if visible_img_idxs:
                image_data_for_gen = [
                    None if i < len(visible_img_idxs) - 1 else agent_data.image_data[visible_img_idxs[i]]
                    for i in range(len(visible_img_idxs))
                ]
                image_uuids = [agent_data.image_uuids[idx] for idx in visible_img_idxs]
            else:
                image_data_for_gen = None
                image_uuids = None

            if visible_img_idxs and vision_start_count != len(visible_img_idxs):
                raise ValueError("Consistency problems.")

            return {
                "prompt_ids": hetero_ids,
                "image_data_for_gen": image_data_for_gen,
                "image_uuids": image_uuids,
                "visible_img_idxs": visible_img_idxs,
                "hetero_ids": hetero_ids,
            }
        else:
            # Legacy mode: pre-check and adjust max_tokens
            current_response_len = len(agent_data.response_mask)
            available_for_generation = self.response_length - current_response_len

            if agent_data.step_count < agent_data.max_steps - 1:
                if available_for_generation > ENV_RESPONSE_TOKEN_LEN:
                    available_for_generation -= ENV_RESPONSE_TOKEN_LEN
                else:
                    logger.warning(
                        f"[{agent_data.request_id}] response_length ({self.response_length}) is too small "
                        f"for multi-turn with ENV_RESPONSE_TOKEN_LEN ({ENV_RESPONSE_TOKEN_LEN}). "
                        f"This trajectory will likely truncate early."
                    )

            requested_max_tokens = sampling_params.get("max_tokens", self.response_length)
            actual_max_tokens = min(requested_max_tokens, available_for_generation)

            if actual_max_tokens <= 0:
                logger.warning(
                    f"[{agent_data.request_id}] No space for generation at step {agent_data.step_count}: "
                    f"current_response_len={current_response_len}, response_length={self.response_length}"
                )
                agent_data.truncated = True
                raise TruncatedError(
                    agent_data=agent_data, message=f"No space for generation at step {agent_data.step_count}"
                )

            if actual_max_tokens != requested_max_tokens:
                logger.info(
                    f"[{agent_data.request_id}] Adjusting max_tokens from {requested_max_tokens} "
                    f"to {actual_max_tokens} (available_for_generation={available_for_generation}, "
                    f"current_response_len={current_response_len})"
                )
            sampling_params["max_tokens"] = actual_max_tokens

            # Only latest image needs actual data; others use UUID cache
            num_images = len(agent_data.image_data)
            if num_images > 0:
                image_data_for_gen = [None] * (num_images - 1) + [agent_data.image_data[-1]]
                image_uuids = agent_data.image_uuids
            else:
                image_data_for_gen = None
                image_uuids = None

            return {
                "prompt_ids": agent_data.prompt_ids,
                "image_data_for_gen": image_data_for_gen,
                "image_uuids": image_uuids,
                "visible_img_idxs": None,
                "hetero_ids": None,
            }

    def _find_vision_token_range(self, token_ids: list[int], start_offset: int = 0) -> tuple[int, int] | None:
        """Find vision token range (start, end+1) in token_ids, or None if not found."""
        vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")

        if vision_start_id is None or vision_end_id is None:
            return None

        vision_start_idx = None
        vision_end_idx = None
        for i, tok_id in enumerate(token_ids):
            if tok_id == vision_start_id and vision_start_idx is None:
                vision_start_idx = i
            elif tok_id == vision_end_id:
                vision_end_idx = i
                break

        if vision_start_idx is None or vision_end_idx is None:
            return None

        return (start_offset + vision_start_idx, start_offset + vision_end_idx + 1)

    def _strip_leading_bos(self, token_ids: list[int]) -> list[int]:
        """Strip leading BOS token if present to avoid duplication."""
        if not token_ids:
            return token_ids

        bos_id = self.tokenizer.bos_token_id
        if bos_id is not None and token_ids[0] == bos_id:
            return token_ids[1:]
        return token_ids

    async def _record_initial_image_range(self, agent_data: GUIAgentData) -> None:
        """Record initial image token range into env_token_ranges[0].

        This finds where the image tokens are in the initial prompt by searching
        for <|vision_start|> and <|vision_end|> tokens directly in the prompt_ids.

        The initial message content is structured as:
        [{"type": "text", "text": instruction}, {"type": "image"}]

        So the expected token structure is:
        <|im_start|>user
        instruction_text
        <|vision_start|><|image_pad|>...<|vision_end|>
        <|im_end|>
        <|im_start|>assistant

        """
        vision_range = self._find_vision_token_range(agent_data.prompt_ids)
        if vision_range is None:
            logger.warning(
                f"[{agent_data.request_id}] _record_initial_image_range: "
                f"Could not find vision tokens in prompt_ids (len={len(agent_data.prompt_ids)})"
            )
            return

        agent_data.env_token_ranges.insert(0, vision_range)

    async def _tokenize_image_message(self, image: Any) -> list[int]:
        """Tokenize an image-only user message using base conversation prefix trimming."""
        assert image is not None, "Image cannot be None"

        env_message = {"role": "user", "content": [{"type": "image"}]}
        messages_with_base = [*self._base_chat_history, env_message]

        formatted_prompt = await self.loop.run_in_executor(
            None,
            lambda: self.processor.apply_chat_template(
                messages_with_base,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            ),
        )

        model_inputs = self.processor(text=[formatted_prompt], images=[image], return_tensors="pt")
        full_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        trimmed_ids = full_ids[self._base_trim_length :]
        return self._strip_leading_bos(trimmed_ids)

    async def _tokenize_env_response(self, agent_data: GUIAgentData, response) -> list[int]:
        """Tokenize environment response and optionally record its image range.

        This method:
        1. Tokenizes the image-only message (full env response)
        2. When trajectory_splitting=True: Records the vision token range in env_token_ranges

        The full env_tokens includes:
        <|im_start|>user\n<|vision_start|>...<|vision_end|><|im_end|>\n<|im_start|>assistant\n

        But env_token_ranges only records the vision part:
        <|vision_start|>...<|vision_end|>

        This ensures that when sliding window drops an image:
        - The user message structure is preserved: <|im_start|>user\n ... <|im_end|>\n
        - Only the vision tokens are removed: <|vision_start|><|image_pad|>×N<|vision_end|>
        - The assistant marker is preserved: <|im_start|>assistant\n

        Result after DROP: <|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n
        (empty user message with preserved structure)
        """
        assert response.image is not None, "Environment response must include an image"
        new_image = response.image[0]
        current_full_pos = len(agent_data.prompt_ids)
        env_tokens = await self._tokenize_image_message(new_image)

        if self.trajectory_splitting or self.use_flex_attention_hetero:
            vision_range = self._find_vision_token_range(env_tokens, start_offset=current_full_pos)
            assert vision_range is not None, "Could not find vision tokens in env_tokens"

            agent_data.env_token_ranges.append(vision_range)

        return env_tokens

    def _compute_projected_hetero_length(self, agent_data: GUIAgentData) -> int:
        """Compute projected heterogeneous context length after adding one env response."""
        hetero_ids, _ = self._get_heterogeneous_prompt_ids(agent_data)
        current_hetero_len = len(hetero_ids)

        num_images_after = len(agent_data.image_data) + 1
        if num_images_after <= VISUAL_WINDOW_SIZE:
            return current_hetero_len + ENV_RESPONSE_TOKEN_LEN
        else:
            # Image count exceeds window: new image adds ~1135, old image drops ~1135
            # They roughly cancel out, only add turn structure tokens (~8):
            # <|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n
            return current_hetero_len + 8

    def _record_turn_metadata(
        self,
        agent_data: GUIAgentData,
        turn_idx: int,
        response_start_in_full: int,
        response_len: int,
        visible_image_indices: list[int],
        hetero_prompt_len: int,
    ) -> None:
        """Record per-turn metadata for trajectory splitting."""
        turn_meta = {
            "turn_idx": turn_idx,
            "response_start_in_full": response_start_in_full,
            "response_len": response_len,
            "visible_image_indices": visible_image_indices,
            "hetero_prompt_len": hetero_prompt_len,
            "logprob_start": len(agent_data.response_logprobs) - response_len,
            "routed_experts_start": len(agent_data.routed_experts) - response_len,
        }
        agent_data.turn_metadata.append(turn_meta)

    def _get_heterogeneous_prompt_ids(self, agent_data: GUIAgentData) -> tuple[list[int], list[int]]:
        """Get heterogeneous prompt_ids with sliding window on images while preserving ALL text.

        TITOUT (Text In, Text Out, Used Tokens) core logic:
        - Keep ALL text tokens (instruction + all model responses)
        - Only keep most recent VISUAL_WINDOW_SIZE images
        - All image blocks are in env_token_ranges (initial image at index 0)
        - Additionally ensure hetero_ids <= prompt_length (may remove more images)

        Example with 5 images and VISUAL_WINDOW_SIZE=3:
            Full: [instr][img0][resp0][env1][resp1][env2][resp2][env3][resp3][env4]
            Hetero: [instr][resp0][resp1][env2][resp2][env3][resp3][env4]  # keep last 3 images
            visible_img_idxs: [2, 3, 4]

        Returns:
            Tuple of (hetero_prompt_ids, visible_image_indices)
        """

        total_images = len(agent_data.image_data)
        total_blocks = len(agent_data.env_token_ranges)

        if total_blocks != total_images:
            logger.error(
                f"[{agent_data.request_id}] CRITICAL: Block count {total_blocks} != image count {total_images}"
            )

        if total_images <= VISUAL_WINDOW_SIZE:
            visible_idxs = list(range(total_images))
            hetero_ids = agent_data.prompt_ids.copy()
        else:
            first_visible_idx = total_images - VISUAL_WINDOW_SIZE
            visible_idxs = list(range(first_visible_idx, total_images))
            hetero_ids = apply_sliding_window_to_images(
                prompt_ids=agent_data.prompt_ids,
                env_token_ranges=agent_data.env_token_ranges,
                keep_indices=set(visible_idxs),
            )

        if self.trajectory_splitting:
            if len(hetero_ids) > self.prompt_length and len(visible_idxs) > 1:
                raise ValueError(
                    f"[{agent_data.request_id}] CRITICAL: Prompt length {self.prompt_length} exceeded "
                    f"after sliding window: {len(hetero_ids)}"
                )

        return hetero_ids, visible_idxs

    def _build_output(self, agent_data: GUIAgentData) -> AgentLoopOutput:
        """Build AgentLoopOutput from accumulated agent_data."""
        if agent_data.truncated and agent_data.final_score is None:
            agent_data.final_score = 0.0

        if not agent_data.remove_sample:
            assert len(agent_data.response_mask) > 0, "response_mask is empty - no model generation happened"
            assert agent_data.response_mask[0] == 1, "First token must be model-generated (mask=1)"

        if len(agent_data.response_mask) > 0:
            assert len(agent_data.prompt_ids) > len(agent_data.response_mask), (
                f"prompt_ids ({len(agent_data.prompt_ids)}) should be > response_mask ({len(agent_data.response_mask)})"
            )

        assert agent_data.final_score is not None, "final_score is None - sandbox did not return a score"

        response_len = len(agent_data.response_mask)
        assert response_len > 0
        response_ids = agent_data.prompt_ids[-response_len:]
        prompt_ids = agent_data.prompt_ids[:-response_len]

        if agent_data.remove_sample:
            response_mask = [0] * len(agent_data.response_mask)
            logger.info(f"[{agent_data.request_id}] remove_sample=True - setting all response_mask to 0")
        else:
            response_mask = agent_data.response_mask

        # ========== Image Transfer Optimization ==========
        # image: Used by _agent_loop_postprocess to compute position_ids (requires image_grid_thw)
        # image_urls: Used by fsdp_workers for lazy loading, skipping Ray transfer of multi_modal_inputs
        if self.use_url_mode:
            if agent_data.image_urls:
                assert len(agent_data.image_urls) == len(agent_data.image_data), (
                    f"image_urls count ({len(agent_data.image_urls)}) != "
                    f"image_data count ({len(agent_data.image_data)})"
                )
            multi_modal_data = {
                "image": agent_data.image_data,
                "image_urls": agent_data.image_urls or [],
            }
        else:
            multi_modal_data = {"image": agent_data.image_data}

        # Validate image_count = step_count for terminated trajectories
        if not agent_data.remove_sample and (agent_data.done or agent_data.step_count >= agent_data.max_steps):
            expected_images = agent_data.step_count
            assert len(agent_data.image_data) == expected_images, (
                f"Image count mismatch: {len(agent_data.image_data)} images != {expected_images} steps."
            )

        metrics = AgentLoopMetrics(
            generate_sequences=agent_data.metrics.get("generate_sequences", 0.0),
            tool_calls=agent_data.metrics.get("sandbox_execute", 0.0),
            sandbox_create=agent_data.metrics.get("sandbox_create", 0.0),
            step_count=agent_data.step_count,
            retry_count=agent_data.metrics.get("retry_count", 0),
            execute_oagi_time=agent_data.metrics.get("execute_oagi_time", 0.0),
            execute_finalize_time=agent_data.metrics.get("execute_finalize_time", 0.0),
            handle_executing_prep_time=agent_data.metrics.get("handle_executing_prep_time", 0.0),
            handle_executing_return_time=agent_data.metrics.get("handle_executing_return_time", 0.0),
            generation_token_counts=agent_data.generation_token_counts,
        )

        # Trajectory splitting: no truncation (full trajectory needed for reconstruction)
        # Legacy mode: truncate to fit length limits
        if not self.trajectory_splitting:
            if len(prompt_ids) > self.prompt_length:
                prompt_ids = prompt_ids[-self.prompt_length :]

            if len(response_ids) > self.response_length:
                response_ids = response_ids[: self.response_length]
                response_mask = response_mask[: self.response_length]
                if agent_data.response_logprobs:
                    agent_data.response_logprobs = agent_data.response_logprobs[: self.response_length]
                if agent_data.routed_experts:
                    agent_data.routed_experts = agent_data.routed_experts[: self.response_length]
                if not agent_data.truncated:
                    agent_data.truncated = True
                    logger.info(f"[{agent_data.request_id}] Marked as truncated due to response_ids overflow")

        if (self.trajectory_splitting or self.use_flex_attention_hetero) and not agent_data.remove_sample:
            assert len(agent_data.turn_metadata) > 0, "turn_metadata is empty but remove_sample=False"
            assert len(agent_data.env_token_ranges) > 0, "env_token_ranges is empty but remove_sample=False"
            expected_counts = (
                (agent_data.step_count,)
                if not agent_data.truncated
                else (agent_data.step_count, agent_data.step_count + 1)
            )
            assert len(agent_data.turn_metadata) in expected_counts, (
                f"turn_metadata count {len(agent_data.turn_metadata)} not in expected {expected_counts}"
            )

        extra_fields = {
            "step_count": agent_data.step_count,
            "done": agent_data.done,
            "final_score": agent_data.final_score,
            "verification_logs": agent_data.verification_logs,
            "truncated": agent_data.truncated,
            "remove_sample": agent_data.remove_sample,
            "raw_metrics": agent_data.metrics,
            "task_config": agent_data.task_config,
        }

        if self.trajectory_splitting or self.use_flex_attention_hetero:
            extra_fields.update(
                {
                    "turn_metadata": agent_data.turn_metadata,
                    "env_token_ranges": agent_data.env_token_ranges,
                    "token_ids": agent_data.prompt_ids,
                }
            )

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs if agent_data.response_logprobs else None,
            routed_experts=agent_data.routed_experts if agent_data.routed_experts else None,
            reward_score=agent_data.final_score,
            step_count=agent_data.step_count,
            metrics=metrics,
            extra_fields=extra_fields,
        )

    def _clone_sampling_params(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
        """Clone sampling params to avoid cross-attempt mutation."""
        return copy.deepcopy(sampling_params)

    async def _release_sandbox_safe(self, request_id: str, task_id: str, is_validate: bool, reason: str = "") -> None:
        """Best-effort sandbox release without raising."""
        try:
            # Clear cache for discarded samples (retry_failed, truncated)
            # Both have remove_sample=True, images won't be downloaded
            clear_cache = reason in ("retry_failed", "truncated") or is_validate

            await self._sandbox_tool.release(request_id, clear_cache=clear_cache)
            if reason:
                logger.warning(
                    f"[{task_id}] Released resources for {reason} "
                    f"(validate={is_validate}, clear_cache={clear_cache}): {request_id}"
                )
        except Exception as release_err:
            logger.warning(f"[{task_id}] Failed to release {request_id} ({reason}): {release_err}")

    async def _build_failed_agent_data(
        self,
        task_id: str,
        raw_prompt: list[dict[str, Any]],
        task_config: dict,
        retry_count: int,
        last_error: Exception | None,
        max_retries: int,
    ) -> GUIAgentData:
        """Construct minimal failed agent_data that satisfies _build_output assertions."""
        request_id = f"{task_id}-failed"
        agent_data = GUIAgentData(request_id=request_id, task_config=task_config)

        pad_id = self.tokenizer.pad_token_id or 0
        eos_id = self.tokenizer.eos_token_id or 0

        try:
            text_prompt = ""
            for msg in raw_prompt:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                    content = " ".join(text_parts)
                text_prompt += f"{role}: {content}\n"

            tokens = self.tokenizer.encode(text_prompt, add_special_tokens=True)
            if len(tokens) > self.prompt_length:
                tokens = tokens[: self.prompt_length]
            agent_data.prompt_ids = tokens
        except Exception as e:
            logger.warning(f"[{request_id}] Failed to tokenize prompt: {e}. Using dummy tokens.")
            dummy_prompt_len = min(100, self.prompt_length)
            agent_data.prompt_ids = [pad_id] * dummy_prompt_len

        # Append a single dummy response token (eos) so _build_output can split prompt/response
        agent_data.prompt_ids += [eos_id]
        agent_data.response_mask = [1]
        agent_data.response_logprobs = [0.0] if self.calculate_log_probs else []
        agent_data.routed_experts = []

        dummy_image = Image.fromarray(np.zeros((self.dummy_image_height, self.dummy_image_width, 3), dtype=np.uint8))
        agent_data.image_data = [dummy_image]
        agent_data.image_uuids = [uuid4().hex]
        agent_data.image_urls = []
        agent_data.remove_sample = True
        agent_data.final_score = 0.0
        agent_data.done = False
        agent_data.step_count = 0
        agent_data.truncated = False  # Explicit: not truncated, just failed
        agent_data.verification_logs = None  # No logs for failed samples
        agent_data.turn_metadata = []  # No turns completed
        agent_data.env_token_ranges = []  # No environment token ranges
        agent_data.generation_token_counts = []

        agent_data.metrics["retry_count"] = retry_count
        if isinstance(last_error, TruncatedError):
            agent_data.metrics["failure_reason"] = f"Sequence truncated: {last_error}"
        else:
            agent_data.metrics["failure_reason"] = f"All {max_retries} attempts failed: {last_error}"
        return agent_data

    async def initialize_only(self, **kwargs) -> "GUIAgentData":
        """Initialize sandbox only, without running generation (for prefetching)."""
        if self._sandbox_tool is None:
            raise RuntimeError("GUIAgentLoop._sandbox_tool is not initialized.")

        task_config, task_id, raw_prompt = self._extract_task_config(kwargs)
        is_validate = kwargs.get("validate", False)
        agent_data = self._create_agent_data(task_id, task_config)
        state = await self._handle_initializing(agent_data, is_validate, raw_prompt)

        if state != GUIAgentState.GENERATING:
            raise RuntimeError(f"Prefetch initialization failed for {task_id}: expected state GENERATING, got {state}")

        return agent_data
