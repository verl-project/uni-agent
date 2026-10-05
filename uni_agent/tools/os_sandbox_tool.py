# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
OS Sandbox Tool for GUI Agent Training

This tool provides a real OS environment (sandbox) for training GUI agents
using VERL's multi-turn VLM training loop. It supports:
1. Passing raw model outputs directly to the sandbox runtime
2. Returning screenshots after each action for the next turn
3. Task verification and final_score computation

The tool follows the verl BaseTool interface and integrates with the
agent loop system, similar to the DeepEyes image search tool.

Key Design: VERL does NOT parse or interpret actions. The raw model output
is passed directly to the sandbox runtime, which handles all action parsing,
execution, and returns screenshots with execution results.

Architecture:
- OSSandboxTool: VERL tool interface (create/execute)
- SandboxInstance: Per-trajectory state management
- SandboxClient: HTTP client for sandbox API

API Endpoints (via VERL Proxy or Executor):
- POST /api/v1/sandbox/create  - Create task
- POST /api/v1/sandbox/execute - Execute action
- GET /health                   - Health check

Based on: VERL_INTEGRATION_DESIGN.md, VERL_SANDBOX_API.md
"""

import asyncio
import base64
import io
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
# Set to INFO by default for better debugging; use VERL_LOGGING_LEVEL=WARN for production
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class InstanceState(Enum):
    """States for a sandbox instance."""

    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Error Classes for Sandbox Operations
# =============================================================================

# Unrecoverable error keywords that indicate Runner/VM state is corrupted
# Note: These patterns are checked against lowercased error messages
# Be careful not to match false positives (e.g., "connection established")
UNRECOVERABLE_ERROR_KEYWORDS = [
    "not found",  # Task lost (e.g., "task not found")
    "crashed",  # Runner crashed
    "connection refused",  # Connection refused (more specific than "connection")
    "connection error",  # Connection error
    "connection lost",  # Connection lost
    "connection reset",  # Connection reset
    "timed out",  # Timed out (more specific than "timeout")
    "execution timeout",  # Execution timeout
]


class TaskUnrecoverableError(Exception):
    """Unrecoverable task error that requires task recreation and re-rollout.

    Raised when the following errors are detected:
    - Task not found: task_id does not exist or has been released
    - Runner crashed: Runner/VM crashed
    - Connection refused/error: Connection issues
    - Execution timeout: Execution timed out

    Attributes:
        task_id: The failed task ID
        error: Original error message
        step_count: Step count at failure time
        instance_id: VERL tool instance ID
    """

    def __init__(
        self,
        task_id: str,
        error: str,
        step_count: int = 0,
        instance_id: str | None = None,
    ):
        self.task_id = task_id
        self.error = error
        self.step_count = step_count
        self.instance_id = instance_id
        super().__init__(f"Task {task_id} unrecoverable at step {step_count}: {error}")


class SystemUnavailableError(Exception):
    """System-level error that can be retried after waiting.

    Raised when the following errors are detected:
    - No available runners: Runner pool exhausted
    - Service unavailable: Service temporarily unavailable

    Attributes:
        error: Original error message
        retry_after: Suggested retry wait time in seconds
    """

    def __init__(self, error: str, retry_after: float = 2.0):
        self.error = error
        self.retry_after = retry_after
        super().__init__(f"System unavailable: {error}. Retry after {retry_after}s")


def is_unrecoverable_error(error_message: str) -> bool:
    """Check if an error message indicates an unrecoverable error.

    Args:
        error_message: The error message to check.

    Returns:
        True if the error is unrecoverable and requires re-rollout.
    """
    if not error_message:
        return False
    error_lower = error_message.lower()
    return any(keyword in error_lower for keyword in UNRECOVERABLE_ERROR_KEYWORDS)


def is_system_unavailable_error(error_message: str) -> bool:
    """Check if an error indicates system unavailability.

    Args:
        error_message: The error message to check.

    Returns:
        True if the error indicates system-level unavailability.
    """
    if not error_message:
        return False
    error_lower = error_message.lower()
    system_keywords = ["no available runners", "service unavailable", "pool exhausted"]
    return any(keyword in error_lower for keyword in system_keywords)


@dataclass
class SandboxInstance:
    """State container for a single sandbox instance (one trajectory).

    Attributes:
        instance_id: Unique identifier, also used as Sandbox task_id.
        task_config: Task configuration from DataProto.non_tensor_batch.
        state: Current state of the instance.
        trajectory: List of executed actions.
        step_count: Number of steps executed.
        screenshots: Captured screenshots (optional, for debugging).
        final_score: Final score from sandbox verification (when done=true).
        verification_logs: Verification process logs.
        done: Whether the task is completed.
        released: Whether the task has been released.
        extra_info: Extra information for the instance.
    """

    instance_id: str
    task_config: dict
    state: InstanceState = InstanceState.INITIALIZING
    trajectory: list[str] = field(default_factory=list)
    step_count: int = 0
    screenshots: list[Image.Image] = field(default_factory=list)
    final_score: Optional[float] = None
    verification_logs: Optional[str] = None  # Verification process logs
    done: bool = False
    released: bool = False
    extra_info: dict = field(default_factory=dict)


class SandboxClient:
    """HTTP client for communicating with Sandbox API.

    Supports connection via VERL Proxy (port 9000) or direct Executor (port 8000).

    API Endpoints:
    - POST /api/v1/sandbox/create  - Create task
    - POST /api/v1/sandbox/execute - Execute action
    - GET /health                   - Health check
    """

    def __init__(self, sandbox_url: str, create_timeout: float = 300.0, execute_timeout: float = 60.0):
        """Initialize the sandbox client.

        Args:
            sandbox_url: Base URL of the sandbox service
                        (e.g., "http://localhost:9000" for Proxy)
            create_timeout: Timeout in seconds for create operations (default: 300s)
            execute_timeout: Timeout in seconds for execute operations (default: 60s)
        """
        self.sandbox_url = sandbox_url.rstrip("/")
        self.create_timeout = create_timeout
        self.execute_timeout = execute_timeout
        self._session = None

    async def _get_session(self):
        """Get or create aiohttp session lazily.

        Note: We use per-request timeouts in create() and execute() methods,
        so the session-level timeout is disabled to allow flexible per-operation control.
        For HTTPS with self-signed certs, SSL verification is disabled.
        """
        if self._session is None:
            try:
                import ssl as _ssl

                import aiohttp

                # Disable session-level timeout - we use per-request timeouts instead
                timeout = aiohttp.ClientTimeout(total=None)
                # Disable SSL verification for self-signed certs (HTTPS sandbox gateway)
                connector = None
                if self.sandbox_url.startswith("https://"):
                    ssl_ctx = _ssl.create_default_context()
                    ssl_ctx.check_hostname = False
                    ssl_ctx.verify_mode = _ssl.CERT_NONE
                    connector = aiohttp.TCPConnector(ssl=ssl_ctx)
                self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            except ImportError as e:
                raise ImportError("aiohttp is required for OSSandboxTool. Install it with: pip install aiohttp") from e
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def health_check(self) -> dict:
        """Check sandbox service health.

        Returns:
            Health status dict with available_runners, busy_runners, etc.
        """
        session = await self._get_session()
        async with session.get(f"{self.sandbox_url}/health") as response:
            response.raise_for_status()
            return await response.json()

    async def create(
        self,
        task_id: str,
        task_config: dict,
        max_steps: int | None = None,
        validate: bool = False,
    ) -> dict:
        """Create a new sandbox task instance.

        Args:
            task_id: Unique task identifier (generated by VERL)
            task_config: Task configuration dict containing:
                - config: list of setup steps
                - evaluator: verification config
            max_steps: Maximum steps for this task (per-task override)
            validate: Whether this is a validate rollout (for cache management)

        Returns:
            Response dict with 'task_id', 'screenshot' (base64), 'error'

        Raises:
            asyncio.TimeoutError: If operation exceeds create_timeout
        """
        import asyncio

        try:
            session = await self._get_session()
            request_body = {"task_id": task_id, "task_config": task_config, "validate": validate}
            if max_steps is not None:
                request_body["max_steps"] = max_steps

            async with session.post(f"{self.sandbox_url}/api/v1/sandbox/create", json=request_body) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"[SandboxClient] Create failed: status={response.status}, body={text[:500]}")
                    response.raise_for_status()

                return await asyncio.wait_for(response.json(), timeout=self.create_timeout)
        except asyncio.TimeoutError as e:
            logger.error(f"[SandboxClient] Create timeout after {self.create_timeout}s for task {task_id}")
            raise asyncio.TimeoutError(f"Sandbox create operation timed out after {self.create_timeout}s") from e
        except Exception as e:
            logger.error(f"[SandboxClient] Create exception: {type(e).__name__}: {e}")
            raise

    async def execute(self, task_id: str, action: str) -> dict:
        """Execute an action in the sandbox.

        Args:
            task_id: The task ID
            action: Raw action string from model output

        Returns:
            Response dict with:
                - task_id: str
                - screenshot: base64 image data
                - step_count: int
                - done: bool
                - score: float | None (only when done=true)
                - released: bool
                - error: str | None
                - execute_oagi_time: float (execution time in seconds)

        Raises:
            asyncio.TimeoutError: If operation exceeds execute_timeout
        """
        import asyncio

        try:
            session = await self._get_session()
            async with session.post(
                f"{self.sandbox_url}/api/v1/sandbox/execute", json={"task_id": task_id, "action": action}
            ) as response:
                response.raise_for_status()
                return await asyncio.wait_for(response.json(), timeout=self.execute_timeout)
        except asyncio.TimeoutError as e:
            logger.error(f"[SandboxClient] Execute timeout after {self.execute_timeout}s for task {task_id}")
            raise asyncio.TimeoutError(f"Sandbox execute operation timed out after {self.execute_timeout}s") from e

    async def release(self, task_id: str, clear_cache: bool = True) -> dict:
        """Release task instance (for VERL retry scenarios).

        Releases the runner on sandbox side and optionally clears screenshot cache.

        Args:
            task_id: Task ID (request_id) to release
            clear_cache: Whether to also clear screenshot cache (default True)

        Returns:
            Response dict with:
                - task_id: str
                - success: bool
                - cache_cleared: int (number of screenshots cleared)
                - error: str | None
        """
        import asyncio

        try:
            session = await self._get_session()
            async with session.post(
                f"{self.sandbox_url}/api/v1/sandbox/release", json={"task_id": task_id, "clear_cache": clear_cache}
            ) as response:
                # NOTE: When calling release, either generate failed (runner and cache not released),
                # or execute failed (runner and cache already released)
                if response.status == 404:
                    # Task not found is not an error - may already be released
                    return {"task_id": task_id, "success": True, "cache_cleared": 0, "error": None}
                response.raise_for_status()
                return await asyncio.wait_for(response.json(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(f"[SandboxClient] Release timeout for task {task_id}")
            return {"task_id": task_id, "success": False, "cache_cleared": 0, "error": "timeout"}
        except Exception as e:
            logger.warning(f"[SandboxClient] Release failed for task {task_id}: {e}")
            return {"task_id": task_id, "success": False, "cache_cleared": 0, "error": str(e)}


class DummySandboxTool:
    """Deterministic dummy sandbox: local images or pure white fallback, score=0 at max_steps.

    Used for testing pipeline without real sandbox infrastructure:
    - "sandbox" mode: Test model generation with controlled environment
    - "full" mode: Test entire pipeline with fixed inputs/outputs

    When ``dummy_images_dir`` is provided (or the env var
    ``DUMMY_IMAGES_DIR`` is set), images are read sequentially from that
    directory (``step_0.jpg``, ``step_1.jpg``, …).  If the step index
    exceeds available images, it wraps around via modulo.  When no
    directory is configured, a pure-white placeholder image is returned
    (original behaviour).

    Interface compatible with OSSandboxTool for drop-in replacement.
    """

    # Hardcoded constants
    IMAGE_WIDTH = 1260
    IMAGE_HEIGHT = 700
    FINAL_SCORE = 0.0

    def __init__(self, max_steps: int):
        self.max_steps = max_steps  # Default max_steps
        self._white_image: Image.Image | None = None
        self._steps: dict[str, int] = {}
        self._task_max_steps: dict[str, int] = {}  # Per-task max_steps

        # Local image directory for sandbox dummy mode
        self._dummy_images_dir = os.path.join(os.path.dirname(__file__), "../../gui_scripts/dummy_sandbox_images")
        self._local_images: list[Image.Image] = []
        if self._dummy_images_dir and os.path.isdir(self._dummy_images_dir):
            self._load_local_images()
        elif self._dummy_images_dir:
            logger.warning(
                f"[DummySandboxTool] dummy_images_dir not found: {self._dummy_images_dir}, falling back to white images"
            )

    def _load_local_images(self) -> None:
        """Load all step_*.jpg images from the local directory, sorted by index."""
        import glob

        pattern = os.path.join(self._dummy_images_dir, "step_*.jpg")
        files = glob.glob(pattern)
        # Sort by numeric index: step_0.jpg, step_1.jpg, ...
        files.sort(key=lambda f: int(os.path.basename(f).replace("step_", "").replace(".jpg", "")))
        for f in files:
            self._local_images.append(Image.open(f).copy())
        logger.info(f"[DummySandboxTool] Loaded {len(self._local_images)} local images from {self._dummy_images_dir}")

    def _get_image(self, step: int) -> Image.Image:
        """Get image for a given step: local image if available, else white."""
        if self._local_images:
            idx = step % len(self._local_images)
            return self._local_images[idx]
        return self._get_white_image()

    def _get_white_image(self) -> Image.Image:
        """Get cached pure white image (deterministic)."""
        if self._white_image is None:
            self._white_image = Image.fromarray(np.full((self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3), 255, dtype=np.uint8))
        return self._white_image

    async def create(self, instance_id: str, **kwargs) -> tuple[str, ToolResponse]:
        """Create dummy instance with initial screenshot."""
        # Extract per-task max_steps from kwargs, fall back to default
        create_kwargs = kwargs.get("create_kwargs", {})
        task_max_steps = create_kwargs.get("max_steps") or self.max_steps
        self._steps[instance_id] = 0
        self._task_max_steps[instance_id] = task_max_steps
        return instance_id, ToolResponse(
            text="Dummy environment initialized.",
            image=[self._get_image(0)],
            image_urls=None,
        )

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float | None, dict]:
        """Execute in dummy sandbox: increment step, return local image or white image."""
        step = self._steps[instance_id] = self._steps.get(instance_id, 0) + 1
        task_max_steps = self._task_max_steps.get(instance_id, self.max_steps)
        is_max_steps = step >= task_max_steps
        metrics = {"step_count": step, "done": False}

        if is_max_steps:
            metrics["error"] = "max_steps_reached"
            self._steps.pop(instance_id, None)
            self._task_max_steps.pop(instance_id, None)
            return (
                ToolResponse(text=f"Max steps reached. Score: {self.FINAL_SCORE}", image=[self._get_image(step)]),
                self.FINAL_SCORE,
                metrics,
            )
        return (
            ToolResponse(text=f"Executed. Step: {step}/{task_max_steps}", image=[self._get_image(step)]),
            None,
            metrics,
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release dummy instance."""
        self._steps.pop(instance_id, None)
        self._task_max_steps.pop(instance_id, None)


class OSSandboxTool(BaseTool):
    """OS Sandbox Tool for GUI Agent multi-turn VLM training.

    This tool provides a real OS environment for training GUI agents.
    Each execution returns a screenshot that is added to the conversation
    context for the next turn.

    Inherits from BaseTool for unified tool interface compatibility with
    tool_registry and tool_agent_loop patterns.

    Config options:
        max_steps (int): Maximum steps per trajectory (default: 20)
        sandbox_url (str): Sandbox service URL (default: "http://localhost:9000")
        create_timeout (float): Timeout in seconds for create operations (default: 300.0)
        execute_timeout (float): Timeout in seconds for execute operations (default: 60.0)

    Note: For testing/debugging without sandbox, use GUI_AGENT_DUMMY_MODE=true
    environment variable which is handled at the agent loop level.
    """

    @staticmethod
    def _default_tool_schema() -> OpenAIFunctionToolSchema:
        """Generate default OpenAI function schema for GUI sandbox.

        This schema is used for tool_registry compatibility. Note that GUI agents
        don't use tool_call parsing - actions are raw text passed directly to sandbox.

        Returns:
            OpenAIFunctionToolSchema with gui_sandbox function definition.
        """
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="gui_sandbox",
                description="Execute action in GUI sandbox environment. Returns screenshot after execution.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "action": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Raw action string to execute (e.g., CLICK(500, 300), TYPE('hello'))",
                        )
                    },
                    required=["action"],
                ),
            ),
        )

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None = None):
        """Initialize the OS Sandbox tool.

        Args:
            config: Configuration dictionary with tool settings.
            tool_schema: Optional tool schema for BaseTool compatibility.
                        If not provided, a default schema is generated.
        """
        # Generate default schema if not provided (backward compatible)
        if tool_schema is None:
            tool_schema = self._default_tool_schema()

        # Initialize BaseTool (sets self.config, self.tool_schema, self.name)
        # Note: BaseTool prints schema to console - we suppress this for cleaner logs
        self.config = config
        self.tool_schema = tool_schema
        self.name = self.tool_schema.function.name

        # OSSandboxTool specific initialization
        self.max_steps = config.get("max_steps", 20)
        self.sandbox_url = config.get("sandbox_url", "http://localhost:9000")
        self.create_timeout = config.get("create_timeout", 300.0)  # 5 minutes for VM startup
        self.execute_timeout = config.get("execute_timeout", 60.0)  # 1 minute for action execution

        # Sandbox client with configurable timeouts
        self._client = SandboxClient(
            self.sandbox_url, create_timeout=self.create_timeout, execute_timeout=self.execute_timeout
        )

        # Instance state storage (instance_id -> SandboxInstance)
        self._instance_dict: dict[str, SandboxInstance] = {}

        logger.info(
            f"Initialized OSSandboxTool: "
            f"name={self.name}, "
            f"max_steps={self.max_steps}, "
            f"sandbox_url={self.sandbox_url}, "
            f"create_timeout={self.create_timeout}s, "
            f"execute_timeout={self.execute_timeout}s"
        )

    def _decode_screenshot(self, b64_data: str) -> Image.Image:
        """Decode base64 screenshot to PIL Image.

        Args:
            b64_data: Base64 encoded image data

        Returns:
            PIL Image
        """
        image_data = base64.b64decode(b64_data)
        return Image.open(io.BytesIO(image_data))

    async def create(
        self,
        instance_id: str | None = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        """Create a tool instance for a trajectory.

        Args:
            instance_id: Unique instance ID (also used as Sandbox task_id).
                        Generated if not provided.
            **kwargs: Creation arguments. Supports both patterns:
                - create_kwargs={"task_config": {...}, ...}  (legacy)
                - task_config={...}, max_steps=20, ...  (direct kwargs)

                Required:
                - task_config: Task configuration dict from DataProto.non_tensor_batch,
                  containing config and evaluator fields.

        Returns:
            Tuple of (instance_id, initial_response with screenshot).
        """
        if instance_id is None:
            raise ValueError("instance_id is required")

        # Support both legacy create_kwargs pattern and direct kwargs
        create_kwargs = kwargs.pop("create_kwargs", None)
        if create_kwargs is not None:
            # Legacy pattern: create_kwargs={"task_config": {...}}
            merged_kwargs = {**create_kwargs, **kwargs}
        else:
            # Direct kwargs pattern: task_config={...}
            merged_kwargs = kwargs

        task_config = merged_kwargs.get("task_config", {})
        if not task_config:
            raise ValueError("task_config is required in kwargs or create_kwargs")

        # Get max_steps from kwargs (per-task) or fall back to config
        max_steps = merged_kwargs.get("max_steps", self.max_steps)
        # Get validate flag from kwargs (for cache management)
        validate = merged_kwargs.get("validate", False)

        # Create instance (instance_id is also used as Sandbox task_id)
        instance = SandboxInstance(
            instance_id=instance_id, task_config=task_config, extra_info=merged_kwargs.get("extra_info", {})
        )

        # Create sandbox task (use instance_id as task_id)
        try:
            result = await self._client.create(instance_id, task_config, max_steps=max_steps, validate=validate)

            if result.get("error"):
                raise RuntimeError(f"Sandbox returned error: {result['error']}")

            # Decode screenshot
            screenshot_b64 = result.get("screenshot")
            if not screenshot_b64:
                raise RuntimeError("Sandbox returned no screenshot")

            screenshot = self._decode_screenshot(screenshot_b64)

            # Extract screenshot_url (for lazy loading optimization)
            screenshot_url = result.get("screenshot_url")
            image_urls = [screenshot_url] if screenshot_url else None

            instance.state = InstanceState.READY
            self._instance_dict[instance_id] = instance

            instruction = task_config.get("instruction", f"Task: {instance_id}")

            return instance_id, ToolResponse(
                text=f"Environment initialized. {instruction}",
                image=[screenshot],
                image_urls=image_urls,
            )
        except Exception as e:
            import traceback

            error_msg = f"{type(e).__name__}: {e}" if str(e) else f"{type(e).__name__} (no message)"
            logger.error(f"[OSSandboxTool] Failed to create sandbox instance: {error_msg}")
            logger.error(f"[OSSandboxTool] Traceback:\n{traceback.format_exc()}")
            print(f"[OSSandboxTool ERROR] Create failed: {error_msg}")
            return instance_id, ToolResponse(text=f"Failed to initialize environment: {error_msg}")

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float | None, dict]:
        """Execute an action in the sandbox.

        Args:
            instance_id: The tool instance ID.
            parameters: Tool parameters with 'action' key.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (tool_response with screenshot, final_score, metrics).
            - final_score is None if task is not done
            - final_score is float (0.0-1.0) when task is done
        """
        instance = self._instance_dict.get(instance_id)
        if instance is None:
            # Return None for score - this is an error, not a scored result
            return ToolResponse(text=f"Error: Instance {instance_id} not found"), None, {"error": "instance_not_found"}

        action = parameters.get("action", "").strip()
        if not action:
            # Return None for score - this is an error, not a scored result
            return ToolResponse(text="Error: No action provided"), None, {"error": "no_action"}

        metrics = {
            "instance_id": instance_id,
            "step": instance.step_count,
            "action": action,
        }

        # Check state
        if instance.state not in [InstanceState.READY, InstanceState.RUNNING]:
            # Return None for score - this is an error, not a scored result
            return (
                ToolResponse(text=f"Error: Instance is in state '{instance.state.value}', cannot execute action"),
                None,
                {"error": "invalid_state"},
            )

        # Check if already done
        if instance.done:
            raise RuntimeError(f"Instance {instance_id} already done, cannot execute action")

        # Execute action
        try:
            result = await self._client.execute(instance.instance_id, action)

            # Check for errors
            error_msg = result.get("error")
            if error_msg and not result.get("done"):
                # max_steps_reached is a normal termination, not an error
                if error_msg == "max_steps_reached":
                    pass
                elif is_unrecoverable_error(error_msg):
                    instance.state = InstanceState.FAILED
                    raise TaskUnrecoverableError(
                        task_id=instance.instance_id,
                        error=error_msg,
                        step_count=result.get("step_count", instance.step_count),
                        instance_id=instance_id,
                    )
                elif is_system_unavailable_error(error_msg):
                    raise SystemUnavailableError(error_msg)
                else:
                    # Return None for score - this is an error, not a scored result
                    return (
                        ToolResponse(text=f"Action failed: {error_msg}"),
                        None,
                        {"error": "execution_failed", "details": error_msg},
                    )

            # Decode screenshot
            screenshot_b64 = result.get("screenshot")
            if screenshot_b64 is None:
                raise RuntimeError("screenshot missing in response")
            screenshot = self._decode_screenshot(screenshot_b64)

            # Update state
            instance.step_count = result.get("step_count", instance.step_count + 1)
            instance.trajectory.append(action)
            instance.done = result.get("done", False)
            instance.released = result.get("released", False)

            is_max_steps = error_msg == "max_steps_reached"
            should_terminate = instance.done or is_max_steps

            # Extract screenshot_url (for lazy loading optimization)
            screenshot_url = result.get("screenshot_url")
            image_urls = [screenshot_url] if screenshot_url else None

            if should_terminate:
                instance.state = InstanceState.COMPLETED
                instance.final_score = result.get("score")
                instance.verification_logs = result.get("verification_logs")
                metrics["done"] = instance.done
                metrics["score"] = instance.final_score
                metrics["step_count"] = instance.step_count
                metrics["verification_logs"] = instance.verification_logs
                if is_max_steps:
                    metrics["error"] = error_msg
                # Extract timing metrics from sandbox response
                for time_key in (
                    "execute_oagi_time",
                    "execute_finalize_time",
                    "sandbox_receive_time",
                    "sandbox_send_time",
                ):
                    if time_key in result:
                        metrics[time_key] = result[time_key]
                reason = "Task completed" if instance.done else "Max steps reached"
                # Cleanup: remove instance from dict to prevent memory leak
                self._instance_dict.pop(instance_id, None)
                return (
                    ToolResponse(
                        text=f"{reason}. Score: {instance.final_score}",
                        image=[screenshot],
                        image_urls=image_urls,
                    ),
                    instance.final_score,
                    metrics,
                )
            else:
                instance.state = InstanceState.RUNNING
                metrics["done"] = False
                metrics["step_count"] = instance.step_count
                # Extract timing metrics from sandbox response
                for time_key in (
                    "execute_oagi_time",
                    "execute_finalize_time",
                    "sandbox_receive_time",
                    "sandbox_send_time",
                ):
                    if time_key in result:
                        metrics[time_key] = result[time_key]
                return (
                    ToolResponse(
                        text=f"Executed. Step: {instance.step_count}/{self.max_steps}",
                        image=[screenshot],
                        image_urls=image_urls,
                    ),
                    None,
                    metrics,
                )

        except (TaskUnrecoverableError, SystemUnavailableError):
            raise
        except asyncio.TimeoutError as e:
            instance.state = InstanceState.FAILED
            metrics["error"] = "timeout"
            raise TaskUnrecoverableError(
                task_id=instance.instance_id,
                error="Execute operation timed out",
                step_count=instance.step_count,
                instance_id=instance_id,
            ) from e
        except Exception as e:
            error_str = str(e)
            if is_unrecoverable_error(error_str):
                instance.state = InstanceState.FAILED
                raise TaskUnrecoverableError(
                    task_id=instance.instance_id,
                    error=error_str,
                    step_count=instance.step_count,
                    instance_id=instance_id,
                ) from e
            logger.error(f"Action execution error: {e}")
            metrics["error"] = error_str
            return ToolResponse(text=f"Action execution error: {e}"), None, metrics

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release task instance (BaseTool compatible interface).

        Releases the runner on sandbox side and optionally clears screenshot cache.
        Called when VERL retries to clean up resources for old request_id.

        Args:
            instance_id: Instance ID (also used as task_id) to release.
            **kwargs: Additional arguments including:
                - clear_cache: Whether to also clear screenshot cache (default True)
        """
        # Extract clear_cache from kwargs (default True for backward compatibility)
        clear_cache = kwargs.get("clear_cache", True)

        # Clean up local instance state
        self._instance_dict.pop(instance_id, None)

        # Call sandbox client release (ignore return value for BaseTool compatibility)
        await self._client.release(instance_id, clear_cache=clear_cache)

    async def cleanup(self) -> None:
        """Cleanup all resources (call on shutdown)."""

        # Close client session
        await self._client.close()
