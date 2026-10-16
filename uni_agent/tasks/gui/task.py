"""GUI task: the agent operates a desktop app in the sandbox; a shell command scores the end state.

The sample says what to do (``metadata["instruction"]``), how to set the desktop up
(``setup_command``) and how to check the result (``verify_command``: exit 0 means solved).
Pair it with an agent that exposes the ``gui_*`` tools and a sandbox image with an X
server, ``xdotool`` and ImageMagick (see ``examples/gui``).
"""

from __future__ import annotations

import logging

from pydantic import Field, model_validator

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task

logger = logging.getLogger(__name__)


class GuiTaskConfig(TaskConfig):
    name: str = "gui"
    display: str = Field(default=":0", description="X display the desktop runs on.")
    setup_command: str | None = Field(
        default=None,
        description="Shell command run before the agent, e.g. to launch the app under test. Must exit 0.",
    )
    verify_command: str = Field(
        description="Shell command run after the agent; exit 0 means the task is solved.",
    )
    setup_timeout: float = Field(default=60.0, gt=0, description="Setup command timeout (s).")
    verify_timeout: float = Field(default=60.0, gt=0, description="Verify command timeout (s).")

    @model_validator(mode="after")
    def _share_display_with_gui_tools(self) -> GuiTaskConfig:
        """Point the agent's ``gui_*`` tools at the task display, rejecting a conflicting one."""
        tools = getattr(self.agent, "tools", None)
        if not isinstance(tools, list):
            return self
        resolved = []
        for entry in tools:
            if isinstance(entry, dict) and str(entry.get("name", "")).startswith("gui_"):
                if entry.get("display", self.display) != self.display:
                    raise ValueError(
                        f"tool {entry['name']!r} uses display {entry['display']!r} "
                        f"but the task display is {self.display!r}"
                    )
                entry = {**entry, "display": self.display}
            resolved.append(entry)
        self.agent.tools = resolved
        return self


@register_task("gui")
class GuiTask(Task):
    name = "gui"
    config_model = GuiTaskConfig

    async def run(self) -> TaskResult:
        cfg: GuiTaskConfig = self.config  # type: ignore[assignment]
        env = {"DISPLAY": cfg.display}
        messages = cfg.prompt or [{"role": "user", "content": str(cfg.metadata.get("instruction", ""))}]
        async with self.build_sandbox() as sandbox:
            if cfg.setup_command:
                setup = await sandbox.exec_shell(cfg.setup_command, timeout=cfg.setup_timeout, env=env)
                if setup.exit_code != 0:
                    raise RuntimeError(f"gui setup_command failed (exit {setup.exit_code}): {setup.stderr[-500:]}")

            agent_result = await self.build_agent().run(sandbox=sandbox, messages=messages)

            verify = await sandbox.exec_shell(cfg.verify_command, timeout=cfg.verify_timeout, env=env)

        solved = verify.exit_code == 0
        logger.info(f"gui task done: solved={solved} verify_exit_code={verify.exit_code}")
        return TaskResult(
            reward=float(solved),
            accuracy=float(solved),
            finished=agent_result.finished,
            extra_info={
                "solved": solved,
                "verify_exit_code": verify.exit_code,
                "verify_stderr": verify.stderr[-500:],
                "steps": agent_result.info.get("steps", 0),
                "agent_error": agent_result.info.get("error"),
            },
        )
