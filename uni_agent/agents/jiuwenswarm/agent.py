"""jiuwenswarm: a black-box agent that runs the jiuwenswarm CLI inside the sandbox.

jiuwenswarm is launched *in* the sandbox from a prebuilt tool image (mounted at
``/opt/jiuwenswarm``) whose ``bin/run_agent.sh`` reads the task prompt from
**stdin** and the gateway config from **env vars**. This host-side agent encodes
the prompt via base64, pipes it in, and parses the resulting JSON out of stdout.

Reference: https://gitcode.com/openJiuwen/jiuwenswarm
"""

from __future__ import annotations

import base64
import json
import logging
import shlex
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)


def build_agent_command(
    *,
    task_b64: str,
    tool_script: str,
    gateway_url: str,
    model_name: str = "unknown",
    api_key: str = "EMPTY",
    project_dir: str = "/testbed",
    conda_env_path: str | None = None,
) -> str:
    """Build the shell command that runs ``run_agent.sh`` inside the sandbox.

    The task prompt is piped via base64-encoded stdin. The gateway config
    (model endpoint, API key, project dir) is passed as env vars so
    ``run_agent.sh`` can write the ``.env`` file for ``jiuwenswarm-app``.
    When ``conda_env_path`` is set, the tool bash is called through that conda
    env so jiuwenswarm's tool subprocesses resolve the repo environment inside
    the project dir.
    """
    conda_env_vars = ""
    if conda_env_path:
        env_dir = PurePosixPath(conda_env_path)
        conda_env_vars = (
            f"CONDA_DEFAULT_ENV={shlex.quote(env_dir.name)} "
            f"CONDA_PREFIX={shlex.quote(str(env_dir))} "
            f"PATH={shlex.quote(str(env_dir / 'bin'))}:"
            f"{shlex.quote(str(env_dir.parent.parent / 'bin'))}:$PATH "
        )
    run_agent_env = (
        conda_env_vars
        + f"JWS_API_BASE={shlex.quote(gateway_url)} "
        f"JWS_MODEL_NAME={shlex.quote(model_name)} "
        f"JWS_API_KEY={shlex.quote(api_key)} "
        f"JWS_PROJECT_DIR={shlex.quote(project_dir)} "
        "PIP_DISABLE_PIP_VERSION_CHECK=1 "
        "PIP_PROGRESS_BAR=off"
    )
    return (
        "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy; "
        f"printf %s {shlex.quote(task_b64)} | base64 -d | "
        f"env {run_agent_env} bash {shlex.quote(tool_script)}"
    )


def parse_agent_result(stdout: str, exit_code: int) -> dict[str, Any]:
    """Parse the result JSON from ``run_agent.sh``'s stdout.

    ``exit_code == -1`` means the sandbox ``exec_shell`` timed out. Otherwise
    the last line starting with ``{`` wins (anti-noise). The jiuwenswarm CLI's
    ``--json`` output carries ``ok``, ``content``, and optionally ``error``;
    we inject ``exit_status`` from ``ok``.
    """
    if exit_code == -1:
        return {"exit_status": "timeout", "content": "", "error": "agent process timed out"}
    stdout = stdout.strip()
    if not stdout:
        return {"exit_status": "error", "content": "", "error": "empty stdout"}
    for line in reversed([ln.strip() for ln in stdout.split("\n") if ln.strip()]):
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
                if "ok" in parsed:
                    parsed.setdefault("exit_status", "ok" if parsed.get("ok") else "error")
                    return parsed
            except json.JSONDecodeError:
                continue
    try:
        parsed = json.loads(stdout)
        if "ok" in parsed:
            parsed.setdefault("exit_status", "ok" if parsed.get("ok") else "error")
            return parsed
    except json.JSONDecodeError:
        logger.warning("jiuwenswarm: failed to parse agent result (stdout tail): %.1000s", stdout)
    return {"exit_status": "error", "content": "", "error": "unparseable stdout"}


class JiuwenswarmConfig(AgentConfig):
    """Black-box launch params for jiuwenswarm (endpoint lives on :attr:`AgentConfig.model`)."""

    name: str = "jiuwenswarm"
    run_timeout: float = Field(default=7200.0, description="Wallclock cap (s) on the agent process.")
    conda_env_path: str | None = Field(
        default=None,
        description="Task-image path of the conda env to activate around the launch "
        "(e.g. /opt/miniconda3/envs/testbed); no conda env is activated when unset.",
    )
    # Tool-image path is bound to the prebuilt tool image's Dockerfile layout
    # (mounted at /opt/jiuwenswarm), so it is required here and declared by
    # the recipe's task config instead of being hardcoded as a default.
    tool_script: str = Field(
        description="In-sandbox entrypoint path (e.g. /opt/jiuwenswarm/bin/run_agent.sh)."
    )


@register_agent("jiuwenswarm")
class JiuwenswarmAgent(Agent):
    """Black-box solver: run jiuwenswarm in the sandbox against ``config.model``."""

    config_model = JiuwenswarmConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
        workdir: str | None = None,
    ) -> AgentResult:
        cfg: JiuwenswarmConfig = self.config  # type: ignore[assignment]
        if cfg.model.base_url is None:
            raise ValueError("jiuwenswarm: config.model.base_url is not set (the gateway/vLLM policy endpoint)")
        task = self._extract_task(messages)
        project_dir = workdir or "/testbed"

        # 1) Base64-encode the task prompt. The agent is tunnel-agnostic: when
        #    a reverse tunnel is configured, run_task has already rewritten
        #    model.base_url to http://127.0.0.1:<proxy_port>, so it passes
        #    through as-is.
        task_b64 = base64.b64encode(task.encode()).decode()

        # 2) Run the in-sandbox bash entrypoint (task via stdin, config via env).
        agent_cmd = build_agent_command(
            task_b64=task_b64,
            tool_script=cfg.tool_script,
            gateway_url=cfg.model.base_url,
            model_name=cfg.model.model_name or "unknown",
            api_key=cfg.model.api_key or "EMPTY",
            project_dir=project_dir,
            conda_env_path=cfg.conda_env_path,
        )
        result = await sandbox.exec_shell(agent_cmd, timeout=cfg.run_timeout, workdir=project_dir)

        # 3) Parse the result JSON from stdout (exit_code -1 == timeout).
        agent_info = parse_agent_result(result.stdout or "", result.exit_code)
        logger.info(
            "jiuwenswarm: done exit_status=%s ok=%s exit_code=%s",
            agent_info.get("exit_status"),
            agent_info.get("ok"),
            result.exit_code,
        )
        return AgentResult(
            output=agent_info,
            transcript=list(messages),
            info={
                "exit_status": agent_info.get("exit_status"),
                "ok": agent_info.get("ok"),
            },
            # CLI produced valid JSON (has "ok") → trajectory is complete
            finished="ok" in agent_info and result.exit_code != -1,
        )

    @staticmethod
    def _extract_task(messages: list[dict[str, Any]]) -> str:
        if len(messages) > 2:
            raise ValueError(f"jiuwenswarm accepts at most 2 messages (system?, user), got {len(messages)}")
        problem = next((m["content"] for m in messages if m.get("role") == "user"), None)
        if not problem:
            raise ValueError("jiuwenswarm requires a 'user' message (the problem statement)")
        return problem