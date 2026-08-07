"""mini-swe-agent: a black-box agent launched inside the task sandbox."""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

_VENV = "/opt/mini-swe-agent-venv"
_DEFAULT_INSTALL_COMMAND = f"""
set -euo pipefail
python3 -m venv {_VENV}
{_VENV}/bin/pip install --disable-pip-version-check --no-cache-dir \
  "mini-swe-agent==2.2.8" "litellm==1.81.7"
""".strip()

_DRIVER_SOURCE = r"""from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    config_path, result_path = sys.argv[1:3]
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))

    from minisweagent.config import builtin_config_dir, get_config_from_spec

    defaults = get_config_from_spec(str(builtin_config_dir / "benchmarks" / "swebench.yaml"))

    from minisweagent.environments.local import LocalEnvironment

    environment_config = dict(defaults.get("environment", {}))
    environment_config.pop("environment_class", None)
    for key in (
        "image",
        "container_timeout",
        "run_args",
        "executable",
        "pull_timeout",
        "forward_env",
        "interpreter",
    ):
        environment_config.pop(key, None)
    environment_config.update(config.get("environment", {}))
    environment = LocalEnvironment(**environment_config)

    from minisweagent.models.litellm_model import LitellmModel

    model_config = dict(defaults.get("model", {}))
    model_config.pop("model_name", None)
    model_config.pop("model_kwargs", None)
    model_config.update(config["model"])
    model = LitellmModel(**model_config)

    from minisweagent.agents.default import DefaultAgent

    agent_config = dict(defaults.get("agent", {}))
    agent_config.update(config.get("agent", {}))
    agent = DefaultAgent(model, environment, **agent_config)

    try:
        info = agent.run(task=config["task"])
        result = {
            "exit_status": info.get("exit_status", "unknown"),
            "submission": info.get("submission", ""),
            "model_stats": {"instance_cost": agent.cost, "api_calls": agent.n_calls},
        }
    except Exception as exc:
        result = {"exit_status": type(exc).__name__, "submission": str(exc)}

    Path(result_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
"""


class MiniSweAgentConfig(AgentConfig):
    """Launch parameters for mini-swe-agent inside a task sandbox."""

    name: str = "mini_swe_agent"
    step_limit: int = Field(default=100, ge=1, description="Maximum mini-swe-agent steps per task.")
    action_timeout: float = Field(default=600.0, gt=0, description="Timeout for one sandbox action.")
    run_timeout: float = Field(default=7200.0, gt=0, description="Wall-clock timeout for the agent process.")
    install_timeout: float = Field(default=600.0, gt=0, description="Wall-clock timeout for dependency setup.")
    install_command: str = Field(
        default=_DEFAULT_INSTALL_COMMAND,
        description="Shell command that creates the isolated mini-swe-agent environment.",
    )


@register_agent("mini_swe_agent")
class MiniSweAgentAgent(Agent):
    """Install and run mini-swe-agent against the configured policy endpoint."""

    config_model = MiniSweAgentConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
    ) -> AgentResult:
        cfg: MiniSweAgentConfig = self.config  # type: ignore[assignment]
        base_url = cfg.model.base_url
        if not base_url:
            raise ValueError("mini_swe_agent: config.model.base_url is not set")
        system_prompt, task = self._split_messages(messages)

        install = await sandbox.exec_shell(cfg.install_command, timeout=cfg.install_timeout)
        if install.exit_code != 0:
            detail = (install.stderr or install.stdout or "unknown error").strip()[-2000:]
            raise RuntimeError(f"mini_swe_agent: install step failed: {detail}")

        run_id = uuid.uuid4().hex
        driver_path = f"/tmp/mini_swe_agent_{run_id}_run_agent.py"
        config_path = f"/tmp/mini_swe_agent_{run_id}_task.json"
        result_path = f"/tmp/mini_swe_agent_{run_id}_result.json"
        await sandbox.write_file(driver_path, _DRIVER_SOURCE)
        await sandbox.write_file(config_path, json.dumps(self._task_config(task, system_prompt), ensure_ascii=False))

        process = await sandbox.exec(
            [_VENV + "/bin/python", driver_path, config_path, result_path],
            timeout=cfg.run_timeout,
        )
        try:
            result_data = await sandbox.read_file(result_path)
            if isinstance(result_data, bytes):
                result_data = result_data.decode("utf-8")
            output = json.loads(result_data)
            if not isinstance(output, dict):
                raise TypeError(f"expected an object, got {type(output).__name__}")
            output["agent_stdout"] = process.stdout or ""
            info = {"exit_code": process.exit_code, "agent_stderr": process.stderr or ""}
        except Exception as exc:  # result corruption should not discard the trajectory
            logger.warning("mini_swe_agent: failed to read result file: %s", exc)
            output = {"exit_status": "runner_error", "submission": "", "agent_stdout": process.stdout or ""}
            info = {
                "exit_code": process.exit_code,
                "agent_stderr": process.stderr or "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        return AgentResult(output=output, info=info)

    @staticmethod
    def _split_messages(messages: list[dict[str, Any]]) -> tuple[str | None, str]:
        if len(messages) > 2:
            raise ValueError(f"mini_swe_agent accepts at most 2 messages (system?, user), got {len(messages)}")
        task = next((message.get("content") for message in messages if message.get("role") == "user"), None)
        if not task:
            raise ValueError("mini_swe_agent requires a 'user' message (the problem statement)")
        system_prompt = next(
            (message.get("content") for message in messages if message.get("role") == "system"),
            None,
        )
        return system_prompt, task

    def _task_config(self, task: str, system_prompt: str | None) -> dict[str, Any]:
        cfg: MiniSweAgentConfig = self.config  # type: ignore[assignment]
        model_name = cfg.model.model_name or "default"
        if not model_name.startswith("openai/"):
            model_name = f"openai/{model_name}"
        api_key = cfg.model.api_key if cfg.model.api_key and cfg.model.api_key != "EMPTY" else "not-needed"
        agent_config: dict[str, Any] = {"step_limit": cfg.step_limit}
        if system_prompt:
            agent_config["system_template"] = system_prompt
        return {
            "task": task,
            "agent": agent_config,
            "environment": {
                "cwd": "/testbed",
                "timeout": cfg.action_timeout,
                "env": {"GIT_PAGER": "cat", "PAGER": "cat"},
            },
            "model": {
                "model_name": model_name,
                "model_kwargs": {
                    "api_base": cfg.model.base_url,
                    "api_key": api_key,
                    "drop_params": True,
                    "temperature": cfg.model.temperature,
                    "top_p": cfg.model.top_p,
                },
                "cost_tracking": "ignore_errors",
            },
        }
