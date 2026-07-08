"""mini-swe-agent: a black-box agent launched *inside* the sandbox.

Like :mod:`~uni_agent.agents.claude_code`, mini-swe-agent owns its own loop and
tools (a single "bash" tool), so the host side only needs to: install it into
an isolated venv (kept off the task image's own Python/conda env so it can
never shadow the repo's dependencies), point it at ``config.model`` via
LiteLLM, launch it against ``/testbed``, and let the task's own reward step
(a plain ``git diff`` + eval harness run) score whatever it left on disk.

Reference: https://github.com/SWE-agent/mini-swe-agent
(``minisweagent.agents.default.DefaultAgent`` + ``minisweagent.models.litellm_model.LitellmModel``
+ ``minisweagent.environments.local.LocalEnvironment``).
"""

from __future__ import annotations

import json
import logging
import shlex
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from ...sandbox import Sandbox

logger = logging.getLogger(__name__)

#: Driver script written into the sandbox and run with the isolated venv's
#: interpreter. Takes ``<config_path> <result_path>`` and never touches
#: stdout for its result (stdout is polluted by litellm/rich logging) --
#: it writes a clean JSON result file instead.
_RUN_AGENT_SCRIPT = '''\
import json
import os
import sys
import traceback

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")


def main() -> None:
    config_path, result_path = sys.argv[1], sys.argv[2]
    config = json.loads(open(config_path, encoding="utf-8").read())
    result: dict = {"exit_status": "error", "submission": "", "model_stats": {}}
    try:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.config import builtin_config_dir, get_config_from_spec
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.litellm_model import LitellmModel

        base = get_config_from_spec(str(builtin_config_dir / "benchmarks" / (config["config_spec"] + ".yaml")))

        env_cfg = {**base.get("environment", {}), **config.get("environment", {})}
        env_cfg.pop("environment_class", None)
        env = LocalEnvironment(**env_cfg)

        model_cfg = {**base.get("model", {}), **config.get("model", {})}
        model = LitellmModel(**model_cfg)

        agent_cfg = {**base.get("agent", {}), **config.get("agent", {})}
        agent = DefaultAgent(model, env, **agent_cfg)

        info = agent.run(task=config["task"])
        result = {
            "exit_status": info.get("exit_status", "unknown"),
            "submission": info.get("submission", ""),
            "model_stats": {"instance_cost": agent.cost, "api_calls": agent.n_calls},
        }
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
    finally:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
'''


class MiniSweAgentConfig(AgentConfig):
    """Black-box launch params for mini-swe-agent (endpoint lives on :attr:`AgentConfig.model`)."""

    name: str = "mini_swe_agent"

    venv_dir: str = Field(
        default="/opt/mini-swe-agent-venv",
        description="Isolated venv inside the sandbox mini-swe-agent runs from, kept off the "
        "task image's own Python/conda env so installing it can't shadow repo deps.",
    )
    install_command: str | None = Field(
        default=None,
        description="Shell run once before launch to ensure mini-swe-agent is on the venv; "
        "None auto-generates 'create venv + pip install' rooted at venv_dir (skipped if already present).",
    )
    config_spec: str = Field(
        default="swebench",
        description="mini-swe-agent builtin config name (under minisweagent/config/benchmarks/) "
        "supplying the default system/instance templates and agent/environment settings.",
    )
    cwd: str = Field(default="/testbed", description="Working directory the agent's shell commands run in.")
    step_limit: int = Field(default=50, description="Max agent turns (0 = unlimited; overrides config_spec's).")
    cost_limit: float = Field(
        default=0.0,
        description="Stop after exceeding this cost (0 = unlimited). Cost tracking is disabled by default "
        "(the served policy model is unpriced), so 0 is the only value that means anything in practice.",
    )
    wall_time_limit_seconds: int = Field(default=0, description="Stop after this many seconds (0 = unlimited).")
    command_timeout: int = Field(default=120, description="Per-shell-command timeout (s) inside the agent's loop.")
    run_timeout: float | None = Field(
        default=None,
        description="Overall cap (s) on the whole run; None defers to the sandbox's own runtime_timeout "
        "(same policy as claude_code).",
    )
    extra_env: dict[str, str] = Field(
        default_factory=lambda: {
            "PAGER": "cat",
            "MANPAGER": "cat",
            "LESS": "-R",
            "PIP_PROGRESS_BAR": "off",
            "TQDM_DISABLE": "1",
            "GIT_PAGER": "cat",
        },
        description="Extra env vars for the agent's shell commands (e.g. to silence pagers/progress bars).",
    )


@register_agent("mini_swe_agent")
class MiniSweAgentAgent(Agent):
    """Black-box solver: launch mini-swe-agent in the sandbox, pointed at ``config.model``."""

    config_model = MiniSweAgentConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
    ) -> AgentResult:
        cfg: MiniSweAgentConfig = self.config  # type: ignore[assignment]
        if cfg.model.base_url is None:
            raise ValueError("mini_swe_agent: config.model.base_url is not set (the endpoint the agent calls)")
        # mini-swe-agent owns its own loop + prompt templates, so we can only seed it: a
        # required user turn (the problem statement) and at most one optional system turn.
        if len(messages) > 2:
            raise ValueError(f"mini_swe_agent accepts at most 2 messages (system?, user), got {len(messages)}")
        problem_statement = next((m["content"] for m in messages if m.get("role") == "user"), None)
        if not problem_statement:
            raise ValueError("mini_swe_agent requires a 'user' message (the problem statement)")
        system_prompt = next((m["content"] for m in messages if m.get("role") == "system"), None)

        venv_python = f"{cfg.venv_dir}/bin/python"
        install_command = cfg.install_command or (
            f"test -x {venv_python} || "
            f"(python3 -m venv {shlex.quote(cfg.venv_dir)} && "
            f"{venv_python} -m pip install -q -U pip mini-swe-agent litellm)"
        )
        res = await sandbox.exec_shell(install_command)
        if res.exit_code != 0:
            raise RuntimeError(
                f"mini_swe_agent install step failed (exit {res.exit_code}); "
                f"ensure python3/venv are available in the sandbox image. stderr: {res.stderr.strip()}"
            )

        run_dir = f"/tmp/mini_swe_agent_{uuid.uuid4().hex}"
        driver_path = f"{run_dir}/run_agent.py"
        config_path = f"{run_dir}/task.json"
        result_path = f"{run_dir}/result.json"
        await sandbox.write_file(driver_path, _RUN_AGENT_SCRIPT)

        model_name = f"openai/{cfg.model.model_name}" if cfg.model.model_name else "openai/default"
        model_kwargs: dict[str, Any] = {
            "api_base": cfg.model.base_url,
            "api_key": cfg.model.api_key,
            "drop_params": True,
            "temperature": cfg.model.temperature,
            "top_p": cfg.model.top_p,
        }
        if cfg.model.max_tokens_per_turn is not None:
            model_kwargs["max_tokens"] = cfg.model.max_tokens_per_turn
        task_config: dict[str, Any] = {
            "config_spec": cfg.config_spec,
            "task": problem_statement,
            "model": {
                "model_name": model_name,
                "model_kwargs": model_kwargs,
                # The served policy model is unpriced from litellm's point of view; don't let a
                # cost-lookup failure crash an otherwise-successful episode.
                "cost_tracking": "ignore_errors",
            },
            "agent": {
                "step_limit": cfg.step_limit,
                "cost_limit": cfg.cost_limit,
                "wall_time_limit_seconds": cfg.wall_time_limit_seconds,
                **({"system_template": system_prompt} if system_prompt else {}),
            },
            "environment": {
                "cwd": cfg.cwd,
                "timeout": cfg.command_timeout,
                "env": cfg.extra_env,
            },
        }
        await sandbox.write_file(config_path, json.dumps(task_config))

        # No client-side timeout by default: the sandbox's runtime_timeout bounds the run
        # (same policy as claude_code); set cfg.run_timeout to cap it explicitly instead.
        proc = await sandbox.exec([venv_python, driver_path, config_path, result_path], timeout=cfg.run_timeout)

        try:
            result = json.loads((await sandbox.read_file(result_path)).decode("utf-8"))
        except Exception as exc:  # missing/corrupt result file (e.g. driver crashed before writing it)
            logger.warning("mini_swe_agent: failed to read result file %s: %s", result_path, exc)
            result = {"exit_status": "runner_error", "submission": "", "error": str(exc)}

        info: dict[str, Any] = {"model_stats": result.get("model_stats", {})}
        if "error" in result:
            info["error"] = result["error"]
        return AgentResult(
            output={
                "exit_status": result.get("exit_status"),
                "submission": result.get("submission", ""),
                "agent_stdout": proc.stdout,
                "exit_code": proc.exit_code,
            },
            info=info,
        )
