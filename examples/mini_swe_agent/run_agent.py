#!/opt/mini-swe-agent/bin/python
"""Run mini-swe-agent inside the sandbox.

Input:  task config JSON from **stdin**
    - task: str — the issue description for the agent to solve
    - gateway_url: str — LLM gateway endpoint (tunnel URL for remote sandbox)
    - agent: dict — agent config (e.g. step_limit)

Output: agent result JSON to **stdout**, or error JSON on failure
"""

from __future__ import annotations

import json
import sys

DEFAULT_ACTION_TIMEOUT = 600


def _fail(msg: str, exit_status: str = "error") -> None:
    """Write error result to stdout and exit."""
    sys.stdout.write(json.dumps({"exit_status": exit_status, "submission": "", "error": msg}))
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> None:
    try:
        # 1. Read task config from stdin
        config = json.load(sys.stdin)
        task = config["task"]
        gateway_url = config["gateway_url"]

        # 2. Load swebench defaults
        from minisweagent.config import builtin_config_dir, get_config_from_spec

        swebench_cfg = get_config_from_spec(str(builtin_config_dir / "benchmarks" / "swebench.yaml"))

        # 3. Create LocalEnvironment (use swebench defaults)
        from minisweagent.environments.local import LocalEnvironment

        env_cfg = dict(swebench_cfg.get("environment", {}))
        env_cfg.pop("environment_class", None)
        env_cfg["timeout"] = DEFAULT_ACTION_TIMEOUT
        env_cfg.setdefault("env", {})
        env_cfg["env"].setdefault("GIT_PAGER", "cat")
        for key in (
            "image",
            "container_timeout",
            "run_args",
            "executable",
            "pull_timeout",
            "forward_env",
            "interpreter",
        ):
            env_cfg.pop(key, None)
        env = LocalEnvironment(**env_cfg)

        # 4. Create LitellmModel pointing at gateway
        from minisweagent.models.litellm_model import LitellmModel

        model_defaults = dict(swebench_cfg.get("model", {}))
        model_defaults.pop("model_name", None)
        model_defaults.pop("model_kwargs", None)
        model_cfg = model_defaults
        model_cfg.update(
            {
                "model_name": "openai/default",
                "model_kwargs": {
                    "api_base": gateway_url,
                    "api_key": "not-needed",
                    "drop_params": True,
                },
                "cost_tracking": "ignore_errors",
            }
        )
        model = LitellmModel(**model_cfg)

        # 5. Create DefaultAgent
        from minisweagent.agents.default import DefaultAgent

        agent_defaults = dict(swebench_cfg.get("agent", {}))
        agent_overrides = config.get("agent", {})
        agent_defaults.update(agent_overrides)
        agent_cfg = agent_defaults
        step_limit = agent_cfg.get("step_limit", 100)
        agent_cfg["step_limit"] = step_limit
        agent = DefaultAgent(model, env, **agent_cfg)

        # 6. Run agent
        try:
            info = agent.run(task=task)
        except Exception as e:
            info = {"exit_status": type(e).__name__, "submission": str(e)}

        # 7. Write result to stdout
        result = {
            "exit_status": info.get("exit_status", "unknown"),
            "submission": info.get("submission", ""),
            "model_stats": {
                "instance_cost": agent.cost,
                "api_calls": agent.n_calls,
            },
        }
        sys.stdout.write(json.dumps(result, ensure_ascii=False))
        sys.stdout.write("\n")
        sys.stdout.flush()

    except Exception as e:
        _fail(str(e), exit_status=type(e).__name__)


if __name__ == "__main__":
    main()
