"""Copy to uni_agent/agents/<your_tool>/agent.py and rename the registration."""

from __future__ import annotations

import json
import shlex
import uuid

from pydantic import Field

from uni_agent.agents import Agent, AgentConfig, AgentResult
from uni_agent.agents.registry import register_agent


class ToolConfig(AgentConfig):
    name: str = "my_tool"
    command: list[str] = Field(min_length=1)
    run_timeout: float = Field(default=600, gt=0)


@register_agent("my_tool")
class ToolAgent(Agent):
    config_model = ToolConfig

    async def run(self, *, sandbox, messages, workdir=None):
        cfg = self.config
        request = {
            "messages": messages,
            "model": cfg.model.model_dump(exclude_none=True),
            "workdir": workdir or "/testbed",
        }
        directory = f"/tmp/sidecar-request-{uuid.uuid4().hex}"
        created = await sandbox.exec_shell(f"mkdir -m 700 {shlex.quote(directory)}", timeout=30)
        if created.exit_code:
            raise RuntimeError("Could not create the private sidecar request directory")
        path = f"{directory}/input.json"
        try:
            await sandbox.write_file(path, json.dumps(request))
            result = await sandbox.exec_shell(
                f"{shlex.join(cfg.command)} < {shlex.quote(path)}",
                timeout=cfg.run_timeout,
                workdir=request["workdir"],
            )
        finally:
            await sandbox.exec_shell(f"rm -f {shlex.quote(path)}; rmdir {shlex.quote(directory)}", timeout=30)
        response = {}
        if result.exit_code == 0:
            try:
                response = json.loads(result.stdout)
            except ValueError:
                pass
        valid = isinstance(response, dict) and isinstance(response.get("output"), dict)
        return AgentResult(
            output=response["output"] if valid else {"error": "Invalid sidecar response or process failure"},
            transcript=list(messages),
            info={"exit_code": result.exit_code},
            finished=bool(valid and response.get("finished") is True and result.exit_code == 0),
        )
