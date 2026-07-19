# Agent

An Agent defines who solves a Task and how the solving loop runs. Every Agent receives a started Sandbox, an OpenAI-style message list, and a runtime-injected model endpoint.

Uni-Agent supports two integration styles:

- **White-box Agent:** Uni-Agent owns the interaction loop, model calls, Tool dispatch, and transcript.
- **Black-box Agent:** an external harness owns the loop and runs as a process inside the Sandbox.

## Configuration

All Agent configurations inherit from `AgentConfig`:

```python
class AgentConfig(BaseModel):
    name: str
    model: ModelConfig
```

`ModelConfig` contains:

- `base_url`, `api_key`, and `model_name`.
- `temperature`, `top_p`, and `top_k`.
- Per-turn and episode token budgets.

The model endpoint is runtime state. Dataset rows and Task YAML should define sampling behavior, but the live runner or Gateway injects `base_url`, credentials, and served model name last.

## Agent Contract

Every Agent implements:

```python
async def run(
    *,
    sandbox: Sandbox,
    messages: list[dict],
) -> AgentResult:
    ...
```

The Task has already started the Sandbox. The Agent must not stop it.

`AgentResult` can carry:

- `output`: final structured output.
- `transcript`: messages and Tool observations produced by a white-box loop.
- `info`: implementation-specific metadata such as exit reason, token counts, or process status.

The Task decides how the resulting Sandbox state and AgentResult are scored.

## White-Box Agent

The built-in ReAct Agent demonstrates the white-box pattern:

1. Build a Toolbox from configured Tool specs.
2. Send messages and Tool schemas to an OpenAI-compatible model endpoint.
3. Execute returned Tool calls.
4. Append Tool observations to the transcript.
5. Stop on a plain assistant answer, `submit`/`finish`, token limit, timeout budget, or max steps.

ReAct configuration exposes the loop:

```yaml
agent:
  name: react
  max_steps: 200
  action_timeout: 300
  tools:
    - name: stateful_shell
    - name: str_replace_editor
    - name: submit
  model:
    temperature: 0.8
    top_p: 0.9
    max_total_tokens: 65536
```

Use this style when you need complete control over Tool schemas, observations, transcripts, and stopping behavior.

## Black-Box Agent

The Claude Code Agent demonstrates the black-box pattern:

1. Ensure the external CLI is installed inside the Sandbox.
2. Convert `ModelConfig` into the environment variables expected by the harness.
3. Launch the harness through `sandbox.exec()`.
4. Return process metadata while the Task evaluates the modified Sandbox.

```yaml
agent:
  name: claude_code
  max_turns: 200
  run_timeout: 4800
  model:
    temperature: 1.0
    top_p: 0.95
    max_total_tokens: 131072
```

Claude Code speaks the Anthropic Messages protocol. Uni-Agent sets `ANTHROPIC_BASE_URL` to either a direct model endpoint or a session-scoped Gateway endpoint.

Use this style when an existing Agent Harness already owns its loop and Tools.

## Custom Agent

Create an Agent-specific configuration and implementation:

```python
from typing import Any

from pydantic import Field

from uni_agent.agents.base import Agent, AgentConfig, AgentResult
from uni_agent.agents.registry import register_agent


class MyAgentConfig(AgentConfig):
    name: str = "my_agent"
    run_timeout: float = Field(default=1800)


@register_agent("my_agent")
class MyAgent(Agent):
    config_model = MyAgentConfig

    async def run(
        self,
        *,
        sandbox,
        messages: list[dict[str, Any]],
    ) -> AgentResult:
        config: MyAgentConfig = self.config
        result = await sandbox.exec(
            ["my-agent", "--endpoint", config.model.base_url],
            timeout=config.run_timeout,
        )
        return AgentResult(
            info={
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
```

Register the lazy module:

```python
AGENT_MODULES["my_agent"] = "my_package.agent"
```

`TaskConfig` resolves an Agent mapping through this registry, so Agent-specific fields remain typed and validated.

## Implementation Rules

- Validate required model endpoint fields before starting work.
- Treat the Sandbox as already started and Task-owned.
- Return `AgentResult` even when the Task primarily scores filesystem changes.
- Keep Agent-specific launch settings in the Agent config; keep task metadata and reward logic in the Task.
- For white-box loops, use Toolbox lifecycle management and preserve a useful transcript.
- For black-box harnesses, map model settings explicitly and enforce process timeouts.
- Register the class and add a lazy module entry so optional dependencies remain isolated.
