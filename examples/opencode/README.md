# OpenCode adapter prototype

This adapter launches OpenCode **1.19.3** inside an already provisioned Sandbox.
It accepts exactly one non-blank text user message and returns the CLI's JSON
events and generated text. The Task owns provisioning, verification, reward, and
Sandbox teardown. The adapter only launches the harness and removes its own
temporary state.

## Provisioning and configuration

Install the pinned CLI in the task image or a separately built runtime image:

```bash
npm install --prefix /opt/opencode --no-audit --no-fund opencode-ai@1.19.3
/opt/opencode/node_modules/.bin/opencode --version
```

Do this while building the image, not during an episode. The npm package selects a
platform binary; npm installation uses Node.js, while the installed native CLI
embeds its runtime. The development probe used macOS ARM64, Node.js 26.7.0 for
installation, Python 3.10.21, and the native OpenCode 1.19.3 binary. Linux runtime
images and their libc/architecture requirements still need deployment testing.
No minimum Node.js version is claimed here. Missing or mismatched binaries fail
before the prompt is submitted; the adapter does not install or upgrade them.

[agent.yaml](agent.yaml) is an **Agent fragment**, not a complete runnable Task or
training recipe. Compose it with an existing Task and Sandbox configuration.
Align the context/output limits with the served model. Set `model.base_url` to a
reachable OpenAI Chat Completions base URL, including `/v1`. A Gateway URL such as
`http://gateway:port/sessions/session-id/v1` is preserved, including the session
path. The standard framework runner performs any reverse-tunnel rewrite before
`Agent.run()`; the adapter creates no tunnel or Gateway session.

For a sandbox that already has the CLI and endpoint, the Agent API is:

```python
from uni_agent.agents.base import ModelConfig
from uni_agent.agents.opencode import OpenCodeAgent, OpenCodeConfig

agent = OpenCodeAgent(OpenCodeConfig(
    model=ModelConfig(base_url=session_base_url, model_name="policy", api_key="EMPTY"),
))
result = await agent.run(
    sandbox=sandbox,
    messages=[{"role": "user", "content": "Resolve the issue in this repository."}],
    workdir="/testbed",
)
```

`workdir` must be absolute and defaults to `/testbed`. System prompts, chat
histories, and multimodal messages are rejected before sandbox operations rather
than silently dropped. OpenCode supplies its normal build-agent system prompt.

## Protocol and results

The adapter selects a custom `uni-agent` provider using
`@ai-sdk/openai-compatible`. Its `options.baseURL` receives `ModelConfig.base_url`,
and `options.apiKey` references a dedicated environment variable. Thus the CLI
posts streaming requests to `<baseURL>/chat/completions`, rather than Responses
or Anthropic Messages. Model names containing a slash remain intact.

The launch uses `run --pure --format json --model uni-agent/<model> --agent build`
and passes the user text as one argument after `--`. It uses `Sandbox.exec(argv)`
without a shell. Only the configured provider is enabled. Title/summary agents,
automatic compaction, sharing, update checks, model-catalog downloads, project
config loading, and external plugins are disabled. The permissions deny
interactive questions, subagent dispatch, and web tools while allowing the build
agent's remaining tools inside the outer Sandbox. Tool permissions are not OS
or network isolation; use a disposable Sandbox with an appropriate network
policy. Task repositories may still supply repository instructions to OpenCode.

Each invocation gets a UUID directory with private HOME, XDG paths, database,
and temporary files. Database, test-home, permission, and workspace environment
overrides are set explicitly to avoid inheriting those OpenCode settings.
The adapter passes credentials in the execution environment, never in OpenCode's
argv or the JSON config. Sandbox backends may transport environment values
differently (for example, Docker's `--env`); the execution host must be trusted.
Returned events, transcript, and diagnostic tails are redacted
for the configured key, bearer values, and structured credential fields. The
entire private state directory is removed in `finally`, including on ordinary
task cancellation. A cleanup failure makes the result unfinished. The Sandbox
and CLI must be trusted with the process environment; this is not a secret vault.
Use a clean Sandbox environment: inherited OpenCode flags and system-managed
configuration are outside the adapter's full control.

`AgentResult.output` holds `text` and `events`; `transcript` includes the input and
generated assistant text. `info` holds the exit code, fixed CLI version, error
category, and redacted diagnostic tails. `finished=True` requires exit code 0,
a valid single-session event stream with no error event, and a final
`step_finish` event whose reason is `stop`. Truncated JSON, missing completion,
length limits, CLI errors, provider errors, and Sandbox timeout code `-1` do not
count as completion. `finished` describes harness completion, not task success.

The execution deadline is passed to `Sandbox.exec`. A provider's `TimeoutError`
also returns an unfinished result; `CancelledError` propagates after cleanup.
Terminating the CLI and its tool subprocesses remains a Sandbox-provider
responsibility. In particular, process-group cancellation in the baseline local
and Docker backends is outside this adapter change (see upstream PRs
[#234](https://github.com/verl-project/uni-agent/pull/234) and
[#199](https://github.com/verl-project/uni-agent/pull/199)). Do not treat the
prototype as validated cancellation containment or a production RL recipe.

## Validation and compatibility

CPU tests use a FakeSandbox and require no CLI, GPU, provider account, or model:

```bash
python -m pytest tests/uni_agent/agents/test_opencode_agent.py -q
```

A separate development probe ran the **real 1.19.3 CLI** through LocalSandbox
against a local synthetic Chat Completions SSE server. Text completion, a
harmless bash tool call followed by completion, and an HTTP 401 failure were
observed. The tool scenario made two requests to the configured session path;
the text scenario made one. These are CLI/protocol observations with scripted
responses, not real model evaluation or a Uni-Agent Gateway integration run.

Actual Gateway trajectory capture, tokenizer/tool-parser behavior, reverse
tunnels, remote Sandbox execution, long-context behavior, GPU inference, and RL
training remain unverified. No public API migration is required. Other OpenCode
versions deliberately fail preflight until their command/config/event contracts
are verified and the adapter's pinned version is updated.

## Sources

- [OpenCode CLI documentation](https://opencode.ai/docs/cli/)
- [OpenCode custom provider documentation](https://opencode.ai/docs/providers/#custom-provider)
- [Pinned run implementation](https://github.com/anomalyco/opencode/blob/e3c4eb242585dcd77be1a867391b7c25fd47ce52/packages/opencode/src/cli/cmd/run.ts)
- [Pinned config schema](https://github.com/anomalyco/opencode/blob/e3c4eb242585dcd77be1a867391b7c25fd47ce52/packages/core/src/v1/config/config.ts)
- [Pinned provider schema](https://github.com/anomalyco/opencode/blob/e3c4eb242585dcd77be1a867391b7c25fd47ce52/packages/core/src/v1/config/provider.ts)
- [Pinned environment flags](https://github.com/anomalyco/opencode/blob/e3c4eb242585dcd77be1a867391b7c25fd47ce52/packages/core/src/flag/flag.ts)
- [Uni-Agent Q4 roadmap](https://github.com/verl-project/uni-agent/issues/191)
