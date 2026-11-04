# OpenCode adapter

This adapter launches stable OpenCode **1.18.x and 1.19.x** releases inside an
already provisioned Sandbox. The gate accepts numeric release triplets only;
prereleases, custom build suffixes, malformed versions, and other release lines
are rejected before a prompt is sent. `result.info["cli_version"]` records the
actual detected version (or `None` when preflight did not succeed).
The SWE-rebench training recipe in
[`examples/blackbox_recipes/opencode_swe_task`](../blackbox_recipes/opencode_swe_task/README.md)
runs the sidecar image on E2B.
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
installation, Python 3.10.21, and native OpenCode binaries. The representative
releases are 1.18.18 and 1.19.3; the integration workflow tests both through a
read-only sidecar on Ubuntu amd64. Other deployments still need validation.
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

`workdir` must be absolute and defaults to `/testbed`. The adapter passes it both
to `Sandbox.exec` and explicitly to OpenCode's `--dir`, so an inherited `PWD`
cannot select a different repository. System prompts, chat
histories, and multimodal messages are rejected before sandbox operations rather
than silently dropped. OpenCode supplies its normal build-agent system prompt.

## Protocol and results

The adapter selects a custom `uni-agent` provider using
`@ai-sdk/openai-compatible`. Its `options.baseURL` receives `ModelConfig.base_url`,
and `options.apiKey` references a dedicated environment variable. Thus the CLI
posts streaming requests to `<baseURL>/chat/completions`, rather than Responses
or Anthropic Messages. Model names containing a slash remain intact.

The launch uses `run --pure --format json --model uni-agent/<model> --agent build --dir <workdir>`
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

## Minimal Task lifecycle demo

For a complete, small local example, run [smoke_task.py](smoke_task.py) from the
repository root. It uses the real pinned CLI with a local **scripted** Chat
Completions server. No provider account, real model, Gateway, GPU, Ray, or RL
training is involved. The fixed tool response is a test fixture, not evidence
of model problem-solving ability.

In a Python 3.10–3.12 virtual environment, install the minimal Python dependencies
and explicitly select a provisioned OpenCode 1.19.3 executable:

```bash
python -m pip install -e . pydantic PyYAML
python -m examples.opencode.smoke_task \
  --opencode-bin /opt/opencode/node_modules/.bin/opencode
```

`PatchSmokeTask` prepares a temporary `calculator.py` whose `add` function is
incorrect, builds the Agent and LocalSandbox from a TaskConfig, and invokes the
CLI. The scripted response requests a relative-path edit. An independent Python
process checks two additions; TaskResult reports the score separately from
Agent completion. The Task then removes its workspace and private CLI state,
and the example closes its local HTTP server. This is a direct Task invocation;
it does not exercise the Framework runner or Gateway trajectory finalization.

The normal run exits 0 and prints `reward: 1.0`, `finished: true`,
`verifier_exit_code: 0`, `requests: 2`, and `workspace_removed: true`.
The negative control deliberately returns a final answer without editing:

```bash
python -m examples.opencode.smoke_task \
  --opencode-bin /opt/opencode/node_modules/.bin/opencode --scenario no-edit
```

This run **intentionally exits 1**, with `reward: 0.0`, `finished: true`,
`verifier_exit_code: 1`, `requests: 1`, and `workspace_removed: true`. A missing
or incompatible CLI also fails the example, rather than producing a passing
score. Unfinished Agent runs are not verified.

LocalSandbox executes on the host and does not provide container isolation.
The example serves only a fixed, short file-writing command and is intended for
this synthetic fixture. It does not fix or validate subprocess termination on
timeout/cancellation; the provider limitations above still apply. Linux image
deployment and real-model task success require separate validation.

## Validation and compatibility

CPU tests use a FakeSandbox and require no CLI, GPU, provider account, or model:

```bash
python -m pytest tests/uni_agent/agents/test_opencode_agent.py -q
```

A separate development probe ran the **real 1.19.3 CLI** through LocalSandbox
against a local synthetic Chat Completions SSE server. Text completion, a
harmless bash tool call followed by completion, an HTTP 401 failure, request
timeout, timeout after a tool process started, cancellation after a tool process
started, missing CLI, and wrong-version rejection were observed. The tool
scenario made two requests to the configured session path; the text scenario
made one. These use scripted responses, not real model evaluation.

An optional integration test also exercised the real local GatewayActor HTTP
routes, session creation/continuation, codec path, and trajectory finalization
with the real CLI and a deterministic synthetic CPU backend. The test substitutes
only a deterministic tool-call parser for the vLLM-specific parser. It verified
a tool-written temporary file, two backend generations, and a four-turn
trajectory. This does not validate remote Ray deployment, the production
tokenizer/parser stack, or real model inference. AgentResult.finished reflects
OpenCode harness completion; trajectory completion is owned by the Task or
Framework.

The Gateway integration probe also covers the pinned 2.2.5 adapter. With the
Gateway CPU dependencies and the checkout's pinned `verl` submodule installed,
run both probes using explicitly provisioned native CLIs:

```bash
OPENCODE_BIN=/path/to/opencode-1.19.3 OPENCODE_VERSION=1.19.3 \
OPENCODE2_BIN=/path/to/opencode-2.2.5 PYTHONPATH=. python -m pytest \
  tests/uni_agent/agents/test_opencode_gateway_integration.py -q -rs
```

For 1.18.18, select that binary and set `OPENCODE_VERSION=1.18.18`. Each probe
checks private-state removal after the Sandbox lifecycle closes. Missing CLI
opt-in or optional CPU dependencies produces an explicit skip; a passing
integration result requires no skips. These local probes do not exercise an
internal E2B template or real model.

The CLI/Gateway probes and real-CLI Task cases are disabled unless OPENCODE_BIN
is explicitly set. To run them against a provisioned OpenCode 1.19.3 executable (as a command
on PATH or an executable path):

```bash
OPENCODE_BIN=/opt/opencode/node_modules/.bin/opencode \
  python -m pytest \
  tests/uni_agent/agents/test_opencode_cli_probe.py \
  tests/uni_agent/agents/test_opencode_gateway_integration.py \
  tests/uni_agent/agents/test_opencode_task_smoke.py -q
```

The synthetic endpoints use no account, credential, GPU, or external model.
They are opt-in because they launch the native CLI and local subprocesses.
Cancellation probes currently expose the LocalSandbox child-process cleanup
limitation; exact test-owned PIDs are terminated by the probe after observation.
This does not validate process-group containment. Reverse tunnels, remote
Sandbox execution, long-context behavior, GPU inference, and RL training remain
unverified. No public API migration is required. Other OpenCode versions
deliberately fail preflight until their command/config/event contracts are
verified and the adapter's pinned version is updated.

The CLI probe deliberately inherits a `PWD` different from its workspace and
uses a relative tool write, including a workspace path containing spaces and a
quote. Task cases check positive and no-edit scores, workspace removal, and
missing-CLI cleanup. The missing-CLI case needs no OpenCode installation; it
still opens a temporary loopback server. Default CI does not provision the
native CLI, so a green CPU job alone does not establish black-box compatibility.

### Cross-PR process cleanup regression

The original CLI probe above records leaks and uses emergency PID cleanup; its
passing result is not a cleanup guarantee. The separate
[`test_opencode_process_cleanup.py`](../../tests/uni_agent/agents/test_opencode_process_cleanup.py)
checks request and active-tool timeout/cancellation with the real pinned CLI.
It asserts that tracked processes have exited **before** emergency cleanup, and
a passing case must send no rescue signals. See
[PROCESS_CLEANUP.md](PROCESS_CLEANUP.md) for the #292 + #234 comparison, the small
Sandbox follow-up, reproducible commands, and the remaining containment limits.

## Sources

- [OpenCode CLI documentation](https://opencode.ai/docs/cli/)
- [OpenCode custom provider documentation](https://opencode.ai/docs/providers/#custom-provider)
- [Pinned run implementation](https://github.com/anomalyco/opencode/blob/e3c4eb242585dcd77be1a867391b7c25fd47ce52/packages/opencode/src/cli/cmd/run.ts)
- [Pinned config schema](https://github.com/anomalyco/opencode/blob/e3c4eb242585dcd77be1a867391b7c25fd47ce52/packages/core/src/v1/config/config.ts)
- [Pinned provider schema](https://github.com/anomalyco/opencode/blob/e3c4eb242585dcd77be1a867391b7c25fd47ce52/packages/core/src/v1/config/provider.ts)
- [Pinned environment flags](https://github.com/anomalyco/opencode/blob/e3c4eb242585dcd77be1a867391b7c25fd47ce52/packages/core/src/flag/flag.ts)
- [Uni-Agent Q4 roadmap](https://github.com/verl-project/uni-agent/issues/191)


## Dual-version integration validation

Select a provisioned CLI with `OPENCODE_BIN` and its exact expected release with
`OPENCODE_VERSION` (default `1.19.3`). This selects test expectations, not the
production version gate. There is no monkeypatch or compatibility bypass:

```bash
OPENCODE_BIN=/path/to/opencode-1.18.18 OPENCODE_VERSION=1.18.18 \
  bash scripts/validation/run_opencode_integration.sh results/1.18.18
OPENCODE_BIN=/path/to/opencode-1.19.3 OPENCODE_VERSION=1.19.3 \
  bash scripts/validation/run_opencode_integration.sh results/1.19.3
```

Install `scripts/validation/opencode-phase1-requirements.txt` in an isolated
Python environment first. Model replies in these tests are synthetic; this is
not real-model, E2B, training, or 2.x acceptance. The sidecar's build version and
archive checksum must match; the adapter does not install or upgrade the CLI.

The Task must own the Sandbox lifecycle. Use `async with LocalSandbox()` or
cancel/join outstanding command callers and then `await sandbox.stop()` before
closing the event loop. This drains deferred spawn cleanups and subprocess pipe
callbacks for that instance. Merely draining the default executor cannot ensure
those callbacks finish. See [process cleanup](PROCESS_CLEANUP.md) for limits.
