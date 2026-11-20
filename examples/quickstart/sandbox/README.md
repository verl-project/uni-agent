# Sandbox Demo

A minimal walkthrough of Uni-Agent's sandbox and tool stack. The demo creates a sandbox, binds a stateful shell and file editor to it, installs a dependency, writes and edits a Python script, and verifies that state persists across tool calls.

The Python driver always runs on the host. Commands and file operations are routed through the selected sandbox backend.

## Run Locally

The local provider requires no credentials, but runs commands directly on the host:

```bash
DEBUG_MODE=1 SANDBOX_PROVIDER=local python examples/quickstart/sandbox/demo.py
```

## Run with Docker

Docker provides a local isolated environment. Its daemon must be running; the image is pulled automatically when it is not already available locally:

```bash
DEBUG_MODE=1 SANDBOX_PROVIDER=docker IMAGE=python:3.12 python examples/quickstart/sandbox/demo.py
```

## Run with Modal

Install and authenticate Modal first:

```bash
pip install modal
modal token set
DEBUG_MODE=1 python examples/quickstart/sandbox/demo.py
```

The default image is `python:3.12`. Override it with:

```bash
DEBUG_MODE=1 IMAGE=python:3.11 python examples/quickstart/sandbox/demo.py
```

## Run with E2B

Install E2B and provide an API key. The provider builds the image-backed
template on first use when it is missing:

```bash
pip install "e2b>=2.26,<3"
export E2B_API_KEY="<e2b-api-key>"
DEBUG_MODE=1 SANDBOX_PROVIDER=e2b IMAGE=python:3.12 python examples/quickstart/sandbox/demo.py
```

## Run with AgentEnv

AgentEnv serves an E2B-compatible API, so it reuses the E2B client and the
`e2b` extra. All three connection settings are required; each one falls back to
its `E2B_*` counterpart when the `AGENTENV_*` variable is unset:

```bash
pip install "e2b>=2.26,<3"
export AGENTENV_API_URL="<agentenv-api-url>"
export AGENTENV_SANDBOX_URL="<agentenv-sandbox-url>"
export AGENTENV_API_KEY="<agentenv-api-key>"
```

Reuse an existing template on the deployment. `sandbox_kwargs.template` takes
priority and skips E2B's template-build API:

```yaml
sandbox:
  provider: agentenv
  runtime_timeout: 3600
  sandbox_kwargs:
    template: uni-agent-swe
```

Passing `image` instead requires the deployment to implement template builds.
`expose_port` needs a reachable `domain`, because the SDK derives the exposed
host as `{port}-{sandbox_id}.{domain}`.

## Split Traffic between Providers

The composite `routes` provider starts exactly one child sandbox from a weighted
list. Routing is deterministic: the same `salt` and `route_key` always select
the same route, and the task runner injects the Gateway session id as the
routing key. Copy this `sandbox` block into a task config:

```yaml
sandbox:
  provider: routes
  sandbox_kwargs:
    strategy: weighted        # "weighted" (default) or "primary"
    salt: rollout-2026-11
    fallback: true
    routes:
      - name: e2b-prod
        weight: 90
        config:
          provider: e2b
          runtime_timeout: 3600
          sandbox_kwargs: {template: uni-agent-swe}
      - name: agentenv-canary
        weight: 10
        config:
          provider: agentenv
          runtime_timeout: 3600
          sandbox_kwargs: {template: uni-agent-swe}
```

Each route's `config` is a full sandbox config, so put `image`, mounts, and
`runtime_timeout` there; the routes-level config only reads `sandbox_kwargs`.
Startup failover tries the remaining routes in declaration order on transient
errors, while configuration/SDK/permission errors are raised. The active route
is logged and never changes during an episode. Outside the framework, set
`sandbox_kwargs.route_key` to a stable value (such as `uuid.uuid4().hex` per
episode) so retries stay on the same route.

See the [Launch a Sandbox](../../../docs/source/quickstart/launch-sandbox.md) guide for a step-by-step explanation.
