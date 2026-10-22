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

## Run with Agent DX

Install the verified Agent DX Environment SDK and configure the service on each worker:

```bash
pip install /path/to/agent-dx/platform/sdk/environment/python
export ADX_SERVER_ADDRESS="<adx-api-host>:<port>"
export ADX_TOKEN="<token>"
export ADX_TLS="1"
export ADX_TLS_VERIFY="1"
DEBUG_MODE=1 SANDBOX_PROVIDER=agent_dx IMAGE=python:3.12 python examples/quickstart/sandbox/demo.py
```

The SDK is optional; other providers work without it. Use a verified source checkout
or wheel rather than assuming the SDK version has been published to a public index.

For a smaller check without the demo's package installation, run:

```bash
python examples/quickstart/sandbox/agent_dx_smoke.py
# Optional: the sandbox should reach an HTTP /health on the worker.
python examples/quickstart/sandbox/agent_dx_smoke.py --upstream 127.0.0.1:8000
```

This creates an Environment, checks commands, binary files, directories and persistent
shell state, and requests deletion. The optional upstream checks the reverse tunnel.
Physical resource cleanup requires deployment-side verification. The task image needs
Python 3, bash, tar, and gzip. The smoke command prints no authentication headers.

See the [Launch a Sandbox](../../../docs/source/quickstart/launch-sandbox.md) guide for a step-by-step explanation.
