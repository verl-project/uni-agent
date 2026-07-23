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

See the [Launch a Sandbox](../../../docs/source/quickstart/launch-sandbox.md) guide for a step-by-step explanation.
