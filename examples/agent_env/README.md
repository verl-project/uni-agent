# Agent Environment Example

This directory contains a minimal example for launching an agent environment and running commands inside a persistent sandbox.

This README is intentionally brief. For the full setup guide, configuration details, and walkthrough, see `../../docs/source/start/agent_env.md`.

## Run

```bash
DEPLOYMENT=<local|vefaas> DEBUG_MODE=1 python examples/agent_env/demo.py
```

- `local`: run with a local sandbox backend.
- `vefaas`: run with a remote veFaaS deployment.

See the main documentation for environment variables, dependencies, and deployment instructions.
