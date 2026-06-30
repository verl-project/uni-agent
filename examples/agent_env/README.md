# Agent Environment Example

A minimal example of the **sandbox + tools** stack: it builds a `Toolbox` over a
persistent sandbox and runs a few steps (install a dep, create/edit a script with
the editor tool, run it, read the output) to show that state persists across calls.
The agent script runs on the host and never enters the sandbox image.

## Run

```bash
# Modal (default) -- set up auth first: pip install modal && modal token set ...
python examples/agent_env/demo.py

# or run locally on the host (no credentials needed):
SANDBOX_PROVIDER=local python examples/agent_env/demo.py

# override the Modal image:
IMAGE=python:3.12 python examples/agent_env/demo.py
```
