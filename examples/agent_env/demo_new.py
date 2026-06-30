# ruff: noqa: E501
"""Minimal Environment demo on the sandbox + tools stack.

The agent (this script) runs on the host and never enters the task image. It
builds an :class:`~uni_agent.environment.Environment` from config; the environment
picks a sandbox provider and exposes a tool surface. There is no session layer --
each tool owns its own state: ``shell`` keeps a persistent shell channel,
opened lazily on first use, while the editor is stateless:

    Environment.from_config(...)            # provider + tools (+ shell env)
        -> reset()                          # boot sandbox, build the toolbox
        -> call("shell", {...})             # one action -> Observation
        -> close()                          # close tools (channels), stop sandbox

Flow (mirrors examples/agent_env/demo.py): install a dep -> create a script with
the editor tool -> run it, writing output to a file -> cat the file (showing the
sandbox persists state across calls).

Run:
    pip install modal && modal token set ...          # one-time Modal auth
    python examples/agent_env/demo_new.py
    # local (host) instead of Modal, no creds needed:
    SANDBOX_PROVIDER=local python examples/agent_env/demo_new.py
    # override the Modal image:
    IMAGE=python:3.12 python examples/agent_env/demo_new.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uni_agent.environment import Environment


def banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def _indent(text, prefix: str = "    | ") -> str:
    # `text` may be an Observation; str() yields its text channel.
    return "\n".join(prefix + line for line in str(text).splitlines()) + "\n"


def build_env_config() -> dict:
    provider = os.getenv("SANDBOX_PROVIDER", "modal")
    return {
        "sandbox": {
            "provider": provider,
            "image": os.getenv("IMAGE", "python:3.12"),
            "runtime_timeout": 3600,
        },
        "tools": [
            {
                "name": "shell",
                "command_timeout": 120,  # per-command timeout (seconds)
                "env_vars": {
                    "PAGER": "cat",
                    "GIT_PAGER": "cat",
                    "PIP_PROGRESS_BAR": "off",
                    "TQDM_DISABLE": "1",
                },
            },
            {"name": "str_replace_editor"},
        ],
    }


async def main() -> None:
    config = build_env_config()
    env = Environment.from_config(config)
    provider = config["sandbox"]["provider"]

    tool_names = [t["name"] for t in config["tools"]]
    banner(f"Environment (provider={provider}); each tool owns its own state")
    print(f"  tools selected   : {tool_names}")
    print("  (shell keeps a persistent shell channel; the editor is stateless)")

    schemas = await env.reset()
    print(f"  -> tool schemas  : {[s['function']['name'] for s in schemas]}")

    try:
        banner("Sandbox demo: install dep -> create script -> run -> cat output")

        # clean slate: local /tmp persists across runs (a fresh remote sandbox is already clean)
        await env.call("shell", {"command": "rm -f /tmp/demo.py /tmp/demo_out.txt"})

        # 0. shell-channel env from config is live
        print("\n[Step 0] shell: show shell env from config")
        print(_indent(await env.call("shell", {"command": "echo PAGER=$PAGER TQDM_DISABLE=$TQDM_DISABLE"})))

        # 1. install a dependency (persists in this sandbox)
        print("[Step 1] shell: pip install numpy")
        print(_indent(await env.call("shell", {"command": "pip install -q numpy && echo installed"})))

        # 2. create a runnable script with the editor tool (writes via data plane)
        script = "import numpy as np\nprint('sum =', int(np.array([1, 2, 4]).sum()))\n"
        print("[Step 2] str_replace_editor create /tmp/demo.py")
        print(
            _indent(
                await env.call("str_replace_editor", {"command": "create", "path": "/tmp/demo.py", "file_text": script})
            )
        )

        # 3. view it back
        print("[Step 3] str_replace_editor view /tmp/demo.py")
        print(_indent(await env.call("str_replace_editor", {"command": "view", "path": "/tmp/demo.py"})))

        # 4. run the script, sending output to a file
        print("[Step 4] shell: run script -> /tmp/demo_out.txt")
        print(_indent(await env.call("shell", {"command": "python3 /tmp/demo.py > /tmp/demo_out.txt 2>&1"})))

        # 5. cat the output file (proves the file persisted in the sandbox)
        print("[Step 5] shell: cat /tmp/demo_out.txt")
        print(_indent(await env.call("shell", {"command": "cat /tmp/demo_out.txt"})))

        # 6. edit the script (sum -> product), then re-run
        print("[Step 6] str_replace_editor str_replace (sum -> product), then re-run")
        await env.call(
            "str_replace_editor",
            {
                "command": "str_replace",
                "path": "/tmp/demo.py",
                "old_str": "print('sum =', int(np.array([1, 2, 4]).sum()))",
                "new_str": "print('product =', int(np.array([1, 2, 4]).prod()))",
            },
        )
        print(_indent(await env.call("shell", {"command": "python3 /tmp/demo.py"})))

        # 7. stateful shell: cd persists across calls (same channel)
        print("[Step 7] stateful shell: cd /tmp, then a later call still sees it")
        await env.call("shell", {"command": "cd /tmp"})
        print(_indent(await env.call("shell", {"command": "echo cwd=$(pwd); python3 demo.py"})))

        banner("Demo done (close() releases the shell channel then stops the sandbox)")
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
