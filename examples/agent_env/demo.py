# ruff: noqa: E501
"""Minimal demo of the sandbox + tools stack (no Environment wrapper).

The agent (this script) runs on the host and never enters the task image. It picks
a sandbox provider, then builds a :class:`~uni_agent.tools.Toolbox` directly from a
``{name, ...kwargs}`` tool spec. There is no session layer -- each tool owns its
own state: ``shell`` keeps a persistent shell channel, opened lazily on first use,
while the editor is stateless:

    build_sandbox(SandboxConfig(...))       # pick a provider
        async with sandbox: ...             # start() on enter, stop() on exit
    Toolbox.from_specs(tools, sandbox=...)  # build the tool surface
        -> call("shell", {...})             # one action -> Observation
        -> close()                          # close tools (release channels)

In the real stack the *task* owns the sandbox lifecycle and the *agent* owns the
toolbox; this script just wires them by hand to show the lower layers in isolation.

Flow: install a dep -> create a script with the editor tool -> run it, writing
output to a file -> cat the file (showing the sandbox persists state across calls).

Run:
    pip install modal && modal token set ...          # one-time Modal auth
    python examples/agent_env/demo.py
    # local (host) instead of Modal, no creds needed:
    SANDBOX_PROVIDER=local python examples/agent_env/demo.py
    # override the Modal image:
    IMAGE=python:3.12 python examples/agent_env/demo.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uni_agent.sandbox import SandboxConfig, build_sandbox
from uni_agent.tools import Toolbox


def banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def _indent(text, prefix: str = "    | ") -> str:
    # `text` may be an Observation; str() yields its text channel.
    return "\n".join(prefix + line for line in str(text).splitlines()) + "\n"


def build_sandbox_config() -> SandboxConfig:
    return SandboxConfig(
        provider=os.getenv("SANDBOX_PROVIDER", "modal"),
        image=os.getenv("IMAGE", "python:3.12"),
        runtime_timeout=3600,
    )


def build_tool_specs() -> list[dict]:
    return [
        {
            "name": "stateful_shell",  # registry key; model sees it as `shell`
            "command_timeout": 120,  # per-command timeout (seconds)
            "env_vars": {
                "PAGER": "cat",
                "GIT_PAGER": "cat",
                "PIP_PROGRESS_BAR": "off",
                "TQDM_DISABLE": "1",
            },
        },
        {"name": "str_replace_editor"},
    ]


async def main() -> None:
    sandbox_config = build_sandbox_config()
    tool_specs = build_tool_specs()

    banner(f"sandbox (provider={sandbox_config.provider}); each tool owns its own state")
    print(f"  tools selected   : {[t['name'] for t in tool_specs]}")
    print("  (shell keeps a persistent shell channel; the editor is stateless)")

    sandbox = build_sandbox(sandbox_config)
    async with sandbox:  # start() on enter, stop() on exit
        toolbox = Toolbox.from_specs(tool_specs, sandbox=sandbox)
        schemas = toolbox.schemas()
        print(f"  -> tool schemas  : {[s['function']['name'] for s in schemas]}")

        banner("Sandbox demo: install dep -> create script -> run -> cat output")

        # clean slate: local /tmp persists across runs (a fresh remote sandbox is already clean)
        await toolbox.call("shell", {"command": "rm -f /tmp/demo.py /tmp/demo_out.txt"})

        # 0. shell-channel env from config is live
        print("\n[Step 0] shell: show shell env from config")
        print(_indent(await toolbox.call("shell", {"command": "echo PAGER=$PAGER TQDM_DISABLE=$TQDM_DISABLE"})))

        # 1. install a dependency (persists in this sandbox)
        print("[Step 1] shell: pip install numpy")
        print(_indent(await toolbox.call("shell", {"command": "pip install -q numpy && echo installed"})))

        # 2. create a runnable script with the editor tool (writes via data plane)
        script = "import numpy as np\nprint('sum =', int(np.array([1, 2, 4]).sum()))\n"
        print("[Step 2] str_replace_editor create /tmp/demo.py")
        print(
            _indent(
                await toolbox.call(
                    "str_replace_editor", {"command": "create", "path": "/tmp/demo.py", "file_text": script}
                )
            )
        )

        # 3. view it back
        print("[Step 3] str_replace_editor view /tmp/demo.py")
        print(_indent(await toolbox.call("str_replace_editor", {"command": "view", "path": "/tmp/demo.py"})))

        # 4. run the script, sending output to a file
        print("[Step 4] shell: run script -> /tmp/demo_out.txt")
        print(_indent(await toolbox.call("shell", {"command": "python3 /tmp/demo.py > /tmp/demo_out.txt 2>&1"})))

        # 5. cat the output file (proves the file persisted in the sandbox)
        print("[Step 5] shell: cat /tmp/demo_out.txt")
        print(_indent(await toolbox.call("shell", {"command": "cat /tmp/demo_out.txt"})))

        # 6. edit the script (sum -> product), then re-run
        print("[Step 6] str_replace_editor str_replace (sum -> product), then re-run")
        await toolbox.call(
            "str_replace_editor",
            {
                "command": "str_replace",
                "path": "/tmp/demo.py",
                "old_str": "print('sum =', int(np.array([1, 2, 4]).sum()))",
                "new_str": "print('product =', int(np.array([1, 2, 4]).prod()))",
            },
        )
        print(_indent(await toolbox.call("shell", {"command": "python3 /tmp/demo.py"})))

        # 7. stateful shell: cd persists across calls (same channel)
        print("[Step 7] stateful shell: cd /tmp, then a later call still sees it")
        await toolbox.call("shell", {"command": "cd /tmp"})
        print(_indent(await toolbox.call("shell", {"command": "echo cwd=$(pwd); python3 demo.py"})))

        banner("Demo done (toolbox.close() releases the shell channel; async-with stops the sandbox)")
        await toolbox.close()  # close tools (channels) before the sandbox stops


if __name__ == "__main__":
    asyncio.run(main())
