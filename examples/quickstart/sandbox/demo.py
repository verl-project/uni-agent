"""Minimal demo of the sandbox + tools stack."""

import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from uni_agent.logging import sample_logging
from uni_agent.sandbox import Sandbox, SandboxConfig, build_sandbox
from uni_agent.tools import Toolbox

logger = logging.getLogger("demo")


def banner(title: str) -> None:
    bar = "=" * 64
    logger.info(f"\n{bar}\n  {title}\n{bar}")


def _indent(text, prefix: str = "    | ") -> str:
    # `text` may be a ToolResult; str() yields its text channel.
    return "Output:\n" + "\n".join(prefix + line for line in str(text).splitlines())


def build_sandbox_config() -> SandboxConfig:
    provider = os.getenv("SANDBOX_PROVIDER", "modal")
    image = os.getenv("IMAGE")

    if provider == "local":
        if image is not None:
            logger.warning("Ignoring IMAGE=%r because SANDBOX_PROVIDER=local does not use container images", image)
        image = None
    elif image is None:
        image = "python:3.12"

    return SandboxConfig(
        provider=provider,
        image=image,
        runtime_timeout=3600,
    )


def build_tool_specs() -> list[dict]:
    return [
        {
            "name": "stateful_shell",
            "command_timeout": 120,
            "env_vars": {
                "PAGER": "cat",
                "GIT_PAGER": "cat",
                "PIP_PROGRESS_BAR": "off",
                "TQDM_DISABLE": "1",
            },
        },
        {"name": "str_replace_editor"},
    ]


async def toolbox_operations(sandbox: Sandbox, tool_specs: list[dict]) -> None:
    """Exercise Toolbox shell + str_replace_editor against the sandbox."""
    banner("toolbox_operations: start — install dep -> create script -> run -> cat output")
    logger.info(f"tools selected : {[t['name'] for t in tool_specs]}")
    logger.info("(shell keeps a persistent shell channel; the editor is stateless)")
    logger.info("each tool owns its own state")

    toolbox = Toolbox.from_specs(tool_specs, sandbox=sandbox)

    async with toolbox.entered(retry=3, timeout=60):
        schemas = toolbox.schemas()
        logger.info(f"-> tool schemas : {[s['function']['name'] for s in schemas]}")

        # clean slate: local /tmp persists across runs (a fresh remote sandbox is already clean)
        await toolbox.call("shell", {"command": "rm -f /tmp/demo.py /tmp/demo_out.txt"})

        logger.info("[Step 0] shell: show shell env from config")
        result = await toolbox.call("shell", {"command": "echo PAGER=$PAGER TQDM_DISABLE=$TQDM_DISABLE"})
        logger.info(_indent(result))

        logger.info("[Step 1] shell: pip install numpy (persists in this sandbox)")
        result = await toolbox.call("shell", {"command": "pip install -q numpy && echo installed"})
        logger.info(_indent(result))

        script = "import numpy as np\nprint('sum =', int(np.array([1, 2, 4]).sum()))\n"
        logger.info("[Step 2] str_replace_editor create /tmp/demo.py (writes via data plane)")
        result = await toolbox.call(
            "str_replace_editor", {"command": "create", "path": "/tmp/demo.py", "file_text": script}
        )
        logger.info(_indent(result))

        logger.info("[Step 3] str_replace_editor view /tmp/demo.py")
        result = await toolbox.call("str_replace_editor", {"command": "view", "path": "/tmp/demo.py"})
        logger.info(_indent(result))

        logger.info("[Step 4] shell: run script -> /tmp/demo_out.txt")
        result = await toolbox.call("shell", {"command": "python3 /tmp/demo.py > /tmp/demo_out.txt 2>&1"})
        logger.info(_indent(result))

        logger.info("[Step 5] shell: cat /tmp/demo_out.txt (proves the file persisted)")
        result = await toolbox.call("shell", {"command": "cat /tmp/demo_out.txt"})
        logger.info(_indent(result))

        logger.info("[Step 6] str_replace_editor str_replace (sum -> product), then re-run")
        await toolbox.call(
            "str_replace_editor",
            {
                "command": "str_replace",
                "path": "/tmp/demo.py",
                "old_str": "print('sum =', int(np.array([1, 2, 4]).sum()))",
                "new_str": "print('product =', int(np.array([1, 2, 4]).prod()))",
            },
        )
        result = await toolbox.call("shell", {"command": "python3 /tmp/demo.py"})
        logger.info(_indent(result))

        logger.info("[Step 7] stateful shell: cd /tmp, then a later call still sees it")
        await toolbox.call("shell", {"command": "cd /tmp"})
        result = await toolbox.call("shell", {"command": "echo cwd=$(pwd); python3 demo.py"})
        logger.info(_indent(result))

    banner("toolbox_operations: done")


async def file_operations(sandbox: Sandbox) -> None:
    """Exercise sandbox write/read/upload/download."""
    banner("file_operations: start — write_file / read_file / upload / download")

    await sandbox.exec_shell("rm -f /tmp/sandbox_write.txt /tmp/sandbox_upload.txt")

    logger.info("[Step 1] sandbox.write_file -> /tmp/sandbox_write.txt")
    await sandbox.write_file("/tmp/sandbox_write.txt", "hello from write_file\n")
    logger.info("[Step 2] sandbox.read_file /tmp/sandbox_write.txt")
    written = await sandbox.read_file("/tmp/sandbox_write.txt")
    logger.info(_indent(written.decode("utf-8")))
    assert written.decode("utf-8") == "hello from write_file\n"

    with tempfile.TemporaryDirectory() as tmp:
        local_src = Path(tmp) / "upload_src.txt"
        local_dst = Path(tmp) / "download_dst.txt"
        local_src.write_text("hello from upload\n", encoding="utf-8")

        logger.info("[Step 3] sandbox.upload host file -> /tmp/sandbox_upload.txt")
        await sandbox.upload(local_src, "/tmp/sandbox_upload.txt")
        remote = await sandbox.read_file("/tmp/sandbox_upload.txt")
        logger.info(_indent(remote.decode("utf-8")))
        assert remote.decode("utf-8") == "hello from upload\n"

        logger.info("[Step 4] sandbox.download /tmp/sandbox_upload.txt -> host")
        await sandbox.download("/tmp/sandbox_upload.txt", local_dst)
        logger.info(_indent(local_dst.read_text(encoding="utf-8")))
        assert local_dst.read_text(encoding="utf-8") == "hello from upload\n"

    banner("file_operations: done")


async def main() -> None:
    sandbox_config = build_sandbox_config()
    tool_specs = build_tool_specs()

    sandbox = build_sandbox(sandbox_config)
    # No log_path -> console only (run with DEBUG_MODE=1 to see the INFO walkthrough).
    async with sample_logging("demo"), sandbox:
        banner(f"sandbox (provider={sandbox_config.provider})")
        await toolbox_operations(sandbox, tool_specs)
        await file_operations(sandbox)

        banner("Demo done")


if __name__ == "__main__":
    asyncio.run(main())
