"""Launch one canonical SWE-bench task image without additional image mounts.

Edit ``CONFIG["sandbox"]`` below, then run from the repository root:

    python smoke_test_sandbox.py
"""

from __future__ import annotations

import asyncio

from uni_agent.sandbox import SandboxConfig, build_sandbox

CONFIG = {
    "sandbox": {
        "provider": "modal",
        "runtime_timeout": 600,
        "image": "swebench/sweb.eval.x86_64.astropy_1776_astropy-13033",
        "image_map": [],
        "image_mounts": [
            {
                "image": "docker.io/yuy2001/uni-agent-images:20260926",
                "mount_path": "/opt/uni-agent/runtime",
            }
        ],
        "executable_paths": {
            "claude": "/opt/uni-agent/runtime/bin/claude",
        },
        "sandbox_kwargs": {},
    }
}


def build_config() -> SandboxConfig:
    return SandboxConfig.model_validate(CONFIG["sandbox"])


async def main() -> None:
    config = build_config()
    print(f"starting provider={config.provider!r} image={config.image!r}")

    sandbox = build_sandbox(config)
    async with sandbox:
        result = await sandbox.exec_shell(
            """
set -eu
echo "sandbox is ready"
echo "user=$(id -u):$(id -g)"
test -x /opt/uni-agent/runtime/bin/claude
claude_bin="$(command -v claude)"
test "$claude_bin" = /usr/bin/claude
echo "claude=$claude_bin -> $(readlink /usr/bin/claude)"
claude --version
test -d /testbed
ls -la /testbed
""".strip(),
            timeout=60,
        )
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="")
        if result.exit_code != 0:
            raise RuntimeError(f"sandbox smoke command failed with exit code {result.exit_code}")

    print("sandbox stopped")


if __name__ == "__main__":
    asyncio.run(main())
