"""Exercise the Agent DX provider against a user-configured development cluster."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from uni_agent.sandbox import SandboxConfig, build_sandbox


async def smoke(args: argparse.Namespace) -> dict:
    kwargs = {"rootfs_readonly": False}
    if args.upstream:
        kwargs.update(upstream=args.upstream, proxy_port=args.proxy_port)
    sandbox = build_sandbox(
        SandboxConfig(provider="agent_dx", image=args.image, runtime_timeout=300, sandbox_kwargs=kwargs)
    )
    checks = []
    async with sandbox.entered(retry=1):
        environment_id = sandbox.environment_id
        value = await sandbox.exec(["python3", "-c", "print('agent-dx-ok')"])
        assert value.exit_code == 0 and value.stdout.strip() == "agent-dx-ok", value
        checks.append("exec")
        value = await sandbox.exec_shell("exit 7")
        assert value.exit_code == 7, value
        checks.append("nonzero-exit")
        await sandbox.write_file("/tmp/uni-agent-smoke.bin", b"\0agent-dx\xff")
        assert await sandbox.read_file("/tmp/uni-agent-smoke.bin") == b"\0agent-dx\xff"
        checks.append("binary-files")
        with tempfile.TemporaryDirectory(prefix="uni-agent-adx-smoke-") as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "data.bin").write_bytes(b"\0directory\xff")
            await sandbox.upload(source, "/tmp/uni-agent-smoke-dir")
            destination = Path(directory) / "download"
            await sandbox.download("/tmp/uni-agent-smoke-dir", destination)
            assert (destination / "data.bin").read_bytes() == b"\0directory\xff"
        checks.append("directory-roundtrip")
        shell = await sandbox.open_shell(cwd="/tmp", env={"UNI_AGENT_SMOKE": "quote ' preserved"})
        try:
            assert (await shell.run("cd / && export UNI_AGENT_STATE=saved")).exit_code == 0
            value = await shell.run('printf "%s|%s|%s" "$PWD" "$UNI_AGENT_SMOKE" "$UNI_AGENT_STATE"')
            assert value.exit_code == 0 and value.stdout == "/|quote ' preserved|saved", value
        finally:
            await shell.close()
        checks.append("persistent-shell")
        if args.upstream:
            url = sandbox.get_tunnel_url().rstrip("/") + args.tunnel_path
            script = (
                "import urllib.request; "
                f"response = urllib.request.urlopen({url!r}, timeout=10); "
                "assert response.status == 200"
            )
            value = await sandbox.exec(["python3", "-c", script], timeout=20)
            assert value.exit_code == 0, value
            checks.append("reverse-tunnel-http-200")
    return {
        "provider": "agent_dx",
        "environment_id": environment_id,
        "checks_passed": checks,
        "cleanup_request": "accepted_by_sdk",
        "physical_deletion": "not_verified",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="python:3.12")
    parser.add_argument("--upstream", help="Optional worker-reachable HTTP service, e.g. 127.0.0.1:8000")
    parser.add_argument("--proxy-port", type=int, default=38197)
    parser.add_argument("--tunnel-path", default="/health", help="HTTP path that returns 200 on --upstream")
    args = parser.parse_args()
    if not args.tunnel_path.startswith("/"):
        parser.error("--tunnel-path must start with /")
    print(json.dumps(asyncio.run(smoke(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
