"""Restricted Codex CLI adapter. Reads one ticket on stdin and emits one JSON answer."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

ALLOWED = {"inventory", "read", "search", "check"}
DISABLED = ("shell_tool", "unified_exec", "plugins", "remote_plugin", "computer_use", "multi_agent")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cli", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--reasoning", required=True, choices=("low", "medium", "high", "xhigh"))
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--helper")
    p.add_argument("--sha256")
    a = p.parse_args()
    if bool(a.helper) != bool(a.sha256) or (a.helper and Path(a.helper).name != a.helper):
        p.error("Specify a helper filename and its SHA-256 together")
    prompt = sys.stdin.buffer.read()
    work = Path.cwd()
    cli = str(a.cli.resolve(strict=True))
    reader = Path(__file__).with_name("reader.py").resolve()
    # Only file-based login state is copied; custom user/project tools and instructions are never copied.
    old = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    auth = old / "auth.json"
    with tempfile.TemporaryDirectory(prefix="skill-eval-private-") as tmp:
        root = Path(tmp)
        profile = root / "profile"
        home = profile / ".codex"
        home.mkdir(parents=True)
        empty = root / "empty"
        empty.mkdir()
        if auth.is_file():
            shutil.copyfile(auth, home / "auth.json")
        env = {k: v for k, v in os.environ.items() if not k.startswith("CODEX_")}
        env.update(CODEX_HOME=str(home), HOME=str(profile), USERPROFILE=str(profile))
        rargs = ["-I", "-B", str(reader), str(work)] + (
            ["--helper", a.helper, "--sha256", a.sha256] if a.helper else []
        )
        cmd = [
            cli,
            "-a",
            "never",
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "--json",
            "-C",
            str(empty),
            "-m",
            a.model,
            "-c",
            "model_reasoning_effort=" + json.dumps(a.reasoning),
            "-c",
            'web_search="disabled"',
            "-c",
            "project_doc_max_bytes=0",
            "-c",
            "mcp_servers.evidence.command=" + json.dumps(sys.executable),
            "-c",
            "mcp_servers.evidence.args=" + json.dumps(rargs, ensure_ascii=False),
            "-c",
            "mcp_servers.evidence.required=true",
            "-c",
            "mcp_servers.evidence.startup_timeout_sec=30",
        ]
        for name in sorted(ALLOWED):
            cmd += ["-c", f'mcp_servers.evidence.tools.{name}.approval_mode="approve"']
        for flag in DISABLED:
            cmd += ["--disable", flag]
        cmd += ["-o", str(root / "answer"), "-"]
        proc = subprocess.Popen(
            cmd,
            cwd=empty,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
        try:
            raw, err = proc.communicate(
                prompt + b"\nInspect scene and any package with evidence tools. Return pure JSON.",
                timeout=a.timeout,
            )
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/pid", str(proc.pid), "/t", "/f"], capture_output=True)
            else:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise SystemExit("Participant timed out") from None
        if proc.returncode:
            # Do not surface raw logs, which can contain local paths and service/account details.
            raise SystemExit(f"CLI failed (exit {proc.returncode}); no answer accepted")
        evidence_calls = []
        tokens = None
        for line in raw.splitlines():
            evt = json.loads(line)
            item = evt.get("item", {})
            kind = item.get("type")
            if evt.get("type") == "turn.completed":
                tokens = evt.get("usage")
            if not kind:
                continue
            if kind == "mcp_tool_call":
                if item.get("server") != "evidence" or item.get("tool") not in ALLOWED:
                    raise SystemExit("Unexpected MCP call; trial discarded")
                if (
                    evt.get("type") == "item.completed"
                    and item.get("status") == "completed"
                    and not (item.get("result") or {}).get("isError")
                ):
                    evidence_calls.append(item["tool"])
            elif kind not in ("agent_message", "reasoning", "plan", "todo_list"):
                raise SystemExit("Unexpected participant action; trial discarded")
        if not evidence_calls:
            raise SystemExit("No successful evidence tool call; trial discarded")
        answer = json.loads((root / "answer").read_text(encoding="utf-8-sig"))
        if not isinstance(answer, dict):
            raise SystemExit("Answer is not an object")
        print(
            json.dumps({"model": a.model, "reasoning": a.reasoning, "tools": evidence_calls, "usage": tokens}),
            file=sys.stderr,
        )
        print(json.dumps(answer, ensure_ascii=True))


if __name__ == "__main__":
    main()
