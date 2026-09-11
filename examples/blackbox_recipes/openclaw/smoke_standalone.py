"""Standalone real-OpenClaw tool smoke with a deterministic fake LLM.

This verifies runtime/tool wiring, NOT Qwen task-solving ability. Run only in
an isolated container. The server binds loopback; no real credentials are used.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def create_mock_server(tool_command=None):
    requests: list[dict] = []
    command = "python3 -c \"from pathlib import Path; Path('/workspace/answer.txt').write_text(str(6*7))\""

    if tool_command is not None:
        command = tool_command

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            index = len(requests)
            if self.path != "/v1/chat/completions" or index > 2:
                self.send_error(400, "unexpected request")
                return
            if index == 1:
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_smoke_exec", "type": "function",
                    "function": {"name": "exec", "arguments": json.dumps({"command": command})},
                }]}
                reason = "tool_calls"
            else:
                message = {"role": "assistant", "content": "DONE"}
                reason = "stop"
            base = {"id": f"chatcmpl-smoke-{index}", "created": int(time.time()),
                    "model": request["model"]}
            self.send_response(200)
            if request.get("stream"):
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                delta = dict(message)
                if "tool_calls" in delta:
                    delta["tool_calls"] = [dict(delta["tool_calls"][0], index=0)]
                chunks = [dict(base, object="chat.completion.chunk", choices=[{
                    "index": 0, "delta": delta, "finish_reason": None}]),
                    dict(base, object="chat.completion.chunk", choices=[{
                    "index": 0, "delta": {}, "finish_reason": reason}],
                    usage={"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130})]
                for chunk in chunks:
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(dict(base, object="chat.completion", choices=[{
                    "index": 0, "message": message, "finish_reason": reason}],
                    usage={"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130})).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, requests


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="openclaw-smoke-"))
    server, requests = create_mock_server()
    model = "Qwen3.5-9B"
    config = {
        "models": {"mode": "replace", "providers": {"vllm": {
            "baseUrl": f"http://127.0.0.1:{server.server_port}/v1", "apiKey": "smoke-placeholder",
            "api": "openai-completions", "agentRuntime": {"id": "openclaw"},
            "models": [{"id": model, "name": model, "reasoning": False, "input": ["text"],
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                        "contextWindow": 32768, "maxTokens": 1024}],
        }}},
        "agents": {"defaults": {
            "model": {"primary": f"vllm/{model}", "fallbacks": []},
            "models": {f"vllm/{model}": {"agentRuntime": {"id": "openclaw"}}},
            "workspace": "/workspace", "cwd": "/workspace", "skipBootstrap": True, "skills": [],
            "embeddedAgent": {"projectSettingsPolicy": "ignore", "executionContract": "default"},
            "contextPruning": {"mode": "off"},
            "compaction": {"enabled": False, "midTurnPrecheck": {"enabled": False},
                           "memoryFlush": {"enabled": False}},
            "sandbox": {"mode": "off"}, "timeoutSeconds": 90, "maxConcurrent": 1,
        }},
        "tools": {"allow": ["exec"], "codeMode": False, "toolSearch": False, "swarm": False,
                  "agentToAgent": {"enabled": False}, "fs": {"workspaceOnly": True},
                  "exec": {"host": "gateway", "security": "full", "ask": "off",
                           "applyPatch": {"enabled": False}, "timeoutSeconds": 10}},
        "plugins": {"allow": ["vllm"]},
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "task.txt").write_text(
        "Use exec to calculate 6*7 and write the result to /workspace/answer.txt. Then reply DONE.",
        encoding="utf-8",
    )
    env = dict(os.environ, HOME=str(root / "home"), OPENCLAW_CONFIG_PATH=str(root / "config.json"),
               OPENCLAW_STATE_DIR=str(root / "state"), OPENCLAW_NO_RESPAWN="1")
    session_id = str(uuid.uuid4())
    command_line = ["/opt/openclaw/bin/openclaw", "agent", "--local", "--agent", "main",
                    "--session-id", session_id, "--message-file", str(root / "task.txt"),
                    "--thinking", "off", "--timeout", "90", "--json"]
    stdout, stderr, exit_code = "", "", -1
    error = None
    try:
        result = subprocess.run(command_line, env=env, capture_output=True, text=True, timeout=120)
        stdout, stderr, exit_code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        error = "outer_timeout"
        stdout = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    finally:
        server.shutdown()
    answer_path = Path("/workspace/answer.txt")
    answer = answer_path.read_text() if answer_path.exists() else None
    prefix_preserved = len(requests) == 2 and (
        requests[1]["messages"][:len(requests[0]["messages"])] == requests[0]["messages"]
    )
    state_files = list((root / "state").rglob("*"))
    database_evidence = []
    for db in state_files:
        if db.is_file() and db.suffix == ".sqlite":
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
                tables = connection.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall()
                database_evidence.append({"path": str(db.relative_to(root)), "tables": tables})
    transcripts = [path for path in state_files if path.is_file() and path.suffix == ".jsonl"]
    events = []
    for path in transcripts:
        events.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    compactions = [e for e in events if e.get("type") in {"compaction", "branch_summary"}]
    # --local may retain the explicit session in its in-process store. The terminal
    # receipt and the two-request prefix check are therefore the smoke evidence.
    report = {
        "test_kind": "deterministic_fake_llm_real_openclaw_not_qwen_validation",
        "session_id": session_id, "exit_code": exit_code, "error": error,
        "request_count": len(requests), "answer": answer, "prefix_preserved": prefix_preserved,
        "transcript_count": len(transcripts), "state_file_count": len(state_files),
        "compaction_count": len(compactions),
        "stdout": stdout, "stderr": stderr,
        "trajectory_verified": False,
        "passed": exit_code == 0 and answer == "42" and prefix_preserved
                  and not compactions,
    }
    artifacts = Path("/artifacts")
    artifacts.mkdir(exist_ok=True)
    shutil.copytree(root / "state", artifacts / "state", dirs_exist_ok=False)
    (artifacts / "sqlite-schema.json").write_text(json.dumps(database_evidence, indent=2))
    (artifacts / "smoke-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (artifacts / "requests.json").write_text(json.dumps(requests, indent=2), encoding="utf-8")
    (artifacts / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
