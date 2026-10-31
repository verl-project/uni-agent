"""Optional probes against a real OpenCode CLI and a synthetic SSE endpoint.

Set ``OPENCODE_BIN`` explicitly to OpenCode 1.19.3 to opt in; this test does
not auto-detect a CLI from ``PATH``. No real model, credentials, or GPU is used;
files and subprocesses created by the probes live under ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from uni_agent.agents.base import ModelConfig
from uni_agent.agents.opencode import OpenCodeAgent, OpenCodeConfig
from uni_agent.sandbox.local import LocalSandbox


def _pid_is_running(pid: int) -> bool:
    try:
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True
    status = result.stdout.strip()
    return bool(status) and not status.startswith("Z")


async def _terminate_pid(pid: int | None) -> dict[str, bool]:
    if pid is None:
        return {"was_running": False, "exited": True, "sigkill_sent": False}
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return {"was_running": False, "exited": True, "sigkill_sent": False}
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return {"was_running": False, "exited": True, "sigkill_sent": False}

    deadline = asyncio.get_running_loop().time() + 1.5
    while _pid_is_running(pid) and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)
    sigkill_sent = False
    if _pid_is_running(pid):
        try:
            os.kill(pid, signal.SIGKILL)
            sigkill_sent = True
        except ProcessLookupError:
            pass
    deadline = asyncio.get_running_loop().time() + 2
    while _pid_is_running(pid) and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)
    return {"was_running": True, "exited": not _pid_is_running(pid), "sigkill_sent": sigkill_sent}


def _configured_cli() -> str:
    setting = os.environ.get("OPENCODE_BIN")
    if not setting:
        pytest.skip("set OPENCODE_BIN explicitly to enable the optional OpenCode CLI probe")
    resolved = shutil.which(setting)
    if resolved is None and Path(setting).is_file() and os.access(setting, os.X_OK):
        resolved = setting
    if resolved is None:
        pytest.fail(f"OPENCODE_BIN={setting!r} is not an executable command or path")
    return resolved


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_real_cli_synthetic_sse_and_process_boundary_probes(tmp_path, monkeypatch):
    cli = _configured_cli()

    # OpenCode otherwise prefers inherited PWD over the subprocess cwd. Use a
    # different, test-owned directory and a relative tool write to catch this.
    inherited_directory = tmp_path / "inherited-directory"
    inherited_directory.mkdir()
    monkeypatch.setenv("PWD", str(inherited_directory))

    state = {"scenario": "text", "requests": [], "tool_command": "", "sleep_seconds": 0.0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(
                {
                    "path": self.path,
                    "body": request,
                    "auth_ok": self.headers.get("Authorization") == "Bearer probe-dummy-only",
                }
            )
            scenario = state["scenario"]
            if scenario == "request-timeout":
                time.sleep(state["sleep_seconds"])
            if scenario == "error":
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"synthetic rejection"}}')
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            has_tool_result = any(message.get("role") == "tool" for message in request["messages"])
            if scenario in {"tool-edit", "tool-timeout", "cancel-after-tool"} and not has_tool_result:
                tool_delta = {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_probe",
                            "type": "function",
                            "function": {"name": "bash", "arguments": json.dumps({"command": state["tool_command"]})},
                        }
                    ]
                }
                finish_reason = "tool_calls"
            else:
                tool_delta = {"content": "PROBE_DONE"}
                finish_reason = "stop"
            common = {
                "id": "chatcmpl-probe",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "synthetic-policy",
            }
            chunks = [
                {**common, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
                {**common, "choices": [{"index": 0, "delta": tool_delta, "finish_reason": None}]},
                {
                    **common,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                },
            ]
            try:
                for chunk in chunks:
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    spawned: list[tuple[tuple[str, ...], asyncio.subprocess.Process]] = []
    original_create = asyncio.create_subprocess_exec

    async def capture_subprocess(*argv, **kwargs):
        process = await original_create(*argv, **kwargs)
        spawned.append((tuple(str(value) for value in argv), process))
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture_subprocess)
    results = []
    controlled_child_pids = []
    cancel_task = None
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/sessions/probe/v1"

        async def run_scenario(name, workspace, *, timeout=20, cli_path=cli):
            state["scenario"] = name
            state["requests"] = []
            cfg = OpenCodeConfig(
                opencode_bin=cli_path,
                run_timeout=timeout,
                state_root=str(tmp_path / "state" / name),
                model=ModelConfig(base_url=endpoint, model_name="synthetic-policy", api_key="probe-dummy-only"),
            )
            return await OpenCodeAgent(cfg).run(
                sandbox=LocalSandbox(),
                messages=[{"role": "user", "content": "Run the harmless local probe and reply."}],
                workdir=str(workspace),
            )

        text_workspace = tmp_path / "text-workspace"
        text_workspace.mkdir()
        state["scenario"] = "text"
        text_result = await run_scenario("text", text_workspace)
        results.append({"scenario": "text", "finished": text_result.finished, "requests": len(state["requests"])})
        assert text_result.finished is True
        assert len(state["requests"]) == 1

        edit_workspace = tmp_path / "edit workspace with 'quote"
        edit_workspace.mkdir()
        edited_file = edit_workspace / "safe-marker.txt"
        state["tool_command"] = "printf 'safe-probe\\n' > safe-marker.txt"
        edit_result = await run_scenario("tool-edit", edit_workspace)
        results.append({"scenario": "tool-edit", "finished": edit_result.finished, "requests": len(state["requests"])})
        assert edit_result.finished is True
        assert len(state["requests"]) == 2
        assert edited_file.read_text(encoding="utf-8") == "safe-probe\n"
        assert not (inherited_directory / "safe-marker.txt").exists()

        state["scenario"] = "error"
        error_result = await run_scenario("error", tmp_path)
        results.append(
            {"scenario": "error", "finished": error_result.finished, "kind": error_result.info["error_kind"]}
        )
        assert error_result.finished is False and error_result.info["error_kind"] == "agent_failure"
        assert len(state["requests"]) == 1

        missing_result = await run_scenario("missing-cli", tmp_path, cli_path=str(tmp_path / "absent-opencode"))
        results.append(
            {"scenario": "missing-cli", "finished": missing_result.finished, "kind": missing_result.info["error_kind"]}
        )
        assert missing_result.finished is False and missing_result.info["error_kind"] == "cli_unavailable"
        assert not state["requests"]

        wrong_version_cli = tmp_path / "wrong-version-opencode"
        wrong_version_cli.write_text(
            '#!/bin/sh\nif [ "$1" = "--version" ]; then echo 1.0.0; exit 0; fi\nexit 2\n',
            encoding="utf-8",
        )
        wrong_version_cli.chmod(0o700)
        wrong_version = await run_scenario("wrong-version", tmp_path, cli_path=str(wrong_version_cli))
        results.append(
            {"scenario": "wrong-version", "finished": wrong_version.finished, "kind": wrong_version.info["error_kind"]}
        )
        assert wrong_version.finished is False and wrong_version.info["error_kind"] == "unsupported_version"
        assert not state["requests"]

        state["sleep_seconds"] = 8
        timeout_result = await run_scenario("request-timeout", tmp_path, timeout=5)
        results.append(
            {
                "scenario": "request-timeout",
                "finished": timeout_result.finished,
                "kind": timeout_result.info["error_kind"],
            }
        )
        assert timeout_result.finished is False and timeout_result.info["error_kind"] == "timeout"
        assert len(state["requests"]) == 1

        tool_workspace = tmp_path / "tool-timeout-workspace"
        tool_workspace.mkdir()
        pid_file = tool_workspace / ".tool-child.pid"
        state["tool_command"] = f"printf '%s\\n' $$ > {shlex.quote(str(pid_file))}; exec sleep 30"
        tool_timeout = await run_scenario("tool-timeout", tool_workspace, timeout=5)
        child_pid = int(pid_file.read_text(encoding="utf-8").strip()) if pid_file.exists() else None
        if child_pid is not None:
            controlled_child_pids.append(child_pid)
        child_cleanup = await _terminate_pid(child_pid)
        results.append(
            {
                "scenario": "tool-timeout",
                "finished": tool_timeout.finished,
                "kind": tool_timeout.info["error_kind"],
                "tool_child_was_running_after_cli_timeout": child_cleanup["was_running"],
                "tool_child_exited_after_exact_cleanup": child_cleanup["exited"],
                "tool_child_sigkill_sent": child_cleanup["sigkill_sent"],
            }
        )
        assert tool_timeout.finished is False and tool_timeout.info["error_kind"] == "timeout"
        assert len(state["requests"]) == 1
        assert child_cleanup["exited"], f"controlled tool child {child_pid} did not exit after exact-PID cleanup"

        cancel_workspace = tmp_path / "cancel-workspace"
        cancel_workspace.mkdir()
        cancel_pid_file = cancel_workspace / ".tool-child.pid"
        state["requests"] = []
        state["tool_command"] = f"printf '%s\\n' $$ > {shlex.quote(str(cancel_pid_file))}; exec sleep 30"
        state["scenario"] = "cancel-after-tool"
        cancel_cfg = OpenCodeConfig(
            opencode_bin=cli,
            run_timeout=30,
            state_root=str(tmp_path / "state" / "cancel-after-tool"),
            model=ModelConfig(base_url=endpoint, model_name="synthetic-policy", api_key="probe-dummy-only"),
        )
        cancel_task = asyncio.create_task(
            OpenCodeAgent(cancel_cfg).run(
                sandbox=LocalSandbox(),
                messages=[{"role": "user", "content": "Run the harmless local probe and reply."}],
                workdir=str(cancel_workspace),
            )
        )
        deadline = asyncio.get_running_loop().time() + 15
        while not cancel_pid_file.exists() and asyncio.get_running_loop().time() < deadline:
            if cancel_task.done():
                break
            await asyncio.sleep(0.05)
        assert cancel_pid_file.exists(), "the synthetic bash tool did not start before the cancellation deadline"
        cancel_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancel_task
        cancel_child_pid = int(cancel_pid_file.read_text(encoding="utf-8").strip())
        controlled_child_pids.append(cancel_child_pid)
        child_cleanup = await _terminate_pid(cancel_child_pid)
        run_processes = [process for argv, process in spawned if len(argv) > 1 and argv[1] == "run"]
        live_cli_pids = [process.pid for process in run_processes if process.returncode is None]
        for pid in live_cli_pids:
            cleanup = await _terminate_pid(pid)
            assert cleanup["exited"], f"controlled OpenCode CLI process {pid} did not exit after exact-PID cleanup"
        for process in run_processes:
            if process.returncode is None:
                await asyncio.wait_for(process.wait(), timeout=3)
            assert process.returncode is not None
        results.append(
            {
                "scenario": "cancel-after-tool",
                "task_cancelled": cancel_task.cancelled(),
                "tool_child_was_running_after_cancel": child_cleanup["was_running"],
                "tool_child_exited_after_exact_cleanup": child_cleanup["exited"],
                "tool_child_sigkill_sent": child_cleanup["sigkill_sent"],
                "cli_pids_requiring_exact_cleanup": live_cli_pids,
            }
        )
        assert cancel_task.cancelled()
        assert child_cleanup["exited"], f"controlled tool child {cancel_child_pid} did not exit after exact-PID cleanup"
        assert len(state["requests"]) == 1
        assert list((tmp_path / "state" / "cancel-after-tool").iterdir()) == []
        print("opencode_cli_probe=" + json.dumps({"results": results, "controlled_child_pids": controlled_child_pids}))
    finally:
        if cancel_task is not None:
            if not cancel_task.done():
                cancel_task.cancel()
            try:
                await asyncio.wait_for(asyncio.gather(cancel_task, return_exceptions=True), timeout=5)
            except TimeoutError:
                pytest.fail("cancelled OpenCode Agent task did not finish cleanup within 5 seconds")
        for pid in controlled_child_pids:
            cleanup = await _terminate_pid(pid)
            assert cleanup["exited"], f"controlled child PID {pid} remained after cleanup"
        for _, process in spawned:
            if process.returncode is None:
                cleanup = await _terminate_pid(process.pid)
                assert cleanup["exited"], f"controlled process PID {process.pid} remained after cleanup"
                await asyncio.wait_for(process.wait(), timeout=3)
        server.shutdown()
        server.server_close()
