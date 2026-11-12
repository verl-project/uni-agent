"""Opt-in cross-PR regression for #292's CLI and #234's LocalSandbox.

Use an explicitly provisioned OpenCode 1.18.x/1.19.x (OPENCODE_BIN). All model replies
come from loopback SSE with a dummy key. Liveness is judged BEFORE any rescue
signals; teardown can clean a failed test, but cannot make that test pass.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from uni_agent.agents.base import ModelConfig
from uni_agent.agents.opencode import OpenCodeAgent, OpenCodeConfig
from uni_agent.sandbox.local import LocalSandbox

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.level0,
    pytest.mark.asyncio,
    pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups"),
]


def _identity(pid: int) -> dict | None:
    """A passive snapshot; zombies have exited and cannot execute tools."""
    result = subprocess.run(
        ["ps", "-o", "ppid=,pgid=,stat=,lstart=", "-p", str(pid)],
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )
    fields = result.stdout.strip().split(maxsplit=3)
    if result.returncode != 0 or len(fields) != 4:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        raise AssertionError(f"cannot observe existing process {pid}: ps returned {result.returncode}")
    try:
        sid = os.getsid(pid)
    except ProcessLookupError:
        return None
    return {
        "pid": pid,
        "ppid": int(fields[0]),
        "pgid": int(fields[1]),
        "sid": sid,
        "status": fields[2],
        "started": fields[3],
    }


def _still_running(identity: dict) -> bool:
    current = _identity(identity["pid"])
    return bool(current and current["started"] == identity["started"] and not current["status"].startswith("Z"))


async def _wait_until(predicate, seconds: float) -> bool:
    deadline = asyncio.get_running_loop().time() + seconds
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            return False
        await asyncio.sleep(0.05)
    return True


def _configured_cli() -> str:
    setting = os.environ.get("OPENCODE_BIN")
    if not setting:
        pytest.skip("set OPENCODE_BIN to an explicitly provisioned OpenCode 1.18.x/1.19.x")
    cli = shutil.which(setting)
    if cli is None:
        pytest.fail(f"OPENCODE_BIN={setting!r} is not executable")
    return cli


@pytest.mark.parametrize("phase", ["request", "tool"])
@pytest.mark.parametrize("termination", ["timeout", "cancel"])
async def test_real_cli_processes_exit_before_test_cleanup(
    tmp_path,
    monkeypatch,
    phase,
    termination,
    *,
    agent_cls=OpenCodeAgent,
    config_cls=OpenCodeConfig,
    cli=None,
    standalone=False,
):
    cli = cli or _configured_cli()
    workspace = tmp_path / "workspace with 'quote"
    workspace.mkdir()
    monkeypatch.setenv("PWD", str(tmp_path))
    # Keep the tool shell, its Python worker and a grandchild alive. The shell
    # itself is spawned by the real CLI, with whatever grouping that CLI uses.
    marker = workspace / "processes.json"
    helper = workspace / "controlled_tool.py"
    helper.write_text(
        "import json, os, pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-I', '-c', 'import time; time.sleep(120)'])\n"
        "def identity(pid):\n"
        "    return dict(pid=pid, pgid=os.getpgid(pid), sid=os.getsid(pid))\n"
        "payload = dict(shell=identity(os.getppid()), worker=identity(os.getpid()), child=identity(child.pid))\n"
        "path = pathlib.Path(sys.argv[1])\n"
        "temporary = path.with_suffix('.tmp')\n"
        "temporary.write_text(json.dumps(payload))\n"
        "temporary.replace(path)\n"
        "child.wait()\n",
        encoding="utf-8",
    )
    command = f"{shlex.quote(sys.executable)} -I {shlex.quote(str(helper))} {shlex.quote(str(marker))}; :"
    requests = []
    received = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(
                {"path": self.path, "auth_ok": self.headers.get("Authorization") == "Bearer cleanup-dummy-only"}
            )
            received.set()
            if phase == "request":
                release.wait(90)
                return
            has_result = any(message.get("role") == "tool" for message in body["messages"])
            delta = (
                {"content": "CONTROLLED_TOOL_FINISHED"}
                if has_result
                else {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_cleanup_probe",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps(
                                    {"command": command, "description": "Run the local controlled process fixture"}
                                ),
                            },
                        }
                    ]
                }
            )
            common = {"id": "chatcmpl-cleanup", "object": "chat.completion.chunk", "created": 0, "model": "probe"}
            chunks = [
                {**common, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
                {**common, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {
                    **common,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop" if has_result else "tool_calls"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                },
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for chunk in chunks:
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    spawned = []
    original_spawn = asyncio.create_subprocess_exec

    async def observe_spawn(*argv, **kwargs):
        # Observe only; keep production spawn arguments and subprocess objects.
        proc = await original_spawn(*argv, **kwargs)
        if len(argv) > 1 and argv[1] == "run":
            spawned.append((proc, _identity(proc.pid)))
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", observe_spawn)
    state_root = tmp_path / "state"
    config = config_cls(
        opencode_bin=cli,
        run_timeout=12 if termination == "timeout" else 45,
        state_root=str(state_root),
        model=ModelConfig(
            base_url=f"http://127.0.0.1:{server.server_port}/sessions/cleanup/v1",
            model_name="probe",
            api_key="cleanup-dummy-only",
        ),
    )
    sandbox = LocalSandbox()
    task = asyncio.create_task(
        agent_cls(config).run(
            sandbox=sandbox,
            messages=[{"role": "user", "content": "Run the controlled local tool fixture."}],
            workdir=str(workspace),
        )
    )
    identities = {}
    evidence = {"phase": phase, "termination": termination, "before": identities, "rescue_signals": []}
    try:
        ready = marker.exists if phase == "tool" else received.is_set
        assert await _wait_until(lambda: ready() or task.done(), 10), "CLI did not reach the requested phase"
        assert ready(), f"CLI stopped before reaching {phase}: {task.result() if task.done() else 'still running'}"
        assert len(spawned) == 1
        proc, identity = spawned[0]
        assert identity is not None
        identities["cli"] = identity
        if standalone:
            children = subprocess.run(
                ["ps", "-axww", "-o", "pid=,ppid=,args="], capture_output=True, text=True, check=True
            ).stdout
            servers = [
                int(row.split(None, 2)[0])
                for row in children.splitlines()
                if len(row.split(None, 2)) == 3
                and row.split(None, 2)[1] == str(proc.pid)
                and "serve" in row.split()[2:]
                and "--stdio" in row.split()[2:]
            ]
            assert len(servers) == 1, [
                row for row in children.splitlines() if len(row.split()) >= 2 and row.split()[1] == str(proc.pid)
            ]
            identities["standalone_server"] = _identity(servers[0])
            # A sidecar supervisor also owns a separate --server client.
            for row in children.splitlines():
                fields = row.split(None, 2)
                if len(fields) == 3 and fields[1] == str(proc.pid) and int(fields[0]) not in servers:
                    identities[f"cli_child_{fields[0]}"] = _identity(int(fields[0]))
        if phase == "tool":
            for role, values in json.loads(marker.read_text()).items():
                identity = _identity(values["pid"])
                assert identity is not None and not identity["status"].startswith("Z")
                identities[role] = identity
            assert identities["worker"]["ppid"] == identities["shell"]["pid"]
            assert identities["child"]["ppid"] == identities["worker"]["pid"]
            assert identities["shell"]["pgid"] != identities["cli"]["pgid"]
            assert identities["shell"]["sid"] != identities["cli"]["sid"]
            # A provisioned executable may be a launcher for the native CLI.
            # Prove ownership and also track every intermediate launcher child.
            parent = identities["shell"]["ppid"]
            seen = set()
            for hop in range(16):
                if parent == identities["cli"]["pid"]:
                    break
                assert parent > 1 and parent not in seen, "tool shell is not descended from the captured CLI"
                seen.add(parent)
                ancestor = _identity(parent)
                assert ancestor is not None and not ancestor["status"].startswith("Z")
                identities[f"launcher_child_{hop}"] = ancestor
                parent = ancestor["ppid"]
            else:
                pytest.fail("tool shell ancestry did not reach the captured CLI within 16 hops")
        assert all(_still_running(identity) for identity in identities.values())
        if termination in {"cancel", "repeat-cancel"}:
            task.cancel()
            if termination == "repeat-cancel":
                for _ in range(3):
                    await asyncio.sleep(0.001)
                    task.cancel()
            _, pending = await asyncio.wait({task}, timeout=5)
            assert not pending, "Agent cancellation did not complete within 5 seconds"
            with pytest.raises(asyncio.CancelledError):
                task.result()
            evidence["cancelled"] = task.cancelled()
        else:
            _, pending = await asyncio.wait({task}, timeout=20)
            assert not pending, "Sandbox timeout did not finish the Agent within 20 seconds"
            result = task.result()
            evidence.update(finished=result.finished, error_kind=result.info.get("error_kind"))
            assert result.finished is False and result.info["error_kind"] == "timeout"
        await sandbox.stop()
        # SIGKILL is asynchronous. Only passive observations are permitted here.
        await _wait_until(lambda: not any(_still_running(value) for value in identities.values()), 3)
        live = {role: _identity(identity["pid"]) for role, identity in identities.items() if _still_running(identity)}
        evidence["live_before_rescue"] = live
        evidence["private_state_removed"] = state_root.exists() and not list(state_root.iterdir())
        assert requests and all(request["auth_ok"] for request in requests)
        assert all(request["path"] == "/sessions/cleanup/v1/chat/completions" for request in requests)
        assert not live, f"processes survived Sandbox {termination} before rescue: {json.dumps(live)}"
        assert evidence["private_state_removed"]
    finally:
        # Failure-only hygiene. The assertion above has already failed for a
        # leak, and the evidence retains the pre-rescue snapshot permanently.
        evidence.setdefault("live_before_rescue", None)  # None means setup/verdict was not reached.
        evidence["requests"] = list(requests)
        print("opencode_process_observation=" + json.dumps(evidence, sort_keys=True), flush=True)
        cleanup_errors = []
        try:
            if not task.done():
                task.cancel()
            # Unlike wait_for(), this cannot wait indefinitely for cancellation
            # to complete before we get a chance to rescue the owned processes.
            await asyncio.wait({task}, timeout=5)
            try:
                unrecorded = json.loads(marker.read_text()) if marker.exists() else {}
            except (OSError, ValueError) as exc:
                unrecorded = {}
                cleanup_errors.append(f"read fixture identities: {exc!r}")
            for role, values in unrecorded.items():
                if role not in identities:
                    try:
                        identity = _identity(values["pid"])
                        if identity and (identity["pgid"], identity["sid"]) == (values["pgid"], values["sid"]):
                            identities[role] = identity
                        elif identity:
                            cleanup_errors.append(f"refuse rescue of changed fixture identity {values['pid']}")
                    except Exception as exc:
                        cleanup_errors.append(f"observe {role} for rescue: {exc!r}")
            for _, identity in spawned:
                if identity:
                    identities.setdefault("cli", identity)
            for identity in reversed(list(identities.values())):
                try:
                    current = _identity(identity["pid"])
                    if not current or current["status"].startswith("Z") or current["started"] != identity["started"]:
                        continue
                    # Only rescue the recorded group/session identity. A
                    # same-start process moving group still failed liveness
                    # above, but is too ambiguous to signal during teardown.
                    if (current["pgid"], current["sid"]) != (identity["pgid"], identity["sid"]):
                        cleanup_errors.append(f"refuse rescue of changed identity {identity['pid']}")
                        continue
                    os.kill(current["pid"], signal.SIGKILL)
                    evidence["rescue_signals"].append(current["pid"])
                except ProcessLookupError:
                    pass
                except Exception as exc:
                    cleanup_errors.append(f"rescue PID {identity['pid']}: {exc!r}")
            waiters = {asyncio.create_task(proc.wait()) for proc, _ in spawned}
            if waiters:
                done, pending = await asyncio.wait(waiters, timeout=5)
                for waiter in done:
                    waiter.result()
                for waiter in pending:
                    waiter.cancel()
                if pending:
                    cleanup_errors.append("CLI pipes did not close within 5 seconds after rescue")
                    await asyncio.wait(pending, timeout=1)
            _, pending = await asyncio.wait({task}, timeout=5)
            if pending:
                cleanup_errors.append("Agent task did not finish within 5 seconds after rescue")
            elif not task.cancelled():
                task.exception()  # Retrieve any setup failure without masking its original assertion.
        except Exception as exc:
            cleanup_errors.append(f"process teardown: {exc!r}")
        finally:
            await sandbox.stop()
            release.set()
            for close in (server.shutdown, server.server_close):
                try:
                    close()
                except Exception as exc:
                    cleanup_errors.append(f"server teardown: {exc!r}")
            thread.join(timeout=2)
            if thread.is_alive():
                cleanup_errors.append("SSE server thread did not stop")
            evidence["cleanup_errors"] = cleanup_errors
            evidence["requests"] = list(requests)
            print("opencode_process_cleanup=" + json.dumps(evidence, sort_keys=True), flush=True)
    # Runs only after a successful main verdict: no passing case may rely on
    # signals sent by the test or hide a failure in emergency teardown.
    assert not evidence["rescue_signals"], evidence
    assert not evidence["cleanup_errors"], evidence
