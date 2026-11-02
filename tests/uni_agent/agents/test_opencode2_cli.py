"""Real 2.2.5, synthetic loopback model only. Explicit OPENCODE2_BIN opt-in."""

import asyncio
import json
import os
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import test_opencode_cli_probe as probe
import test_opencode_process_cleanup as cleanup
import test_opencode_sidecar_environment as sidecar

from examples.opencode.smoke_task import run_smoke, scripted_endpoint
from uni_agent.agents.base import ModelConfig
from uni_agent.agents.opencode2 import OpenCode2Agent, OpenCode2Config
from uni_agent.sandbox.local import LocalSandbox

pytestmark = [pytest.mark.cpu, pytest.mark.level0, pytest.mark.asyncio]


@pytest.fixture
def cli():
    value = os.environ.get("OPENCODE2_BIN")
    if not value:
        pytest.skip("set OPENCODE2_BIN to a provisioned 2.2.5 executable")
    assert shutil.which(value), f"not executable: {value}"
    return value


async def test_cli_contract(tmp_path, monkeypatch, cli):
    await probe.test_real_cli_synthetic_sse_and_process_boundary_probes(
        tmp_path, monkeypatch, cli=cli, agent_cls=OpenCode2Agent, config_cls=OpenCode2Config
    )


async def test_disallowed_tools_not_offered_to_model(tmp_path, cli):
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"synthetic rejection"}}')

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        cfg = OpenCode2Config(
            opencode_bin=cli,
            run_timeout=20,
            state_root=str(tmp_path / "state"),
            model=ModelConfig(
                base_url=f"http://127.0.0.1:{server.server_port}/v1", model_name="probe", api_key="dummy"
            ),
        )
        async with LocalSandbox() as sandbox:
            result = await OpenCode2Agent(cfg).run(
                sandbox=sandbox,
                messages=[{"role": "user", "content": "Inspect available tools"}],
                workdir=str(tmp_path),
            )
        assert not result.finished and result.info["error_kind"] == "agent_failure"
        assert len(bodies) == 1
        names = {tool["function"]["name"] for tool in bodies[0]["tools"]}
        assert names & {"bash", "shell"}, names
        assert not names & {"task", "subagent", "webfetch", "websearch", "question", "image_generate"}, names
        assert not any(name.startswith("schedule") for name in names), names
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("phase", ["request", "tool"])
@pytest.mark.parametrize("termination", ["timeout", "cancel", "repeat-cancel"])
async def test_private_server_and_tools_exit(tmp_path, monkeypatch, cli, phase, termination):
    await cleanup.test_real_cli_processes_exit_before_test_cleanup(
        tmp_path,
        monkeypatch,
        phase,
        termination,
        cli=cli,
        agent_cls=OpenCode2Agent,
        config_cls=OpenCode2Config,
        standalone=True,
    )


@pytest.mark.parametrize("scenario,reward,requests", [("patch", 1.0, 2), ("no-edit", 0.0, 1)])
async def test_independent_task_verifier(cli, scenario, reward, requests):
    result = await run_smoke(cli, scenario, config_cls=OpenCode2Config)
    assert result.finished and result.reward == result.accuracy == reward
    assert result.extra_info["requests"] == requests
    assert result.extra_info["cli_version"] == "opencode v2.2.5"
    assert result.extra_info["workspace_removed"]


async def test_concurrent_isolated_state_and_project_config(tmp_path, monkeypatch, cli):
    inherited = tmp_path / "inherited"
    inherited.mkdir()
    (inherited / "auth.json").write_text('{"sentinel":"untouched"}')
    database = inherited / "personal.db"
    database.write_text("untouched")
    monkeypatch.setenv("OPENCODE_TEST_HOME", str(inherited))
    monkeypatch.setenv("OPENCODE_DB", str(database))
    monkeypatch.setenv("PWD", str(inherited))
    monkeypatch.setenv("OPENCODE_CONFIG_PROJECT_DISABLE", "0")
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", '{"model":"invalid/provider"}')
    state = tmp_path / "private-state"
    workspaces = [tmp_path / f"workspace {index}" for index in range(2)]
    for workspace in workspaces:
        workspace.mkdir()
        (workspace / "calculator.py").write_text("def add(a, b): return a - b\n")
        (workspace / "opencode.json").write_text(
            json.dumps(
                {
                    "providers": {"uni-agent": {"settings": {"baseURL": "http://127.0.0.1:1/wrong"}}},
                    "permissions": [{"action": "*", "resource": "*", "effect": "deny"}],
                }
            )
        )

    async def launch(workspace, endpoint):
        cfg = OpenCode2Config(
            opencode_bin=cli,
            state_root=str(state),
            run_timeout=30,
            model=ModelConfig(base_url=endpoint.base_url, model_name="scripted-policy", api_key="dummy"),
        )
        async with LocalSandbox() as sandbox:
            return await OpenCode2Agent(cfg).run(
                sandbox=sandbox, messages=[{"role": "user", "content": "Fix calculator.py"}], workdir=str(workspace)
            )

    with scripted_endpoint("patch") as one, scripted_endpoint("patch") as two:
        results = await asyncio.gather(launch(workspaces[0], one), launch(workspaces[1], two))
        assert [one.requests, two.requests] == [2, 2]
    assert all(result.finished for result in results), [r.info for r in results]
    sessions = [{e["sessionID"] for e in result.output["events"]} for result in results]
    assert sessions[0].isdisjoint(sessions[1])
    assert all("return a + b" in (w / "calculator.py").read_text() for w in workspaces)
    assert database.read_text() == "untouched"
    assert list(state.iterdir()) == []
    assert sorted(p.name for p in inherited.iterdir()) == ["auth.json", "personal.db"]


async def test_unrelated_server_survives_cancellation(tmp_path, monkeypatch, cli):
    root = tmp_path / "control-server"
    root.mkdir()
    cfg = OpenCode2Config(model=ModelConfig(base_url="http://127.0.0.1:1/v1", model_name="unused", api_key="dummy"))
    env = {**os.environ, **OpenCode2Agent._environment(cfg, str(root), str(root), OpenCode2Agent._runtime_config(cfg))}
    env["OPENCODE_PASSWORD"] = "control-server-dummy"
    server = await asyncio.create_subprocess_exec(
        cli,
        "serve",
        "--stdio",
        "--port",
        "0",
        env=env,
        cwd=root,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        ready = json.loads(await asyncio.wait_for(server.stdout.readline(), 10))
        assert ready["url"].startswith("http://127.0.0.1:")
        identity = cleanup._identity(server.pid)
        assert identity
        task_dir = tmp_path / "cancelled-task"
        task_dir.mkdir()
        await cleanup.test_real_cli_processes_exit_before_test_cleanup(
            task_dir,
            monkeypatch,
            "tool",
            "repeat-cancel",
            cli=cli,
            agent_cls=OpenCode2Agent,
            config_cls=OpenCode2Config,
            standalone=True,
        )
        assert server.returncode is None and cleanup._still_running(identity)
        server.stdin.close()
        assert await asyncio.wait_for(server.wait(), 10) == 0
    finally:
        if server.returncode is None:
            server.kill()
            await server.wait()


@pytest.mark.parametrize("library_env", ["unset", "empty", "glibc", "custom"])
async def test_mounted_sidecar_libraries(tmp_path, monkeypatch, library_env):
    cli = os.environ.get("OPENCODE2_SIDECAR_BIN")
    if not cli:
        pytest.skip("set OPENCODE2_SIDECAR_BIN to the Linux mounted payload")
    await sidecar.test_sidecar_preserves_task_library_environment(
        tmp_path, monkeypatch, library_env, cli=cli, agent_cls=OpenCode2Agent, config_cls=OpenCode2Config
    )
