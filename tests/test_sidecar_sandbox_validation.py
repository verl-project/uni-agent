"""Failure-path checks for the standalone remote validation entrypoint."""

import io
import json
import logging
import sys
from types import SimpleNamespace

import pytest

import test_sandbox as validation


@pytest.fixture
def fake_remote(monkeypatch, tmp_path):
    state = SimpleNamespace(
        fail_command=False,
        fail_cleanup=False,
        still_running=False,
        killed=False,
        created=False,
        connection=None,
        create_kwargs=None,
        files={},
        check_output='{"result":"passed"}',
    )
    source = tmp_path / "sandbox_api.py"
    source.write_text('tunnel_ws_url = "wss://example.invalid"\n')
    monkeypatch.setenv("AKERNEL_SERVER_ADDRESS", "example.invalid:443")
    monkeypatch.setenv("AKERNEL_TOKEN", "private-service-value")
    monkeypatch.setenv("TUNNEL_SSL_VERIFY", "1")
    monkeypatch.setattr(validation.importlib.metadata, "version", lambda _: "0.10.2rc9")
    monkeypatch.setattr(validation.time, "sleep", lambda _: None)
    monkeypatch.setattr(validation.urllib.request, "urlopen", lambda *args, **kwargs: io.StringIO('{"is_alive": true}'))
    previous_logging_level = logging.root.manager.disable

    class FakeSandbox:
        sandbox_id = "validation-instance"

        def __init__(self, **kwargs):
            state.created = True
            state.create_kwargs = kwargs
            self.commands = self
            self.files = SimpleNamespace(write=lambda path, content: state.files.update({path: content}))

        def run(self, command, **kwargs):
            if state.fail_command:
                raise RuntimeError("failure contains private-service-value")
            if "SANDBOX_OK" in command:
                output = "SANDBOX_OK\n/usr/bin/bash\nPython 3.11.15"
            elif "is_alive" in command:
                output = '{"is_alive": true}'
            elif "metadata.version" in command:
                output = "1.4.0"
            elif "/probe.py" in command:
                output = state.check_output
            else:
                output = ""
            return SimpleNamespace(exit_code=0, stdout=output, stderr="")

        def get_port_url(self, *args, **kwargs):
            return "http://example.invalid:443/validation-instance/8000"

        def kill(self):
            state.killed = True
            if state.fail_cleanup:
                raise RuntimeError("cleanup failed with private-service-value")

        def is_running(self):
            return state.still_running

    def connection_config(**kwargs):
        state.connection = SimpleNamespace(**kwargs)
        return state.connection

    monkeypatch.setitem(
        sys.modules,
        "yr_sandbox",
        SimpleNamespace(
            Sandbox=FakeSandbox,
            Mount=SimpleNamespace,
            ConnectionConfig=connection_config,
            __file__=str(source),
        ),
    )
    yield state, source, tmp_path / "report.json"
    logging.disable(previous_logging_level)


def run_validation(report):
    return validation.main(["--image", "task-image:v1", "--report", str(report)])


def test_intermediate_failure_still_cleans_up_and_redacts(fake_remote, capsys):
    state, _, report = fake_remote
    state.fail_command = True
    assert run_validation(report) == 1
    result = json.loads(report.read_text())
    assert state.killed
    assert result["stages"]["cleanup"]["confirmed"]
    assert not result["stages"]["command"]["success"]
    captured = capsys.readouterr()
    assert "private-service-value" not in report.read_text() + captured.out + captured.err
    assert "[REDACTED]" in report.read_text()


@pytest.mark.parametrize("failure", ["fail_cleanup", "still_running"])
def test_cleanup_failure_cannot_report_success(fake_remote, failure):
    state, _, report = fake_remote
    setattr(state, failure, True)
    assert run_validation(report) == 1
    result = json.loads(report.read_text())
    assert result["stages"]["external_health"]["success"]
    assert not result["success"]
    assert not result["stages"]["cleanup"]["success"]


def test_requires_pinned_openyuanrong_sandbox(fake_remote, monkeypatch):
    state, _, report = fake_remote
    monkeypatch.setattr(validation.importlib.metadata, "version", lambda _: "0.10.3")
    assert run_validation(report) == 1
    assert not state.created
    result = json.loads(report.read_text())
    assert not result["stages"]["sdk"]["success"]
    assert "0.10.2rc9" in result["stages"]["sdk"]["error"]


def test_frontend_uses_jwt_and_tls_on_port_443(fake_remote, monkeypatch):
    state, _, report = fake_remote
    monkeypatch.setenv("OPENYUANRONG_TOKEN", "legacy-acl-identifier")
    assert (
        validation.main(
            [
                "--image",
                "task-image:v1",
                "--report",
                str(report),
                "--cpu",
                "1000",
                "--memory",
                "2048",
            ]
        )
        == 0
    )
    assert state.connection.token == "private-service-value"
    assert state.connection.server_address == "example.invalid:443"
    assert state.connection.use_tls is True
    assert state.connection.gateway_use_tls is True
    assert state.connection.verify_tls is True
    assert state.create_kwargs["connection"] is state.connection
    assert (state.create_kwargs["cpu"], state.create_kwargs["memory"]) == (1000, 2048)
    result = json.loads(report.read_text())
    assert result["resources"] == {"cpu_millicores": 1000, "memory_mb": 2048}
    assert result["stages"]["cleanup"]["confirmed"]


@pytest.mark.parametrize("option", [["--sidecar-recipe", "my-tool"], ["--sidecar-image", "tool:v1"]])
def test_sidecar_options_must_be_paired(option):
    with pytest.raises(SystemExit, match="2"):
        validation.parse_args(["--image", "task:v1", *option])


@pytest.mark.parametrize("check_success", [True, False])
def test_custom_sidecar_mount_and_check_failure_cleanup(fake_remote, tmp_path, check_success):
    state, _, report = fake_remote
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    manifest = tmp_path / "recipes.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recipes": {
                    "custom-tool": {
                        "context": ".",
                        "dockerfile": "Dockerfile",
                        "image": "custom-tool",
                        "tag": "{TOOL_VERSION}",
                        "platform": "linux/amd64",
                        "mount_target": "/opt/custom-tool",
                        "entrypoint": ["/opt/custom-tool/bin/run"],
                        "probe_argv": ["/opt/custom-tool/bin/run", "--version"],
                        "version_arg": "TOOL_VERSION",
                        "build_args": {"BASE_IMAGE": "debian:bookworm-slim", "TOOL_VERSION": "1"},
                    }
                },
            }
        )
    )
    request, expected = tmp_path / "input", tmp_path / "expected"
    request.write_text("request")
    expected.write_text("response")
    state.check_output = '{"result":"passed"}' if check_success else '{"result":"failed"}'
    code = validation.main(
        [
            "--image",
            "task:v1",
            "--sidecar-recipe",
            "custom-tool",
            "--sidecar-image",
            "registry.example/tool:1",
            "--recipe-file",
            str(manifest),
            "--stdin-file",
            str(request),
            "--expected-stdout-file",
            str(expected),
            "--report",
            str(report),
        ]
    )
    assert code == (0 if check_success else 1)
    mount = state.create_kwargs["mounts"][0]
    assert (mount.target, mount.image_url) == ("/opt/custom-tool", "registry.example/tool:1")
    assert any(path.endswith("/case.json") for path in state.files)
    assert state.killed
    result = json.loads(report.read_text())
    assert result["stages"]["cleanup"]["confirmed"]
    assert result["stages"]["sidecar_entrypoint"]["success"] is check_success
