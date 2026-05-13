import asyncio

from uni_agent.deployment.local import deployment as local_deployment
from uni_agent.deployment.local.deployment import (
    LocalDeployment,
    _is_apptainer_runtime,
    _normalize_apptainer_image,
)


def _clear_runtime_env(monkeypatch):
    for env_var in local_deployment._CONTAINER_RUNTIME_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


def test_default_container_runtime_uses_env_override_before_path(monkeypatch):
    monkeypatch.setenv("UNI_AGENT_CONTAINER_RUNTIME", "/custom/bin/runtime")
    monkeypatch.setenv("LOCAL_CONTAINER_RUNTIME", "/local/bin/runtime")
    monkeypatch.setattr(local_deployment.shutil, "which", lambda runtime: f"/usr/bin/{runtime}")

    assert local_deployment._default_container_runtime() == "/custom/bin/runtime"


def test_default_container_runtime_uses_local_env_override(monkeypatch):
    monkeypatch.delenv("UNI_AGENT_CONTAINER_RUNTIME", raising=False)
    monkeypatch.setenv("LOCAL_CONTAINER_RUNTIME", "/local/bin/apptainer")
    monkeypatch.setattr(local_deployment.shutil, "which", lambda runtime: f"/usr/bin/{runtime}")

    assert local_deployment._default_container_runtime() == "/local/bin/apptainer"


def test_default_container_runtime_discovers_supported_runtime_from_path(monkeypatch):
    _clear_runtime_env(monkeypatch)
    paths = {"singularity": "/usr/bin/singularity", "docker": "/usr/bin/docker"}
    calls = []

    def fake_which(runtime):
        calls.append(runtime)
        return paths.get(runtime)

    monkeypatch.setattr(local_deployment.shutil, "which", fake_which)

    assert local_deployment._default_container_runtime() == "/usr/bin/singularity"
    assert calls == ["apptainer", "singularity"]


def test_default_container_runtime_falls_back_to_apptainer_name(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setattr(local_deployment.shutil, "which", lambda runtime: None)

    assert local_deployment._default_container_runtime() == "apptainer"


def test_apptainer_runtime_detection_accepts_paths_and_singularity_alias():
    assert _is_apptainer_runtime("/opt/apptainer/bin/apptainer")
    assert _is_apptainer_runtime("singularity")
    assert not _is_apptainer_runtime("docker")


def test_normalize_apptainer_image_adds_docker_scheme_for_oci_names():
    assert _normalize_apptainer_image("python:3.12") == "docker://python:3.12"
    assert _normalize_apptainer_image("registry.example.com/ns/image:tag") == (
        "docker://registry.example.com/ns/image:tag"
    )
    assert _normalize_apptainer_image("docker://python:3.12") == "docker://python:3.12"
    assert _normalize_apptainer_image("local-image.sif") == "local-image.sif"


def test_apptainer_command_places_runtime_args_before_image_and_uses_shell():
    deployment = LocalDeployment(
        run_id="test",
        type="local",
        container_runtime="/opt/apptainer/bin/apptainer",
        image="python:3.12",
        shell="/bin/sh",
        extra_run_args=["--bind", "/host:/mnt"],
    )

    command = deployment._build_apptainer_command("python3 -m swerex.server --port 3456")

    assert command == [
        "/opt/apptainer/bin/apptainer",
        "exec",
        "--cleanenv",
        "--compat",
        "--bind",
        "/host:/mnt",
        "docker://python:3.12",
        "/bin/sh",
        "-lc",
        "python3 -m swerex.server --port 3456",
    ]


def test_format_command_supports_token_and_port_placeholders():
    deployment = LocalDeployment(
        run_id="test",
        type="local",
        container_runtime="apptainer",
        command="server --port {port} --auth-token {token}",
    )

    assert deployment._format_command(token="secret", port=4567) == "server --port 4567 --auth-token secret"


def test_docker_command_keeps_published_to_runtime_port_mapping():
    deployment = LocalDeployment(
        run_id="test",
        type="local",
        container_runtime="docker",
        image="python:3.12",
        runtime_port=8000,
        shell="/bin/bash",
        extra_run_args=["--cpus", "1"],
    )
    deployment._get_current_container_network = lambda: None

    command = deployment._build_run_command("sandbox-name", 4567, "server")

    assert command == [
        "docker",
        "run",
        "--rm",
        "-d",
        "--name",
        "sandbox-name",
        "--entrypoint",
        "/bin/bash",
        "-p",
        "4567:8000",
        "--cpus",
        "1",
        "python:3.12",
        "-lc",
        "server",
    ]


class _FakeRuntime:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class _FakeProcess:
    def __init__(self):
        self.terminated = False
        self.killed = False
        self.wait_timeout = None

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.wait_timeout = timeout
        return 0

    def kill(self):
        self.killed = True


def test_stop_closes_runtime_and_apptainer_process_without_docker_rm(tmp_path):
    deployment = LocalDeployment(
        run_id="test",
        type="local",
        container_runtime="apptainer",
    )
    runtime = _FakeRuntime()
    process = _FakeProcess()
    log_path = tmp_path / "apptainer.log"
    log_path.write_text("server log", encoding="utf-8")
    log_handle = log_path.open("a", encoding="utf-8")

    deployment._runtime = runtime
    deployment._server_process = process
    deployment._server_log_path = log_path
    deployment._server_log_handle = log_handle
    deployment._container_name = "sandbox-name"
    deployment._container_id = "container-id"
    deployment._stopped = False

    asyncio.run(deployment.stop())

    assert runtime.closed
    assert process.terminated
    assert not process.killed
    assert process.wait_timeout == 10
    assert deployment._server_process is None
    assert deployment._server_log_handle is None
    assert deployment._server_log_path is None
    assert deployment._container_name is None
    assert deployment._container_id is None
    assert not log_path.exists()
