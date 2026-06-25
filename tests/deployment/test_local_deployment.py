import asyncio
from pathlib import Path
from types import SimpleNamespace

from uni_agent.deployment.local import deployment as local_deployment
from uni_agent.deployment.local.deployment import (
    LocalDeployment,
    _is_apptainer_runtime,
    _is_running_in_container,
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


def test_running_in_container_detects_docker_or_podman_markers(monkeypatch):
    marker_paths = {
        "/.dockerenv": True,
        "/run/.containerenv": False,
    }

    original_exists = Path.exists

    def fake_exists(path):
        normalized_path = path.as_posix()
        if normalized_path.endswith("/.dockerenv"):
            return marker_paths["/.dockerenv"]
        if normalized_path.endswith("/run/.containerenv"):
            return marker_paths["/run/.containerenv"]
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fake_exists)

    assert _is_running_in_container()

    marker_paths["/.dockerenv"] = False
    marker_paths["/run/.containerenv"] = True

    assert _is_running_in_container()

    marker_paths["/run/.containerenv"] = False

    assert not _is_running_in_container()


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

    command = deployment._build_run_command("sandbox-name", 4567, "server", network=None)

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


class _CapturingRuntime:
    configs = []

    @classmethod
    def from_config(cls, config, run_id=None):
        cls.configs.append((config, run_id))
        return cls()


def _capture_exec(commands):
    def fake_exec(args, check=True):
        commands.append(args)
        return _completed_container()

    return fake_exec


def _completed_container(container_id="container-id"):
    return SimpleNamespace(stdout=f"{container_id}\n")


def test_oci_runtime_uses_published_port_when_connecting_from_host(monkeypatch):
    deployment = LocalDeployment(
        run_id="test",
        type="local",
        container_runtime="docker",
        image="python:3.12",
        runtime_port=8000,
    )
    deployment._runtime_exec = lambda args, check=True: _completed_container()
    deployment._get_current_container_network = lambda: None
    monkeypatch.setattr(local_deployment, "_is_running_in_container", lambda: False)
    monkeypatch.setattr(local_deployment, "LocalRuntime", _CapturingRuntime)
    _CapturingRuntime.configs = []

    asyncio.run(deployment._start_oci_container(token="secret", container_name="sandbox-name", published_port=4567))

    runtime_config, run_id = _CapturingRuntime.configs[-1]
    assert run_id == "test"
    assert runtime_config.host == "http://127.0.0.1"
    assert runtime_config.port == 4567


def test_oci_runtime_uses_published_port_with_explicit_host(monkeypatch):
    deployment = LocalDeployment(
        run_id="test",
        type="local",
        container_runtime="docker",
        image="python:3.12",
        host="http://docker-host.example",
        runtime_port=8000,
    )
    deployment._runtime_exec = lambda args, check=True: _completed_container()
    monkeypatch.setattr(local_deployment, "LocalRuntime", _CapturingRuntime)
    _CapturingRuntime.configs = []

    asyncio.run(deployment._start_oci_container(token="secret", container_name="sandbox-name", published_port=4567))

    runtime_config, run_id = _CapturingRuntime.configs[-1]
    assert run_id == "test"
    assert runtime_config.host == "http://docker-host.example"
    assert runtime_config.port == 4567


def test_oci_runtime_uses_container_port_with_explicit_host_from_container(monkeypatch):
    deployment = LocalDeployment(
        run_id="test",
        type="local",
        container_runtime="docker",
        image="python:3.12",
        host="http://sandbox",
        runtime_port=8000,
        network="agent-net",
    )
    deployment._runtime_exec = lambda args, check=True: _completed_container()
    monkeypatch.setattr(local_deployment, "_is_running_in_container", lambda: True)
    monkeypatch.setattr(local_deployment, "LocalRuntime", _CapturingRuntime)
    _CapturingRuntime.configs = []

    asyncio.run(deployment._start_oci_container(token="secret", container_name="sandbox-name", published_port=4567))

    runtime_config, run_id = _CapturingRuntime.configs[-1]
    assert run_id == "test"
    assert runtime_config.host == "http://sandbox"
    assert runtime_config.port == 8000


def test_oci_runtime_uses_published_port_with_host_gateway_from_container(monkeypatch):
    for host in ("http://host.docker.internal", "http://host.containers.internal"):
        deployment = LocalDeployment(
            run_id="test",
            type="local",
            container_runtime="docker",
            image="python:3.12",
            host=host,
            runtime_port=8000,
            network="agent-net",
        )
        deployment._runtime_exec = lambda args, check=True: _completed_container()
        monkeypatch.setattr(local_deployment, "_is_running_in_container", lambda: True)
        monkeypatch.setattr(local_deployment, "LocalRuntime", _CapturingRuntime)
        _CapturingRuntime.configs = []

        asyncio.run(deployment._start_oci_container(token="secret", container_name="sandbox-name", published_port=4567))

        runtime_config, run_id = _CapturingRuntime.configs[-1]
        assert run_id == "test"
        assert runtime_config.host == host
        assert runtime_config.port == 4567


def test_oci_runtime_uses_container_port_with_explicit_host_and_host_network(monkeypatch):
    deployment = LocalDeployment(
        run_id="test",
        type="local",
        container_runtime="docker",
        image="python:3.12",
        host="http://docker-host.example",
        runtime_port=8000,
        network="host",
    )
    deployment._runtime_exec = lambda args, check=True: _completed_container()
    monkeypatch.setattr(local_deployment, "LocalRuntime", _CapturingRuntime)
    _CapturingRuntime.configs = []

    asyncio.run(deployment._start_oci_container(token="secret", container_name="sandbox-name", published_port=4567))

    runtime_config, run_id = _CapturingRuntime.configs[-1]
    assert run_id == "test"
    assert runtime_config.host == "http://docker-host.example"
    assert runtime_config.port == 8000


def test_oci_runtime_uses_container_port_with_host_network(monkeypatch):
    deployment = LocalDeployment(
        run_id="test",
        type="local",
        container_runtime="docker",
        image="python:3.12",
        runtime_port=8000,
        network="host",
    )
    deployment._runtime_exec = lambda args, check=True: _completed_container()
    monkeypatch.setattr(local_deployment, "LocalRuntime", _CapturingRuntime)
    _CapturingRuntime.configs = []

    asyncio.run(deployment._start_oci_container(token="secret", container_name="sandbox-name", published_port=4567))

    runtime_config, run_id = _CapturingRuntime.configs[-1]
    assert run_id == "test"
    assert runtime_config.host == "http://127.0.0.1"
    assert runtime_config.port == 8000


def test_oci_command_omits_port_publish_with_host_network():
    deployment = LocalDeployment(
        run_id="test",
        type="local",
        container_runtime="docker",
        image="python:3.12",
        runtime_port=8000,
        shell="/bin/bash",
    )

    command = deployment._build_run_command("sandbox-name", 4567, "server", network="host")

    assert command == [
        "docker",
        "run",
        "--rm",
        "-d",
        "--name",
        "sandbox-name",
        "--entrypoint",
        "/bin/bash",
        "--network",
        "host",
        "python:3.12",
        "-lc",
        "server",
    ]


def test_oci_runtime_uses_container_port_when_connecting_over_container_network(monkeypatch):
    commands = []
    deployment = LocalDeployment(
        run_id="test",
        type="local",
        container_runtime="docker",
        image="python:3.12",
        runtime_port=8000,
        network="agent-net",
    )
    deployment._runtime_exec = _capture_exec(commands)
    deployment._get_container_ip = lambda container_name: "172.18.0.9"
    monkeypatch.setattr(local_deployment, "_is_running_in_container", lambda: True)
    monkeypatch.setattr(local_deployment, "LocalRuntime", _CapturingRuntime)
    _CapturingRuntime.configs = []

    asyncio.run(deployment._start_oci_container(token="secret", container_name="sandbox-name", published_port=4567))

    runtime_config, run_id = _CapturingRuntime.configs[-1]
    assert run_id == "test"
    assert runtime_config.host == "http://172.18.0.9"
    assert runtime_config.port == 8000
    assert commands[0] == [
        "docker",
        "run",
        "--rm",
        "-d",
        "--name",
        "sandbox-name",
        "--entrypoint",
        "/bin/bash",
        "--network",
        "agent-net",
        "-p",
        "4567:8000",
        "python:3.12",
        "-lc",
        "python3 -m pip install -q swe-rex && python3 -m swerex.server --host 0.0.0.0 --port 8000 --auth-token secret",
    ]


def test_oci_runtime_uses_published_port_with_custom_network_from_host(monkeypatch):
    deployment = LocalDeployment(
        run_id="test",
        type="local",
        container_runtime="docker",
        image="python:3.12",
        runtime_port=8000,
        network="agent-net",
    )
    deployment._runtime_exec = lambda args, check=True: _completed_container()
    deployment._get_container_ip = lambda container_name: "172.18.0.9"
    monkeypatch.setattr(local_deployment, "_is_running_in_container", lambda: False)
    monkeypatch.setattr(local_deployment, "LocalRuntime", _CapturingRuntime)
    _CapturingRuntime.configs = []

    asyncio.run(deployment._start_oci_container(token="secret", container_name="sandbox-name", published_port=4567))

    runtime_config, run_id = _CapturingRuntime.configs[-1]
    assert run_id == "test"
    assert runtime_config.host == "http://127.0.0.1"
    assert runtime_config.port == 4567


def test_oci_runtime_uses_container_port_when_inheriting_current_container_network(monkeypatch):
    commands = []
    deployment = LocalDeployment(
        run_id="test",
        type="local",
        container_runtime="docker",
        image="python:3.12",
        runtime_port=8000,
    )
    deployment._runtime_exec = _capture_exec(commands)
    deployment._get_current_container_network = lambda: "agent-net"
    deployment._get_container_ip = lambda container_name: "172.18.0.9"
    monkeypatch.setattr(local_deployment, "_is_running_in_container", lambda: True)
    monkeypatch.setattr(local_deployment, "LocalRuntime", _CapturingRuntime)
    _CapturingRuntime.configs = []

    asyncio.run(deployment._start_oci_container(token="secret", container_name="sandbox-name", published_port=4567))

    runtime_config, run_id = _CapturingRuntime.configs[-1]
    assert run_id == "test"
    assert runtime_config.host == "http://172.18.0.9"
    assert runtime_config.port == 8000
    assert commands[0] == [
        "docker",
        "run",
        "--rm",
        "-d",
        "--name",
        "sandbox-name",
        "--entrypoint",
        "/bin/bash",
        "--network",
        "agent-net",
        "-p",
        "4567:8000",
        "python:3.12",
        "-lc",
        "python3 -m pip install -q swe-rex && python3 -m swerex.server --host 0.0.0.0 --port 8000 --auth-token secret",
    ]


def test_oci_runtime_uses_container_port_when_inheriting_host_network(monkeypatch):
    deployment = LocalDeployment(
        run_id="test",
        type="local",
        container_runtime="docker",
        image="python:3.12",
        runtime_port=8000,
    )
    deployment._runtime_exec = lambda args, check=True: _completed_container()
    deployment._get_current_container_network = lambda: "host"
    monkeypatch.setattr(local_deployment, "_is_running_in_container", lambda: True)
    monkeypatch.setattr(local_deployment, "LocalRuntime", _CapturingRuntime)
    _CapturingRuntime.configs = []

    asyncio.run(deployment._start_oci_container(token="secret", container_name="sandbox-name", published_port=4567))

    runtime_config, run_id = _CapturingRuntime.configs[-1]
    assert run_id == "test"
    assert runtime_config.host == "http://127.0.0.1"
    assert runtime_config.port == 8000


def test_oci_runtime_falls_back_to_published_port_when_container_ip_is_unavailable(monkeypatch):
    deployment = LocalDeployment(
        run_id="test",
        type="local",
        container_runtime="docker",
        image="python:3.12",
        runtime_port=8000,
        network="agent-net",
    )
    deployment._runtime_exec = lambda args, check=True: _completed_container()
    deployment._get_container_ip = lambda container_name: None
    monkeypatch.setattr(local_deployment, "_is_running_in_container", lambda: True)
    monkeypatch.setattr(local_deployment, "LocalRuntime", _CapturingRuntime)
    _CapturingRuntime.configs = []

    asyncio.run(deployment._start_oci_container(token="secret", container_name="sandbox-name", published_port=4567))

    runtime_config, run_id = _CapturingRuntime.configs[-1]
    assert run_id == "test"
    assert runtime_config.host == "http://127.0.0.1"
    assert runtime_config.port == 4567


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
