from pathlib import PurePath
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class HostDeploymentConfig(BaseModel):
    """Configuration for host-local execution (no container)."""

    type: Literal["host"] = "host"
    """Discriminator for (de)serialization. Do not change."""
    timeout: float = 60.0
    """Default timeout for runtime operations."""
    startup_timeout: float = 120.0
    """Timeout for the initial bash session handshake.

    During parameter-sync weight reloads, fork()/exec() and even the asyncio event loop can be
    starved for tens of seconds.
    """

    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        from .host.deployment import HostDeployment

        return HostDeployment.from_config(self, run_id)


class LocalNativeDeploymentConfig(BaseModel):
    """Configuration for in-process pexpect-based host execution.

    Like ``HostDeploymentConfig`` this runs commands directly on the host (no
    container), but drives bash via ``pexpect`` / PTY rather than
    ``asyncio.create_subprocess_exec``. Compatible with the framework's
    sync-style ``auto_await`` API. See
    ``uni_agent/deployment/local_native/runtime.py`` for details.
    """

    type: Literal["local_native"] = "local_native"
    """Discriminator for (de)serialization. Do not change."""
    timeout: float = 60.0
    """Default timeout for runtime operations."""
    startup_timeout: float = 120.0
    """Timeout for the initial bash session handshake."""

    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        from .local_native.deployment import LocalNativeDeployment

        return LocalNativeDeployment.from_config(self, run_id)


class LocalDeploymentConfig(BaseModel):
    """Configuration for a local sandbox."""

    image: str = "python:3.12"
    """Container image used for the sandbox."""
    command: str = (
        "python3 -m pip install -q swe-rex && "
        "python3 -m swerex.server --host 0.0.0.0 --port {port} --auth-token {token}"
    )
    """Command to run inside the sandbox."""
    timeout: float = 60.0
    """Timeout for runtime operations."""
    startup_timeout: float = 180.0
    """Timeout waiting for runtime to start."""
    container_runtime: str = "apptainer"
    """Container runtime executable. If omitted by the user, local deployment discovers one at startup."""
    container_name: str | None = None
    """Optional container name override."""
    host: str | None = None
    """Override the runtime host. Defaults to localhost outside containers and container IP inside containers."""
    published_port: int | None = None
    """Host port mapped to the sandbox runtime port. If unset, a free local port is chosen."""
    runtime_port: int = 8000
    """Port exposed by the swerex server inside the sandbox."""
    network: str | None = None
    """Optional Docker network to attach the sandbox to."""
    shell: str = "/bin/bash"
    """Shell executable used as the container entrypoint."""
    extra_run_args: list[str] = Field(default_factory=list)
    """Extra args appended to the container runtime startup command."""

    type: Literal["local"] = "local"
    """Discriminator for (de)serialization/CLI. Do not change."""
    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        from .local.deployment import LocalDeployment

        return LocalDeployment.from_config(self, run_id)


class ModalDeploymentConfig(BaseModel):
    """Configuration for Modal deployment."""

    image: str | PurePath = "python:3.11"
    """Image to use for the deployment."""
    startup_timeout: float = 180.0
    """Timeout waiting for runtime to start."""
    runtime_timeout: float = 60.0
    """Timeout for runtime operations."""
    deployment_timeout: float = 3600.0
    """Timeout for the Modal sandbox."""
    modal_sandbox_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Additional keyword arguments passed to `modal.Sandbox.create`."""
    proxy: str | None = None
    """Proxy to use for runtime HTTP requests."""
    type: Literal["modal"] = "modal"
    """Discriminator for (de)serialization/CLI. Do not change."""
    install_pipx: bool = True
    """Whether to install pipx in the Modal image."""

    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        from .modal.deployment import ModalDeployment

        return ModalDeployment.from_config(self, run_id)


class VefaasDeploymentConfig(BaseModel):
    """Configuration for veFaaS deployment."""

    image: str | None = None
    """Docker image to use for the sandbox."""
    command: str = "python3 -m swerex.server --auth-token {token}"
    """Command to run in the sandbox with authentication token."""
    timeout: float = 60.0
    """Timeout for runtime operations."""
    startup_timeout: float = 120.0
    """Timeout waiting for runtime to start."""
    function_id: str | None = None
    """veFaaS function ID."""
    function_route: str | None = None
    """veFaaS function route."""
    proxy: str | None = None
    """Proxy to use for the connection."""

    type: Literal["vefaas"] = "vefaas"
    """Discriminator for (de)serialization/CLI. Do not change."""
    model_config = ConfigDict(extra="forbid")

    def get_deployment(self, run_id: str):
        from .vefaas.deployment import VefaasDeployment

        return VefaasDeployment.from_config(self, run_id)


DeployConfig: TypeAlias = Annotated[
    VefaasDeploymentConfig
    | LocalDeploymentConfig
    | HostDeploymentConfig
    | LocalNativeDeploymentConfig
    | ModalDeploymentConfig,
    Field(discriminator="type"),
]
