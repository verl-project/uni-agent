import uuid
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict

from uni_agent.async_logging import get_logger
from uni_agent.deployment.vefaas.runtime import RemoteRuntime


class LocalRuntimeConfig(BaseModel):
    auth_token: str
    """The token to use for authentication."""
    host: str = "http://127.0.0.1"
    """The host to connect to."""
    port: int = 8000
    """The port to connect to."""
    timeout: float = 5
    """The timeout for runtime operations."""
    base_url: str | None = None
    """The base URL for remote runtime connection."""
    proxy: str | None = None
    """Optional proxy to use for the connection."""

    type: Literal["local-runtime"] = "local-runtime"
    """Discriminator for local runtime config."""
    model_config = ConfigDict(extra="forbid")

    def get_runtime(self) -> "LocalRuntime":
        return LocalRuntime.from_config(self)


class LocalRuntime(RemoteRuntime):
    def __init__(self, run_id: str, **kwargs: Any):
        self._config = LocalRuntimeConfig(**kwargs)
        self.logger = get_logger("runtime", run_id)
        if not self._config.host.startswith("http"):
            self.logger.warning(f"Host {self._config.host} does not start with http, adding http://")
            self._config.host = f"http://{self._config.host}"

    @classmethod
    def from_config(cls, config: LocalRuntimeConfig, run_id: str | None = None) -> Self:
        if run_id is None:
            run_id = str(uuid.uuid4())
        return cls(run_id=run_id, **config.model_dump())
