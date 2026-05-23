"""LocalAttach deployment: attach to a user-managed swerex server.

Unlike ``LocalDeployment`` (which ``docker run``s a fresh sandbox per agent
run), this deployment **does not** start, stop, or otherwise manage a
container. The user is responsible for launching a container ahead of time,
running ``swerex.server`` inside it, and exposing it on a reachable host/port.

What the framework still does:

- ``start()`` opens an HTTP client to the existing swerex server, verifies it
  is alive, and (re)creates the default bash session (whose name comes from
  ``CreateBashSessionRequest``'s default, currently ``"default"``).
  **No container is started.**
- ``stop()`` is a no-op. The user-managed container is left running and
  untouched.

Because the sandbox is shared across runs, ``start()`` closes any stale
session under that default name before re-creating it so each agent run
starts on a clean shell.

Typical usage::

    # User side, manually (once per sandbox):
    docker run -d --name my-sandbox -p 18000:8000 \\
      -v ~/.uni-agent/docker-lark-auth:/root/.lark-cli \\
      node:20-bookworm bash -lc \\
        'npm install -g -q @larksuite/cli &&
         pip install -q swe-rex &&
         python3 -m swerex.server --host 0.0.0.0 --port 8000 --auth-token mytoken'

    # Framework side, every agent run -- identical to any other deployment:
    env_config = AgentEnvConfig(deployment={
        "type": "local_attach",
        "host": "http://127.0.0.1",
        "port": 18000,
        "auth_token": "mytoken",
    })
    env = AgentEnv(run_id=..., env_config=env_config)
    env.start()              # attaches to swerex, creates session; no container started
    env.install_tools(...)
    ...
    env.close()              # no-op for container; sandbox keeps running
"""

import uuid
from typing import Any, Self

from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import (
    DeploymentNotStartedError,
    SessionDoesNotExistError,
    SessionExistsError,
)
from swerex.runtime.abstract import (
    CloseSessionRequest,
    CreateBashSessionRequest,
    IsAliveResponse,
)
from swerex.utils.wait import _wait_until_alive

from uni_agent.async_logging import get_logger
from uni_agent.deployment.config import LocalAttachDeploymentConfig
from uni_agent.deployment.remote_runtime import RemoteRuntime, RemoteRuntimeConfig


class LocalAttachDeployment(AbstractDeployment):
    """Attach to an externally-managed swerex server. See module docstring."""

    def __init__(self, run_id: str, **kwargs: Any):
        self.run_id = run_id
        self._config = LocalAttachDeploymentConfig(**kwargs)
        self._runtime: RemoteRuntime | None = None
        self.logger = get_logger("local-attach-deployment", run_id)
        self._hooks = CombinedDeploymentHook()

    def add_hook(self, hook: DeploymentHook) -> None:
        self._hooks.add_hook(hook)

    @classmethod
    def from_config(cls, config: LocalAttachDeploymentConfig, run_id: str | None = None) -> Self:
        if not run_id:
            run_id = str(uuid.uuid4())
        return cls(run_id=run_id, **config.model_dump())

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        if self._runtime is None:
            raise DeploymentNotStartedError(
                "LocalAttach deployment is not started yet; call deployment.start() before use"
            )
        return await self._runtime.is_alive(timeout=timeout)

    async def start(self, max_retries: int = 5) -> None:  # noqa: ARG002
        """Attach to the user-managed swerex server.

        Does **not** start a container. Only opens the HTTP client, verifies
        the server is alive, and (re)creates a clean default bash session
        (name resolved from ``CreateBashSessionRequest``'s default).
        Idempotent: safe to call multiple times across runs.

        Raises:
            RuntimeError: if the server is not reachable within
                ``startup_timeout``.
        """
        if self._runtime is None:
            runtime_config = RemoteRuntimeConfig(
                auth_token=self._config.auth_token,
                host=self._config.host,
                port=self._config.port,
                timeout=self._config.timeout,
                proxy=self._config.proxy,
            )
            self._runtime = RemoteRuntime.from_config(runtime_config, run_id=self.run_id)
            self.logger.info(
                f"LocalAttach: client initialized for {self._config.host}:{self._config.port} (no container started)"
            )

        try:
            await _wait_until_alive(
                self._runtime.is_alive,
                timeout=self._config.startup_timeout,
                function_timeout=0.5,
            )
        except TimeoutError as exc:
            raise RuntimeError(
                f"LocalAttach: swerex at {self._config.host}:{self._config.port} not reachable within "
                f"{self._config.startup_timeout}s. Is the user-managed container running and exposing the port?"
            ) from exc

        session_name = CreateBashSessionRequest.model_fields["session"].default
        try:
            await self._runtime.close_session(CloseSessionRequest(session=session_name))
            self.logger.debug(f"LocalAttach: closed stale session {session_name!r} from previous run")
        except SessionDoesNotExistError:
            pass
        except Exception as exc:
            self.logger.debug(f"LocalAttach: stale session close failed (continuing): {exc}")

        try:
            await self._runtime.create_session(
                CreateBashSessionRequest(startup_source=["/root/.bashrc"], startup_timeout=60)
            )
        except SessionExistsError:
            self.logger.debug(f"LocalAttach: session {session_name!r} already exists, reusing")

        self.logger.info(f"LocalAttach: ready (session={session_name!r})")

    async def stop(self) -> None:
        """No-op. The user-managed container is left running and untouched.

        Kept as an empty method only to satisfy ``AbstractDeployment``.
        """
        return None

    @property
    def runtime(self) -> RemoteRuntime:
        if self._runtime is None:
            raise DeploymentNotStartedError(
                "LocalAttach deployment is not started yet; call deployment.start() before accessing .runtime"
            )
        return self._runtime

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None
