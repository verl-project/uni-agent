"""LocalNative deployment: pexpect-based, in-process bash runtime.

Adapted from ``swerex.deployment.local.LocalDeployment``. Use this when you
want to run tools / shell commands directly on the host (no container, no
HTTP layer), and you call ``env.start()`` / ``env.communicate()`` from sync
code via ``auto_await``. See ``runtime.py`` for the rationale.
"""

import uuid
from typing import Any, Self

from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import DeploymentNotStartedError
from swerex.runtime.abstract import (
    CreateBashSessionRequest,
    IsAliveResponse,
)

from uni_agent.async_logging import get_logger
from uni_agent.deployment.config import LocalNativeDeploymentConfig
from uni_agent.deployment.local_native.runtime import LocalNativeRuntime


class LocalNativeDeployment(AbstractDeployment):
    """Deployment that runs tool scripts directly on the host machine.

    Drives a single ``main`` bash session via pexpect, mirroring how SWE-ReX's
    ``LocalRuntime`` is used in-process. Compatible with the framework's
    ``auto_await`` sync-style API: each ``asyncio.run`` invocation can see a
    fresh event loop without re-binding any pexpect state.
    """

    def __init__(self, run_id: str, **kwargs: Any):
        self.run_id = run_id
        self._config = LocalNativeDeploymentConfig(**kwargs)
        self._runtime: LocalNativeRuntime | None = None
        self.logger = get_logger("local-native-deployment", run_id)
        self._hooks = CombinedDeploymentHook()
        self._stopped = False

    def add_hook(self, hook: DeploymentHook):
        self._hooks.add_hook(hook)

    @classmethod
    def from_config(cls, config: LocalNativeDeploymentConfig, run_id: str | None = None) -> Self:
        if not run_id:
            run_id = str(uuid.uuid4())
        return cls(run_id=run_id, **config.model_dump())

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        if self._runtime is None:
            return IsAliveResponse(is_alive=False, message="Runtime is None.")
        return await self._runtime.is_alive(timeout=timeout)

    async def start(self, max_retries: int = 5):
        """Start the runtime and pre-create the default bash session.

        Pre-creating the session whose name matches ``BashAction.session``'s
        default lets callers issue ``BashAction(command=...)`` without
        managing session names.
        """
        self._runtime = LocalNativeRuntime(run_id=self.run_id)
        await self._runtime.create_session(
            CreateBashSessionRequest(
                session=CreateBashSessionRequest.model_fields["session"].default,
                startup_source=[],
                startup_timeout=self._config.startup_timeout,
            )
        )
        self._stopped = False
        self.logger.info("Local-native deployment started")

    async def stop(self):
        if self._stopped:
            return
        if self._runtime is not None:
            try:
                await self._runtime.close()
            except Exception as exc:
                self.logger.error(f"Failed to close local-native runtime: {exc}")
            self._runtime = None
        self._stopped = True
        self.logger.info("Local-native deployment stopped")

    @property
    def runtime(self) -> LocalNativeRuntime:
        if self._runtime is None:
            raise DeploymentNotStartedError()
        return self._runtime

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
