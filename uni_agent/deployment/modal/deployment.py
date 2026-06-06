import asyncio
import logging
import os
import time
import uuid
from pathlib import Path, PurePath
from typing import Any, Self

import boto3
import modal
from botocore.exceptions import NoCredentialsError
from swerex import PACKAGE_NAME, REMOTE_EXECUTABLE_NAME
from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import DeploymentNotStartedError
from swerex.runtime.abstract import CreateBashSessionRequest, IsAliveResponse
from swerex.utils.wait import _wait_until_alive

from uni_agent.async_logging import get_logger
from uni_agent.deployment.config import ModalDeploymentConfig
from uni_agent.deployment.remote_runtime import RemoteRuntime, RemoteRuntimeConfig

__all__ = ["ModalDeployment"]


# Cap how many Modal sandboxes are simultaneously in the "created but runtime
# not yet alive" state; too many at once cause "Runtime did not start" timeouts.
# The semaphore is process-local, so MODAL_MAX_STARTING_PER_WORKER is a
# per-worker cap (size it as fleet-wide target / num rollout workers).
# MODAL_INIT_WALL_BUDGET caps a single trajectory's total init wall-clock.
_DEFAULT_MAX_STARTING_PER_WORKER = 64
_DEFAULT_INIT_WALL_BUDGET = 900.0
_STARTING_SEMA: asyncio.Semaphore | None = None


def _get_starting_semaphore() -> asyncio.Semaphore:
    """Lazy-init the per-worker STARTING semaphore.

    Lazy so it is built inside the running loop and the env var is read at
    first use, not at import (so vars set after import still apply).
    """
    global _STARTING_SEMA
    if _STARTING_SEMA is None:
        per_worker = max(1, int(os.getenv("MODAL_MAX_STARTING_PER_WORKER", str(_DEFAULT_MAX_STARTING_PER_WORKER))))
        _STARTING_SEMA = asyncio.Semaphore(per_worker)
    return _STARTING_SEMA


def _get_init_wall_budget() -> float:
    """Per-trajectory init wall-clock budget (seconds), resolved at call time."""
    return float(os.getenv("MODAL_INIT_WALL_BUDGET", str(_DEFAULT_INIT_WALL_BUDGET)))


def _get_modal_user() -> str:
    # not sure how to get the user from the modal api
    return modal.config._profile  # type: ignore


class _ImageBuilder:
    """_ImageBuilder.auto() is used by ModalDeployment."""

    def __init__(self, *, install_pipx: bool = True, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("modal_image_builder")
        self._install_pipx = install_pipx

    def from_file(self, image: PurePath, *, build_context: PurePath | None = None) -> modal.Image:
        self.logger.info(f"Building image from file {image}")
        if build_context is None:
            build_context = Path(image).resolve().parent
        build_context = Path(build_context)
        self.logger.debug(f"Using build context {build_context}")
        return modal.Image.from_dockerfile(
            str(image),
            context_dir=str(build_context),
        )

    def from_registry(self, image: str) -> modal.Image:
        self.logger.info(f"Building image from docker registry {image}")
        if os.environ.get("DOCKER_USERNAME") and os.environ.get("DOCKER_PASSWORD"):
            secret = modal.Secret.from_dict(
                {
                    "DOCKER_USERNAME": os.environ["DOCKER_USERNAME"],
                    "DOCKER_PASSWORD": os.environ["DOCKER_PASSWORD"],
                }
            )
            secrets = [secret]
            self.logger.debug("Docker login credentials were provided")
        else:
            self.logger.warning("DOCKER_USERNAME and DOCKER_PASSWORD not set. Using public images.")
            secrets = None
        return modal.Image.from_registry(image, secrets=secrets)

    def from_ecr(self, image: str) -> modal.Image:
        self.logger.info(f"Building image from ECR {image}")
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            aws_access_key_id = credentials.access_key
            aws_secret_access_key = credentials.secret_key
            secret = modal.Secret.from_dict(
                {
                    "AWS_ACCESS_KEY_ID": aws_access_key_id,
                    "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
                }
            )
            return modal.Image.from_ecr(  # type: ignore
                image,
                secrets=[secret],
            )
        except NoCredentialsError as e:
            msg = "AWS credentials not found. Please configure your AWS credentials."
            raise ValueError(msg) from e

    def ensure_pipx_installed(self, image: modal.Image) -> modal.Image:
        image = image.apt_install("pipx", env={"DEBIAN_FRONTEND": "noninteractive"})
        return image.run_commands("pipx ensurepath")

    def auto(self, image_spec: str | modal.Image | PurePath) -> modal.Image:
        if isinstance(image_spec, modal.Image):
            image = image_spec
        elif isinstance(image_spec, PurePath) and not Path(image_spec).is_file():
            msg = f"File {image_spec} does not exist"
            raise FileNotFoundError(msg)
        elif Path(image_spec).is_file():
            image = self.from_file(Path(image_spec))
        elif "amazonaws.com" in image_spec:  # type: ignore
            image = self.from_ecr(image_spec)  # type: ignore
        else:
            image = self.from_registry(image_spec)  # type: ignore

        if self._install_pipx:
            image = self.ensure_pipx_installed(image)

        return image


class ModalDeployment(AbstractDeployment):
    # Leave the constructor args for now, because image can take a modal.Image
    # but we don't want to make this part of the config class because it would
    # force us to have modal installed/import it.
    def __init__(
        self,
        run_id: str,
        *,
        logger: logging.Logger | None = None,
        image: str | modal.Image | PurePath,
        startup_timeout: float = 300.0,
        runtime_timeout: float = 60.0,
        modal_sandbox_kwargs: dict[str, Any] | None = None,
        install_pipx: bool = True,
        deployment_timeout: float = 3600.0,
        proxy: str | None = None,
    ):
        """Deployment for modal.com. The deployment starts when `start` is called."""
        self.run_id = run_id
        self.logger = logger or get_logger("deployment", run_id)
        self._image_name = str(image)
        self._image = _ImageBuilder(install_pipx=install_pipx, logger=self.logger).auto(image)
        self._runtime: RemoteRuntime | None = None
        self._startup_timeout = startup_timeout
        self._sandbox: modal.Sandbox | None = None
        self._port = 8880
        self._app: modal.App | None = None
        self._user = _get_modal_user()
        self._runtime_timeout = runtime_timeout
        self._deployment_timeout = deployment_timeout
        self._proxy = proxy
        if modal_sandbox_kwargs is None:
            modal_sandbox_kwargs = {}
        modal_sandbox_kwargs = dict(modal_sandbox_kwargs)
        for _key in ("cpu", "memory"):
            if isinstance(modal_sandbox_kwargs.get(_key), list):
                modal_sandbox_kwargs[_key] = tuple(modal_sandbox_kwargs[_key])
        self._modal_kwargs = modal_sandbox_kwargs
        self._hooks = CombinedDeploymentHook()

    def add_hook(self, hook: DeploymentHook):
        self._hooks.add_hook(hook)

    @classmethod
    def from_config(cls, config: ModalDeploymentConfig, run_id: str | None = None) -> Self:
        if run_id is None:
            run_id = str(uuid.uuid4())
        return cls(
            run_id=run_id,
            image=config.image,
            install_pipx=config.install_pipx,
            startup_timeout=config.startup_timeout,
            runtime_timeout=config.runtime_timeout,
            deployment_timeout=config.deployment_timeout,
            modal_sandbox_kwargs=config.modal_sandbox_kwargs,
            proxy=config.proxy,
        )

    def _get_token(self) -> str:
        return str(uuid.uuid4())

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        """Checks if the runtime is alive. The return value can be tested with bool()."""
        if self._runtime is None or self._sandbox is None:
            raise DeploymentNotStartedError()
        exit_code = await self._sandbox.poll.aio()
        if exit_code is not None:
            msg = "Container process terminated."
            output = "stdout:\n" + await self._sandbox.stdout.read.aio()  # type: ignore
            output += "\nstderr:\n" + await self._sandbox.stderr.read.aio()  # type: ignore
            msg += "\n" + output
            raise RuntimeError(msg)
        return await self._runtime.is_alive(timeout=timeout)

    async def _wait_until_alive(self, timeout: float = 10.0):
        assert self._runtime is not None
        return await _wait_until_alive(self.is_alive, timeout=timeout, function_timeout=self._runtime._config.timeout)

    def _start_swerex_cmd(self, token: str) -> str:
        """Start swerex-server on the remote. If swerex is not installed, use pipx."""
        rex_args = f"--port {self._port} --auth-token {token}"
        return f"{REMOTE_EXECUTABLE_NAME} {rex_args} || pipx run {PACKAGE_NAME} {rex_args}"

    async def get_modal_log_url(self) -> str:
        """Returns URL to modal logs."""
        task_id = await self.sandbox._get_task_id.aio()
        return f"https://modal.com/apps/{self._user}/main/deployed/{self.app.name}?activeTab=logs&taskId={task_id}"

    async def _start(self):
        """Starts the runtime once."""
        if self._runtime is not None and self._sandbox is not None:
            self.logger.warning("Deployment is already started. Ignoring duplicate start() call.")
            return

        if self._app is None:
            self._app = await modal.App.lookup.aio("swe-rex", create_if_missing=True)

        # Hold the STARTING permit only through runtime startup; the tool-call
        # body afterwards is LLM-bound and must not occupy a permit.
        async with _get_starting_semaphore():
            self.logger.info(f"Starting modal sandbox with image {self._image_name}")
            self._hooks.on_custom_step("Starting modal sandbox")
            t0 = time.time()
            token = self._get_token()
            self._sandbox = await modal.Sandbox.create.aio(
                "/usr/bin/env",
                "bash",
                "-c",
                self._start_swerex_cmd(token),
                image=self._image,
                timeout=int(self._deployment_timeout),
                encrypted_ports=[self._port],
                app=self._app,
                **self._modal_kwargs,
            )
            tunnels = await self._sandbox.tunnels.aio()
            tunnel = tunnels[self._port]
            elapsed_sandbox_creation = time.time() - t0
            self.logger.info(f"Sandbox ({self._sandbox.object_id}) created in {elapsed_sandbox_creation:.2f}s")
            self.logger.info(f"Check sandbox logs at {await self.get_modal_log_url()}")
            self.logger.info(f"Sandbox created with id {self._sandbox.object_id}")
            await asyncio.sleep(1)
            self.logger.info(f"Starting runtime at {tunnel.url}")
            self._hooks.on_custom_step("Starting runtime")
            runtime_config = RemoteRuntimeConfig(
                host=tunnel.url,
                timeout=self._runtime_timeout,
                auth_token=token,
                proxy=self._proxy,
            )
            self._runtime = RemoteRuntime.from_config(runtime_config, run_id=self.run_id)
            remaining_startup_timeout = max(0, self._startup_timeout - elapsed_sandbox_creation)
            t1 = time.time()
            await self._wait_until_alive(timeout=remaining_startup_timeout)
            await self.runtime.create_session(CreateBashSessionRequest(startup_timeout=60))
            self.logger.info(f"Runtime started in {time.time() - t1:.2f}s")

    async def start(self, max_retries: int = 2):
        """Start the runtime with retry, bounded by MODAL_INIT_WALL_BUDGET.

        Few retries + a wall-clock cap stop a stuck trajectory from holding a
        STARTING permit for many minutes; on exhaustion it raises and the outer
        agent_loop turns it into a reward=0 masked sample.
        """
        last_error: Exception | None = None
        wall_budget = _get_init_wall_budget()
        deadline = time.monotonic() + wall_budget
        for retry in range(max_retries):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.logger.critical(f"Wall-clock budget {wall_budget}s exhausted before attempt {retry + 1}")
                break
            try:
                await asyncio.wait_for(self._start(), timeout=max(60.0, remaining))
                return
            except Exception as exc:
                last_error = exc
                self.logger.critical(f"Failed to create modal sandbox: {exc}")
                await self.stop()
                if retry < max_retries - 1 and time.monotonic() < deadline:
                    sleep_time = min(10, 2**retry)
                    self.logger.info(f"Retrying modal deployment startup in {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time)

        raise RuntimeError(
            f"Failed to create modal sandbox after {max_retries} retries (wall budget {wall_budget}s)"
        ) from last_error

    async def stop(self):
        """Stop the runtime, best-effort.

        Each step is wrapped so a transient failure (e.g. runtime.close raising
        when the socket is already gone) never skips sandbox terminate -- a
        leaked sandbox counts against the account's concurrent-sandbox cap.
        """
        if self._runtime is not None:
            try:
                await self._runtime.close()
            except Exception as exc:
                self.logger.warning(f"runtime.close() swallowed (continuing teardown): {type(exc).__name__}: {exc}")
            self._runtime = None

        # CRITICAL — must always run to avoid leaking the modal sandbox.
        if self._sandbox is not None:
            try:
                exit_code = await self._sandbox.poll.aio()
                if exit_code is None:
                    await self._sandbox.terminate.aio()
            except Exception as exc:
                self.logger.warning(
                    f"sandbox poll/terminate first attempt failed: "
                    f"{type(exc).__name__}: {exc}; retrying terminate once."
                )
                try:
                    await self._sandbox.terminate.aio()
                except Exception as exc2:
                    self.logger.error(
                        f"sandbox.terminate.aio() retry also failed: {type(exc2).__name__}: {exc2}. Sandbox may leak."
                    )
            self._sandbox = None

        self._app = None

    @property
    def runtime(self) -> RemoteRuntime:
        """Returns the runtime if running."""
        if self._runtime is None:
            raise DeploymentNotStartedError()
        return self._runtime

    @property
    def app(self) -> modal.App:
        """Returns the modal app."""
        if self._app is None:
            raise DeploymentNotStartedError()
        return self._app

    @property
    def sandbox(self) -> modal.Sandbox:
        """Returns the modal sandbox."""
        if self._sandbox is None:
            raise DeploymentNotStartedError()
        return self._sandbox
