import asyncio
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING, Any, Self

from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import DeploymentNotStartedError
from swerex.runtime.abstract import CreateBashSessionRequest, IsAliveResponse
from swerex.utils.wait import _wait_until_alive

from uni_agent.async_logging import get_logger
from uni_agent.deployment.config import VefaasDeploymentConfig

if TYPE_CHECKING:
    from uni_agent.deployment.remote_runtime import RemoteRuntime

PUB_VOLCES_IMG_URL_TEMPLATE = {
    "swe-bench": (
        "enterprise-public-cn-beijing.cr.volces.com"
        "/swe-bench/sweb.eval.x86_64.{project_name}_1776_{instance_number}:latest"
    ),
    "swe-bench-verified": (
        "enterprise-public-cn-beijing.cr.volces.com"
        "/swe-bench-verified/sweb.eval.x86_64.{project_name}_1776_{instance_number}:v2"
    ),
    "swe-rebench": (
        "enterprise-public-2-cn-beijing.cr.volces.com/swe-rebench/{project_name}_1776_{instance_number}:latest"
    ),
    "swe-bench-live": (
        "enterprise-public-cn-beijing.cr.volces.com/swe-bench-live/{project_name}_1776_{instance_number}:latest"
    ),
    "r2e-gym-subset": "enterprise-public-cn-beijing.cr.volces.com/r2e-gym-subset/{instance_number}:latest",
}


def get_vefaas_image_name(dataset_id: str, instance_id: str) -> str:
    assert dataset_id in PUB_VOLCES_IMG_URL_TEMPLATE, (
        f"only support {list(PUB_VOLCES_IMG_URL_TEMPLATE.keys())}, got {dataset_id}"
    )
    parts = instance_id.split("__")
    assert len(parts) == 2
    project_name = parts[0].lower()
    instance_number = parts[1].lower()

    if dataset_id in ["swe-bench", "swe-bench-verified", "swe-bench-live", "swe-rebench"]:
        return PUB_VOLCES_IMG_URL_TEMPLATE[dataset_id].format(
            project_name=project_name,
            instance_number=instance_number,
        )
    elif dataset_id == "r2e-gym-subset":
        return PUB_VOLCES_IMG_URL_TEMPLATE[dataset_id].format(instance_number=instance_number)
    else:
        assert dataset_id in PUB_VOLCES_IMG_URL_TEMPLATE, (
            f"only support {list(PUB_VOLCES_IMG_URL_TEMPLATE.keys())}, got {dataset_id}"
        )


class VefaasDeployment(AbstractDeployment):
    def __init__(self, run_id: str, **kwargs: Any):
        from dotenv import load_dotenv

        load_dotenv()
        self.run_id = run_id
        self._config = VefaasDeploymentConfig(**kwargs)
        self._runtime: Any | None = None
        self.logger = get_logger("deployment", run_id)
        self._hooks = CombinedDeploymentHook()
        self._sandbox_id: str | None = None
        self._stopped: bool = False

        access_key = os.getenv("VOLCE_ACCESS_KEY") or os.getenv("VOLCENGINE_ACCESS_KEY")
        secret_key = os.getenv("VOLCE_SECRET_KEY") or os.getenv("VOLCENGINE_SECRET_KEY")
        region = os.getenv("VEFAAS_REGION", "cn-beijing")
        if not all([access_key, secret_key, region]):
            raise ValueError("VOLCE_ACCESS_KEY, VOLCE_SECRET_KEY, and VEFAAS_REGION must be set")
        self._vefaas_client = get_vefaas_client(access_key, secret_key, region, self._config.proxy)

    def add_hook(self, hook: DeploymentHook):
        self._hooks.add_hook(hook)

    @classmethod
    def from_config(cls, config: VefaasDeploymentConfig, run_id: str | None = None) -> Self:
        if not run_id:
            run_id = str(uuid.uuid4())

        return cls(run_id=run_id, **config.model_dump())

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        if self._runtime is None:
            raise DeploymentNotStartedError("Runtime not started")
        return await self._runtime.is_alive(timeout=timeout)

    async def _wait_until_alive(self, timeout: float = 10.0):
        try:
            return await _wait_until_alive(self.is_alive, timeout=timeout, function_timeout=0.5)
        except TimeoutError as e:
            self.logger.error("Runtime did not start within timeout.")
            await self.stop()
            raise e

    def _get_token(self) -> str:
        return str(uuid.uuid4())

    async def start(self, max_retries: int = 5) -> None:
        from uni_agent.deployment.remote_runtime import RemoteRuntime, RemoteRuntimeConfig

        self.logger.info(
            f"Starting vefaas deployment,function_id = {self._config.function_id},image = {self._config.image}."
        )
        function_id = self._config.function_id or os.getenv("VEFAAS_FUNCTION_ID")
        if not function_id:
            raise ValueError("VEFAAS_FUNCTION_ID environment variable not set")

        image = self._config.image
        if not image:
            raise ValueError("No image specified and no image list provided")

        token = self._get_token()
        command = self._config.command.format(token=token)

        self.logger.info(f"Creating sandbox with image {image}, command = {command}")
        self._hooks.on_custom_step("Creating vefaas sandbox")
        loop = asyncio.get_running_loop()
        create_sanbox_done = False

        for retry in range(max_retries):
            try:
                self._sandbox_id = await loop.run_in_executor(
                    None,
                    create_sandbox,
                    self._vefaas_client,
                    function_id,
                    image,
                    command,
                    self.logger,
                )
                if self._sandbox_id:
                    create_sanbox_done = True
                    break
            except Exception as e:
                self.logger.critical(f"Failed to create sandbox: {e}")
                sleep_time = min(30, 2**retry)
                self.logger.info(f"Retrying in {sleep_time} seconds...")
                await asyncio.sleep(sleep_time)

        if not create_sanbox_done:
            raise RuntimeError(f"Failed to create sandbox after {max_retries} retries")

        self.logger.info(f"Sandbox {self._sandbox_id} created")
        self._hooks.on_custom_step("Starting runtime")

        function_route = self._config.function_route or os.getenv("VEFAAS_FUNCTION_ROUTE")
        if not function_route:
            raise ValueError("VEFAAS_FUNCTION_ROUTE environment variable not set")

        runtime_config = RemoteRuntimeConfig(
            base_url=function_route,
            extra_params={"faasInstanceName": self._sandbox_id},
            auth_token=token,
            timeout=self._config.timeout,
            proxy=self._config.proxy,
        )
        self._runtime = RemoteRuntime.from_config(runtime_config, run_id=self.run_id)

        # await self._wait_until_alive(timeout=self._config.startup_timeout)
        await self.runtime.create_session(
            CreateBashSessionRequest(startup_source=["/root/.bashrc"], startup_timeout=60)
        )
        # await self._post_setup()

    async def stop(self):
        # Prevent duplicate stops
        if getattr(self, "_stopped", False):
            return

        if self._runtime:
            try:
                await self._runtime.close()
            except Exception as e:
                self.logger.error(f"Failed to close vefaas runtime within timeout: {e}")
            self._runtime = None

        if self._sandbox_id:
            self.logger.info(f"Deleting sandbox {self._sandbox_id}")
            function_id = self._config.function_id or os.getenv("VEFAAS_FUNCTION_ID")
            if not function_id:
                self.logger.error("VEFAAS_FUNCTION_ID not set, cannot delete sandbox")
                return

            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    delete_sandbox,
                    self._vefaas_client,
                    function_id,
                    self._sandbox_id,
                    self.logger,
                )
                self.logger.info(f"Sandbox {self._sandbox_id} deleted")
            except Exception as e:
                self.logger.error(f"Failed to delete sandbox {self._sandbox_id}: {e}")
            finally:
                self._sandbox_id = None

        self._stopped = True

    @property
    def runtime(self) -> "RemoteRuntime":
        if self._runtime is None:
            raise DeploymentNotStartedError()
        return self._runtime

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    def __del__(self):
        if hasattr(self, "_sandbox_id") and self._sandbox_id and not getattr(self, "_stopped", False):
            msg = "Ensuring vefaas deployment is stopped because object is deleted"
            try:
                self.logger.debug(msg)
            except Exception:
                print(msg)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.stop())
                else:
                    loop.run_until_complete(self.stop())
            except Exception:
                pass
        # Mark as stopped to prevent duplicate cleanup
        self._stopped = True


def get_vefaas_client(access_key: str, secret_key: str, region: str, proxy: str | None = None):
    import volcenginesdkcore
    import volcenginesdkvefaas

    configuration = volcenginesdkcore.Configuration()
    configuration.ak = access_key
    configuration.sk = secret_key
    configuration.read_timeout = 60
    configuration.connect_timeout = 60
    configuration.auto_retry = False
    configuration.region = region
    configuration.client_side_validation = True
    if proxy:
        configuration.http_proxy = proxy
        configuration.https_proxy = proxy
    api_client = volcenginesdkcore.ApiClient(configuration)
    return volcenginesdkvefaas.VEFAASApi(api_client)


def create_sandbox(
    client: Any,
    function_id: str,
    image: str,
    command: str,
    logger: logging.Logger,
) -> str | None:
    import volcenginesdkvefaas

    if image.startswith("swebench/"):
        image_name = image.replace("swebench/", "", 1)
        image = f"enterprise-public-cn-beijing.cr.volces.com/swe-bench/{image_name}"

    instance_image_info = volcenginesdkvefaas.InstanceImageInfoForCreateSandboxInput(
        image=image,
        port=8000,  # swerex server port
        command=command,
    )
    start_time = time.time()
    try:
        resp = client.create_sandbox(
            volcenginesdkvefaas.CreateSandboxRequest(
                function_id=function_id,
                instance_image_info=instance_image_info,
                timeout=120,  # 2 hours
            )
        )
        end_time = time.time()
        logger.info(f"Sandbox {resp.sandbox_id} created in {end_time - start_time:.2f}s")
        return resp.sandbox_id
    except Exception as e:
        end_time = time.time()
        logger.error(f"Sandbox creation for {image} failed in {end_time - start_time:.2f}s: {e}")
        return None


def delete_sandbox(
    client: Any,
    function_id: str,
    sandbox_id: str,
    logger: logging.Logger,
):
    import volcenginesdkvefaas
    from volcenginesdkcore.rest import ApiException

    if sandbox_id is None:
        return
    try:
        client.kill_sandbox(
            volcenginesdkvefaas.KillSandboxRequest(
                function_id=function_id,
                sandbox_id=sandbox_id,
            )
        )
    except ApiException as e:
        logger.error(f"Exception when deleting sandbox {sandbox_id}: {e}")
