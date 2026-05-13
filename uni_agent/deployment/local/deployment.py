import asyncio
import os
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Self

from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import DeploymentNotStartedError
from swerex.runtime.abstract import Command, CreateBashSessionRequest, IsAliveResponse, UploadRequest
from swerex.utils.wait import _wait_until_alive

from uni_agent.async_logging import get_logger
from uni_agent.deployment.config import LocalDeploymentConfig
from uni_agent.deployment.remote_runtime import RemoteRuntime as LocalRuntime
from uni_agent.deployment.remote_runtime import RemoteRuntimeConfig as LocalRuntimeConfig

_APPTAINER_RUNTIMES = {"apptainer", "singularity"}
_CONTAINER_RUNTIME_ENV_VARS = ("UNI_AGENT_CONTAINER_RUNTIME", "LOCAL_CONTAINER_RUNTIME")
_DEFAULT_CONTAINER_RUNTIME_CANDIDATES = ("apptainer", "singularity", "docker", "podman")
_IMAGE_URI_PREFIXES = (
    "docker://",
    "oras://",
    "library://",
    "shub://",
    "instance://",
    "http://",
    "https://",
    "file://",
)


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()
    return sanitized or "uni-agent-local"


def _is_running_in_container() -> bool:
    return Path("/.dockerenv").exists()


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _default_container_runtime() -> str:
    for env_var in _CONTAINER_RUNTIME_ENV_VARS:
        runtime = os.getenv(env_var)
        if runtime:
            return runtime
    for runtime in _DEFAULT_CONTAINER_RUNTIME_CANDIDATES:
        runtime_path = shutil.which(runtime)
        if runtime_path:
            return runtime_path
    return "apptainer"


def _runtime_basename(runtime: str) -> str:
    return Path(runtime).name.lower()


def _is_apptainer_runtime(runtime: str) -> bool:
    return _runtime_basename(runtime) in _APPTAINER_RUNTIMES


def _normalize_apptainer_image(image: str) -> str:
    if image.startswith(_IMAGE_URI_PREFIXES):
        return image
    image_path = Path(image)
    if image.startswith(("/", ".")) or image_path.exists() or image_path.suffix in {".sif", ".sqsh", ".img"}:
        return image
    return f"docker://{image}"


class LocalDeployment(AbstractDeployment):
    def __init__(self, run_id: str, **kwargs: Any):
        self.run_id = run_id
        config_kwargs = dict(kwargs)
        if not config_kwargs.get("container_runtime"):
            config_kwargs["container_runtime"] = _default_container_runtime()
        self._config = LocalDeploymentConfig(**config_kwargs)
        self._runtime: LocalRuntime | None = None
        self.logger = get_logger("deployment", run_id)
        self._hooks = CombinedDeploymentHook()
        self._container_name: str | None = None
        self._container_id: str | None = None
        self._server_process: subprocess.Popen[str] | None = None
        self._server_log_path: Path | None = None
        self._server_log_handle: Any | None = None
        self._stopped = False

    def add_hook(self, hook: DeploymentHook):
        self._hooks.add_hook(hook)

    @classmethod
    def from_config(cls, config: LocalDeploymentConfig, run_id: str | None = None) -> Self:
        if not run_id:
            run_id = str(uuid.uuid4())
        config_kwargs = config.model_dump()
        if "container_runtime" not in config.model_fields_set:
            config_kwargs["container_runtime"] = _default_container_runtime()
        return cls(run_id=run_id, **config_kwargs)

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        if self._runtime is None:
            raise DeploymentNotStartedError("Runtime not started")
        return await self._runtime.is_alive(timeout=timeout)

    async def _wait_until_alive(self, timeout: float) -> IsAliveResponse:
        try:
            return await _wait_until_alive(self.is_alive, timeout=timeout, function_timeout=0.5)
        except TimeoutError as e:
            self.logger.error("Local runtime did not start within timeout.")
            await self.stop()
            raise e

    def _get_token(self) -> str:
        return str(uuid.uuid4())

    def _runtime_exec(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        self.logger.debug(f"Running container runtime command: {_shell_join(args)}")
        try:
            return subprocess.run(args, check=check, text=True, capture_output=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Container runtime {self._config.container_runtime!r} was not found in PATH") from exc

    def _get_current_container_network(self) -> str | None:
        if not _is_running_in_container():
            return None

        container_id = os.getenv("HOSTNAME")
        if not container_id:
            return None

        try:
            result = self._runtime_exec(
                [
                    self._config.container_runtime,
                    "inspect",
                    container_id,
                    "--format",
                    "{{range $name, $_ := .NetworkSettings.Networks}}{{println $name}}{{end}}",
                ]
            )
        except subprocess.CalledProcessError as exc:
            self.logger.warning(
                f"Failed to inspect current container network: {exc.stderr.strip() or exc.stdout.strip()}"
            )
            return None

        networks = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not networks:
            return None
        return networks[0]

    def _get_container_ip(self, container_name: str) -> str | None:
        try:
            result = self._runtime_exec(
                [
                    self._config.container_runtime,
                    "inspect",
                    container_name,
                    "--format",
                    "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                ]
            )
        except subprocess.CalledProcessError as exc:
            self.logger.warning(f"Failed to inspect sandbox IP: {exc.stderr.strip() or exc.stdout.strip()}")
            return None

        ip_address = result.stdout.strip()
        return ip_address or None

    def _get_runtime_host(self, container_name: str) -> str:
        if self._config.host:
            return self._config.host

        if _is_running_in_container() or self._config.network:
            container_ip = self._get_container_ip(container_name)
            if container_ip:
                return f"http://{container_ip}"

        return "http://127.0.0.1"

    def _format_command(self, token: str, port: int) -> str:
        return self._config.command.format(token=token, port=port)

    def _build_run_command(self, container_name: str, published_port: int, command: str) -> list[str]:
        network = self._config.network or self._get_current_container_network()

        args = [
            self._config.container_runtime,
            "run",
            "--rm",
            "-d",
            "--name",
            container_name,
            "--entrypoint",
            self._config.shell,
        ]
        if network:
            args.extend(["--network", network])
        args.extend(["-p", f"{published_port}:{self._config.runtime_port}"])
        args.extend(self._config.extra_run_args)
        args.extend([self._config.image, "-lc", command])
        return args

    def _build_apptainer_command(self, command: str) -> list[str]:
        args = [
            self._config.container_runtime,
            "exec",
            "--cleanenv",
            "--compat",
        ]
        args.extend(self._config.extra_run_args)
        args.extend([_normalize_apptainer_image(self._config.image), self._config.shell, "-lc", command])
        return args

    def _get_container_logs(self, container_name: str) -> str:
        if _is_apptainer_runtime(self._config.container_runtime):
            return self._get_apptainer_logs()

        try:
            result = self._runtime_exec([self._config.container_runtime, "logs", container_name], check=False)
        except Exception as exc:
            return f"<failed to fetch logs: {exc}>"
        return (result.stdout or result.stderr).strip()

    def _get_apptainer_logs(self) -> str:
        if self._server_log_handle:
            try:
                self._server_log_handle.flush()
            except Exception:
                pass
        if not self._server_log_path:
            return ""
        try:
            return self._server_log_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as exc:
            return f"<failed to fetch logs: {exc}>"

    def _start_apptainer_process(self, args: list[str]) -> subprocess.Popen[str]:
        fd, log_path = tempfile.mkstemp(prefix=f"uni-agent-local-{_sanitize_name(self.run_id)}-", suffix=".log")
        os.close(fd)
        self._server_log_path = Path(log_path)
        self._server_log_handle = self._server_log_path.open("w", encoding="utf-8", errors="replace")
        self.logger.debug(f"Running Apptainer command: {_shell_join(args)}")
        return subprocess.Popen(
            args,
            stdout=self._server_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    async def _start_apptainer(self, token: str, published_port: int) -> None:
        command = self._format_command(token=token, port=published_port)
        self._server_process = self._start_apptainer_process(self._build_apptainer_command(command))
        runtime_config = LocalRuntimeConfig(
            auth_token=token,
            host=self._config.host or "http://127.0.0.1",
            port=published_port,
            timeout=self._config.timeout,
        )
        self._runtime = LocalRuntime.from_config(runtime_config, run_id=self.run_id)

    async def _start_oci_container(self, token: str, container_name: str, published_port: int) -> None:
        command = self._format_command(token=token, port=self._config.runtime_port)
        result = await asyncio.to_thread(
            self._runtime_exec, self._build_run_command(container_name, published_port, command)
        )
        self._container_id = result.stdout.strip()
        host = await asyncio.to_thread(self._get_runtime_host, container_name)
        runtime_config = LocalRuntimeConfig(
            auth_token=token,
            host=host,
            port=self._config.runtime_port,
            timeout=self._config.timeout,
        )
        self._runtime = LocalRuntime.from_config(runtime_config, run_id=self.run_id)

    async def start(self, max_retries: int = 5) -> None:
        token = self._get_token()
        published_port = self._config.published_port or _pick_free_port()
        container_name = self._config.container_name or f"uni-agent-{_sanitize_name(self.run_id)}"
        self._stopped = False
        last_error: Exception | None = None
        for attempt in range(max_retries):
            self._stopped = False
            self.logger.info(
                f"Starting local deployment with runtime={self._config.container_runtime}, image={self._config.image}."
            )
            self._hooks.on_custom_step("Creating local sandbox")
            self._container_name = container_name

            try:
                if _is_apptainer_runtime(self._config.container_runtime):
                    await self._start_apptainer(token=token, published_port=published_port)
                else:
                    await self._start_oci_container(
                        token=token,
                        container_name=container_name,
                        published_port=published_port,
                    )

                await self._wait_until_alive(timeout=self._config.startup_timeout)
                await self.runtime.create_session(
                    CreateBashSessionRequest(startup_source=["/root/.bashrc"], startup_timeout=60)
                )
                self._stopped = False
                return
            except Exception as exc:
                last_error = exc
                logs = self._get_container_logs(container_name)
                self.logger.error(f"Failed to start local sandbox: {exc}\nContainer logs:\n{logs}")
                await self.stop()
                if attempt < max_retries - 1:
                    sleep_time = min(30, 2**attempt)
                    self.logger.info(f"Retrying local deployment startup in {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time)

        raise RuntimeError(f"Failed to create local sandbox after {max_retries} retries") from last_error

    async def copy_to_container(self, src: Path, tgt: Path):
        await self.runtime.execute(Command(command=["mkdir", "-p", str(tgt.parent)]))
        await self.runtime.upload(UploadRequest(source_path=str(src), target_path=str(tgt)))

    @property
    def tool_install_dir(self) -> Path:
        """Directory inside the container where tool scripts are installed."""
        return Path("/usr/local/bin")

    async def stop(self):
        if self._stopped:
            return

        if self._runtime:
            try:
                await self._runtime.close()
            except Exception as exc:
                self.logger.error(f"Failed to close local runtime within timeout: {exc}")
            self._runtime = None

        if self._server_process:
            try:
                if self._server_process.poll() is None:
                    self._server_process.terminate()
                    await asyncio.to_thread(self._server_process.wait, 10)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
                await asyncio.to_thread(self._server_process.wait)
            except Exception as exc:
                self.logger.error(f"Failed to stop local Apptainer process: {exc}")
            finally:
                self._server_process = None

        if self._server_log_handle:
            try:
                self._server_log_handle.close()
            except Exception:
                pass
            finally:
                self._server_log_handle = None

        if self._server_log_path:
            try:
                self._server_log_path.unlink(missing_ok=True)
            except Exception:
                pass
            finally:
                self._server_log_path = None

        if self._container_name and not _is_apptainer_runtime(self._config.container_runtime):
            try:
                await asyncio.to_thread(
                    self._runtime_exec,
                    [self._config.container_runtime, "rm", "-f", self._container_name],
                    False,
                )
            except Exception as exc:
                self.logger.error(f"Failed to delete local sandbox {self._container_name}: {exc}")
            finally:
                self._container_name = None
                self._container_id = None
        else:
            self._container_name = None
            self._container_id = None

        self._stopped = True

    @property
    def runtime(self) -> LocalRuntime:
        if self._runtime is None:
            raise DeploymentNotStartedError()
        return self._runtime

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    def __del__(self):
        if hasattr(self, "_container_name") and self._container_name and not getattr(self, "_stopped", False):
            msg = "Ensuring local deployment is stopped because object is deleted"
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
        self._stopped = True
