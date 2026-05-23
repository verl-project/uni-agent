import re
import shlex
from pathlib import Path, PurePath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from swerex.exceptions import BashIncorrectSyntaxError, CommandTimeoutError
from swerex.runtime.abstract import (
    BashAction,
    BashInterruptAction,
    Command,
    ReadFileRequest,
    UploadRequest,
    WriteFileRequest,
)

from uni_agent.async_logging import get_logger
from uni_agent.deployment import DeployConfig
from uni_agent.skills.manager import SkillsManager
from uni_agent.tools.base import AbstractTool
from uni_agent.utils import auto_await


class ActionTimeoutError(Exception):
    pass


class ActionIncorrectSyntaxError(Exception):
    pass


class TerminalNotAliveError(Exception):
    pass


class AgentEnvConfig(BaseModel):
    deployment: DeployConfig = Field(description="Deployment configuration")
    env_variables: dict[str, str] | None = Field(
        default=None, description="Optional environment variables to set after start"
    )
    post_setup_cmd: str | None = Field(default=None, description="Command to run after environment startup")
    tool_install_dir: Path = Field(
        default=Path("/usr/local/bin"), description="Directory where tool scripts are installed"
    )
    model_config = ConfigDict(extra="forbid")


class AgentEnv:
    def __init__(
        self,
        run_id: str,
        env_config: AgentEnvConfig,
    ):
        """
        This class represents the environment in which we solve the tasks.

        Args:
            run_id: Run ID for the environment
            env_config: environment configuration
        """
        super().__init__()
        self.deployment = env_config.deployment.get_deployment(run_id)
        self.env_variables = env_config.env_variables
        self.post_setup_cmd = env_config.post_setup_cmd
        self.tool_install_dir = env_config.tool_install_dir
        self.logger = get_logger("environment", run_id)

    @auto_await
    async def start(self, max_retries: int = 5) -> None:
        """Start the environment"""

        self.logger.info("Beginning environment startup...")

        await self.deployment.start(max_retries=max_retries)
        self.logger.info("Runtime initialized")
        if self.env_variables:
            await self.set_env_variables(self.env_variables)
        if self.post_setup_cmd:
            await self.communicate(self.post_setup_cmd, check="raise")

    @auto_await
    async def install_tools(self, tools: list[AbstractTool]) -> None:
        install_dir = self.tool_install_dir
        await self.communicate(f"export PATH={shlex.quote(install_dir.as_posix())}:$PATH", check="raise")
        for tool in tools:
            tool_name = tool.name
            if tool.copy_to_remote:
                local_tool_path = tool.local_path
                assert local_tool_path is not None and local_tool_path.is_file(), (
                    f"Tool {tool_name} has copy_to_remote=True but local_path={local_tool_path!r} is not a file"
                )
                container_tool_path = install_dir / tool_name
                await self.copy_to_container(
                    src=local_tool_path,
                    tgt=container_tool_path,
                )
                await self.communicate(f"chmod +x {container_tool_path.as_posix()}", check="raise")
            install_cmd = tool.get_install_command()
            if install_cmd:
                await self.communicate(install_cmd, check="raise")
            # check if tool is installed
            await self.communicate(f"which {tool_name}", check="raise", error_msg=f"Failed to install tool {tool_name}")
            self.logger.info(f"Tool {tool_name} successfully installed")

    @auto_await
    async def copy_to_container(self, src: Path, tgt: Path) -> None:
        await self.deployment.runtime.execute(Command(command=["mkdir", "-p", str(tgt.parent)]))
        await self.deployment.runtime.upload(UploadRequest(source_path=str(src), target_path=str(tgt)))

    @auto_await
    async def install_skills(self, skills_manager: "SkillsManager") -> None:
        """Resolve each skill's runtime path and (if needed) copy it in.

        Mutates ``skills_manager.runtime_paths`` so the subsequent
        ``build_manifest`` call renders the right ``<location>`` for each
        skill:

        - **Host-style runtime** (``HostDeployment`` / ``LocalNativeDeployment``):
          skills are read in place from their host ``source_dir``; no copy.
        - **Container runtime** (everything else): each skill directory is
          uploaded to ``/opt/uni-agent/skills/<name>``.
        """
        from uni_agent.deployment.host.deployment import HostDeployment

        host_types: tuple[type, ...] = (HostDeployment,)
        try:
            from uni_agent.deployment.local_native.deployment import LocalNativeDeployment

            host_types = host_types + (LocalNativeDeployment,)
        except ImportError:
            pass

        if isinstance(self.deployment, host_types):
            for skill in skills_manager.skills:
                skills_manager.runtime_paths[skill.name] = skill.source_dir
            names = "\n".join(s.name for s in skills_manager.skills)
            self.logger.info(f"Host runtime: {len(skills_manager.skills)} skill(s) read in place, no copy\n{names}")
            return

        for skill in skills_manager.skills:
            tgt = Path("/opt/uni-agent/skills") / skill.name
            await self.copy_to_container(src=skill.source_dir, tgt=tgt)
            skills_manager.runtime_paths[skill.name] = tgt
            self.logger.info(f"Skill {skill.name} installed at {tgt}")
        self.logger.info(f"Installed {len(skills_manager.skills)} skill(s) into runtime")

    @auto_await
    async def close(self) -> None:
        """Shutdown SWE-ReX deployment etc."""
        self.logger.info("Beginning environment shutdown...")
        try:
            await self.deployment.stop()
        except Exception as e:
            self.logger.error(f"Failed to stop environment deployment: {e}")
            return
        self.logger.info("Environment shutdown completed")

    @auto_await
    async def run_action(self, action_cmd: str, action_timeout: int, max_observation_length: int = 100_000) -> str:
        try:
            observation = await self.communicate(input=action_cmd, timeout=action_timeout, check="ignore")
            observation = re.sub(r"\x1b\[[0-9;]*m|\r", "", observation)
            if observation.strip() == "":
                observation = "Your command ran successfully and did not produce any output."
            elif len(observation) > max_observation_length:
                observation = (
                    f"Observation:\n{observation[:max_observation_length]}<response clipped>\n"
                    f"<NOTE>Observations should not exceeded {max_observation_length} characters. "
                    f"{max_observation_length - len(observation)} characters were elided. "
                    "Please try a different command that produces less output or "
                    "use head/tail/grep/redirect the output to a file. Do not use interactive pagers.</NOTE>"
                )
            else:
                observation = f"Observation:\n{observation}"
            return observation
        except CommandTimeoutError:
            # interrupt timeout action
            # if terminal is still alive after interrupt, raise error
            try:
                await self.interrupt_session()
            except Exception:
                self.logger.error("Failed to interrupt session after command timeout")
                # check current terminal is still alive
                terminal_alive = False
                for _ in range(5):
                    probe_output = await self.communicate("echo 'terminal still alive'", check="ignore")
                    # Use substring match on stripped lines so residual marker
                    # noise from a recovering session does not fail the probe.
                    if isinstance(probe_output, str) and any(
                        line.strip() == "terminal still alive" for line in probe_output.splitlines()
                    ):
                        terminal_alive = True
                        break
                if not terminal_alive:
                    error_message = "Terminal did not respond to health checks"
                    self.logger.critical(error_message)
                    raise TerminalNotAliveError(error_message) from None

            # if terminal is still alive, return timeout observation
            observation = (
                f"The command '{action_cmd}' was cancelled because it took more than {action_timeout} seconds. "
                "Please try a different command that completes more quickly. Note: A common source of this error is "
                "if the command is interactive or requires user input (it is impossible to receive user input "
                "in the current environment, so the command will never complete)."
            )
            raise ActionTimeoutError(observation) from None

        except BashIncorrectSyntaxError as e:
            # this should not happen, so add critical logs here
            self.logger.error("Action command has incorrect syntax")
            error_message = (
                "Your bash command contained syntax errors and was NOT executed. "
                "Please fix the syntax errors and try again. This can be the result "
                "of not adhering to the syntax for multi-line commands. Here is the output of `bash -n`:\n"
                f"{e.extra_info['bash_stdout']}\n{e.extra_info['bash_stderr']}"
            )
            raise ActionIncorrectSyntaxError(error_message) from None

    @auto_await
    async def interrupt_session(self):
        self.logger.info("Interrupting session")
        await self.deployment.runtime.run_in_session(BashInterruptAction(timeout=10))

    @auto_await
    async def communicate(
        self,
        input: str,
        timeout: int | float = 60,
        check: Literal["warn", "ignore", "raise"] = "ignore",
        error_msg: str = "Command failed",
    ) -> str:
        """Executes a command in the running shell. The details of this are handled by
        the SWE-ReX deployment/runtime.

        Args:
            input: input to send to container
            timeout_duration: duration to wait for output
            check: `ignore`: do not extract exit code (more stable), `warn`: extract exit code and log error if
                exit code is non-zero, `raise`: raise error if exit code is non-zero
            error_msg: error message to raise if the command fails

        Returns:
            output: output from container
        """
        self.logger.debug(f"Input:\n{input}")
        rex_check = "silent" if check else "ignore"
        r = await self.deployment.runtime.run_in_session(BashAction(command=input, timeout=timeout, check=rex_check))
        output = r.output
        self.logger.debug(f"Output:\n{output}")
        if check != "ignore" and r.exit_code != 0:
            self.logger.error(f"{error_msg}:\n{output}")
            msg = f"Command {input!r} failed ({r.exit_code=}): {error_msg}"
            if check == "raise":
                await self.close()
                raise RuntimeError(msg)
        return output

    @auto_await
    async def read_file(self, path: str | PurePath, encoding: str | None = None, errors: str | None = None) -> str:
        """Read file contents from container

        Args:
            path: Absolute path to file
            encoding: Encoding to use when reading the file. None means default encoding.
                This is the same as the `encoding` argument of `Path.read_text()`
            errors: Error handling to use when reading the file. None means default error handling.
                This is the same as the `errors` argument of `Path.read_text()`

        Returns:
            file_contents: Contents of file as string
        """
        r = await self.deployment.runtime.read_file(ReadFileRequest(path=str(path), encoding=encoding, errors=errors))
        return r.content

    @auto_await
    async def write_file(self, path: str | PurePath, content: str) -> None:
        """Write content to file in container"""
        await self.deployment.runtime.write_file(WriteFileRequest(path=str(path), content=content))

    @auto_await
    async def set_env_variables(self, env_variables: dict[str, str]) -> None:
        """Set environment variables in the environment."""
        _env_setters = [f"export {k}={shlex.quote(str(v))}" for k, v in env_variables.items()]
        command = " && ".join(_env_setters)
        await self.communicate(command, check="raise")
