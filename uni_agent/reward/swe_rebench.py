import re
import time
import uuid
from pathlib import Path

from swebench.harness.constants import (
    END_TEST_OUTPUT,
    FAIL_ONLY_REPOS,
    START_TEST_OUTPUT,
    EvalType,
    ResolvedStatus,
    TestStatus,
)
from swebench.harness.grading import get_eval_tests_report, get_resolution_status
from swebench.harness.test_spec.python import get_test_directives
from swebench.harness.utils import get_modified_files

from uni_agent.async_logging import get_logger
from uni_agent.interaction import AgentEnv
from uni_agent.reward.base import AbstractRewardSpec
from uni_agent.reward.registry import register_reward_spec
from uni_agent.utils import auto_await


def _make_eval_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch) -> list:
    """
    Applies the test patch and runs the tests.
    """
    HEREDOC_DELIMITER = "EOF_114329324912"
    test_files = get_modified_files(test_patch)
    # Reset test files to the state they should be in before the patch.
    if test_files:
        reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    else:
        reset_tests_command = "echo 'skip reset'"
    apply_test_patch_command = f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"

    test_cmd = specs["test_cmd"]
    if isinstance(test_cmd, list):
        test_cmd = " ".join(test_cmd)
    test_command = " ".join([test_cmd, *get_test_directives(instance)])
    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",  # for nonroot user
        f"cd {repo_directory}",
        # This is just informational, so we have a record
        "git status",
        "git show",
        f"git -c core.fileMode=false diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])
    eval_commands += [
        reset_tests_command,
        apply_test_patch_command,
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
        reset_tests_command,  # Revert tests after done, leave the repo in the same state as before
    ]
    return eval_commands


def parse_log_pytest(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


def parse_log_pytest_v2(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework (Later Version)

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    escapes = "".join([chr(char) for char in range(1, 32)])
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line)
        translator = str.maketrans("", "", escapes)
        line = line.translate(translator)
        if any([line.startswith(x.value) for x in TestStatus]):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) >= 2:
                test_status_map[test_case[1]] = test_case[0]
        # Support older pytest versions by checking if the line ends with the test status
        elif any([line.endswith(x.value) for x in TestStatus]):
            test_case = line.split()
            if len(test_case) >= 2:
                test_status_map[test_case[0]] = test_case[1]
    return test_status_map


@register_reward_spec("swe_rebench")
class SWEREBenchRewardSpec(AbstractRewardSpec):
    def __init__(self, *, run_id: str, metadata: dict, env: AgentEnv, eval_timeout: int = 300):
        self.run_id = run_id
        self.metadata = metadata
        self.env = env
        self.logger = get_logger("reward_spec", run_id=run_id)
        self.eval_timeout = eval_timeout

    @auto_await
    async def apply_gold_patch(self) -> str:
        gold_patch = self.metadata["patch"]
        await self._apply_patch(gold_patch)

    @auto_await
    async def compute_reward(self, **kwargs) -> tuple[dict | None, bool]:
        """Run eval script in container via env.communicate (no execute). Returns (eval_report, success)."""
        result = {
            "eval_completed": False,
            "eval_execution_time": None,
            "eval_report": None,
            "resolved": False,
        }

        # 1. eval script
        instance = self.metadata
        specs = {
            "test_cmd": instance["test_cmd"],
            "eval_commands": instance.get("eval_commands", ""),
            "install": instance.get("install", ""),
        }
        env_name = "testbed"
        repo_directory = f"/{env_name}"
        base_commit = instance["base_commit"]
        test_patch = instance["test_patch"]
        eval_script_list = _make_eval_script_list(
            instance=instance,
            specs=specs,
            env_name=env_name,
            repo_directory=repo_directory,
            base_commit=base_commit,
            test_patch=test_patch,
        )
        eval_script = "\n".join(["#!/bin/bash", "set -uxo pipefail"] + eval_script_list) + "\n"
        try:
            # write eval script to container
            eval_script_container = Path(f"/tmp/eval_script_{uuid.uuid4()}.sh")
            await self.env.write_file(eval_script_container, eval_script)

            execution_t0 = time.perf_counter()

            cmd_str = f"bash {eval_script_container}"
            output = await self.env.communicate(cmd_str, timeout=self.eval_timeout, check="ignore")

            execution_time = time.perf_counter() - execution_t0
            result["eval_completed"] = True
            result["eval_execution_time"] = execution_time

            # Remove ANSI escape codes and \r
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)

            eval_report = self._get_eval_report(output)
            result["eval_report"] = eval_report
            self.logger.info(f"Eval report: {eval_report}")
            result["resolved"] = eval_report["resolved"]
        except Exception as e:
            self.logger.error(f"Failed to evaluate: {e}")
        return result["resolved"], result

    @auto_await
    async def _apply_patch(self, patch: str) -> None:
        """Apply a patch string to the env. Tries multiple apply strategies in order."""
        if not patch or not patch.strip():
            self.logger.info("Empty patch, nothing to apply.")
            return
        patch_path = Path(f"/tmp/patch_{uuid.uuid4()}.diff")
        await self.env.write_file(patch_path, patch)
        commands = [
            f"cd /testbed && git apply --whitespace=fix {patch_path.as_posix()}",
            f"cd /testbed && git apply --reject --whitespace=nowarn {patch_path.as_posix()}",
            f"cd /testbed && patch --batch --fuzz=5 -p1 -i {patch_path.as_posix()}",
        ]
        last_error: Exception | None = None
        for cmd in commands:
            try:
                await self.env.communicate(cmd, check="raise")
                self.logger.info("Applied patch successfully!")
                return
            except RuntimeError as e:
                last_error = e
                continue
        raise RuntimeError("Failed to apply patch with any command") from last_error

    def _get_logs_eval(self, eval_output: str):
        instance = self.metadata
        if instance["log_parser"] == "parse_log_pytest":
            log_parser_fn = parse_log_pytest
        elif instance["log_parser"] == "parse_log_pytest_v2":
            log_parser_fn = parse_log_pytest_v2
        else:
            raise NotImplementedError(f"Log parser {instance['log_parser']} is not implemented.")
        if START_TEST_OUTPUT in eval_output and END_TEST_OUTPUT in eval_output:
            test_content = eval_output.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
            status_map = log_parser_fn(test_content)
            return status_map, True
        else:
            status_map = {}
            return status_map, False

    def _get_eval_report(self, eval_output: str):
        eval_report = {
            "resolved": False,
            "found_eval_status": False,
            "test_status": None,
        }

        # step 1: get logs eval
        status_map, found = self._get_logs_eval(eval_output)
        eval_report["found_eval_status"] = found

        if not found:
            return eval_report

        # step 2: get eval tests report
        eval_ref = {
            "instance_id": self.metadata["instance_id"],
            "FAIL_TO_PASS": self.metadata.get("FAIL_TO_PASS", []),
            "PASS_TO_PASS": self.metadata.get("PASS_TO_PASS", []),
        }
        repo = self.metadata["repo"]
        eval_type = EvalType.FAIL_ONLY if repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL
        report = get_eval_tests_report(status_map, eval_ref, eval_type=eval_type)
        eval_report["test_status"] = report
        if get_resolution_status(report) == ResolvedStatus.FULL.value:
            eval_report["resolved"] = True
        return eval_report
