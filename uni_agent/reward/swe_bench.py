import json
import re
import time
import uuid
from pathlib import Path

from swebench.harness.constants import (
    END_TEST_OUTPUT,
    FAIL_ONLY_REPOS,
    MAP_REPO_VERSION_TO_SPECS,
    START_TEST_OUTPUT,
    EvalType,
    ResolvedStatus,
)
from swebench.harness.grading import get_eval_tests_report, get_resolution_status
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.python import get_test_directives
from swebench.harness.utils import get_modified_files

from __future__ import annotations

from typing import TYPE_CHECKING

from uni_agent.async_logging import get_logger
from uni_agent.reward.base import AbstractRewardSpec

if TYPE_CHECKING:
    from uni_agent.interaction import AgentEnv
from uni_agent.reward.registry import register_reward_spec
from uni_agent.utils import auto_await


def make_eval_script(metadata: dict, workdir: str, env_name: str = "testbed") -> str:
    """Build a self-contained bash eval script for a SWE-bench instance.

    Returns the full script string (with shebang). Caller writes it to a file
    and executes it; output is parsed by :func:`parse_eval_output`.
    """
    repo, version = metadata["repo"], metadata["version"]
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    cmds = _make_eval_script_list(
        instance=metadata, specs=specs, env_name=env_name,
        repo_directory=workdir, base_commit=metadata.get("base_commit", ""),
        test_patch=metadata["test_patch"],
    )
    return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + cmds) + "\n"


def parse_eval_output(metadata: dict, eval_output: str) -> tuple[bool, dict]:
    """Parse raw eval output and return (solved, report).

    Pure function — no env/sandbox dependency. Reusable by any caller that
    ran the script from :func:`make_eval_script`.
    """
    repo = metadata["repo"]
    report = {"resolved": False, "found_eval_status": False, "test_status": None}

    if START_TEST_OUTPUT not in eval_output or END_TEST_OUTPUT not in eval_output:
        return False, report

    test_content = eval_output.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
    status_map = MAP_REPO_TO_PARSER[repo](test_content, None)
    report["found_eval_status"] = True

    eval_ref = {
        "instance_id": metadata["instance_id"],
        "FAIL_TO_PASS": json.loads(metadata.get("FAIL_TO_PASS", "[]")),
        "PASS_TO_PASS": json.loads(metadata.get("PASS_TO_PASS", "[]")),
    }
    eval_type = EvalType.FAIL_ONLY if repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL
    eval_tests_report = get_eval_tests_report(status_map, eval_ref, eval_type=eval_type)
    report["test_status"] = eval_tests_report
    report["resolved"] = get_resolution_status(eval_tests_report) == ResolvedStatus.FULL.value
    return report["resolved"], report


# fix: https://github.com/SWE-bench/SWE-bench/issues/518
def _make_eval_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch):
    """
    Same as swebench's make_eval_script_list_py, but when test_patch only adds new files,
    get_modified_files returns [] and swebench would run `git checkout base_commit` (no paths),
    which resets the whole repo (e.g. reverts tox.ini). We use no-op instead.
    """
    _HEREDOC_DELIMITER = "EOF_114329324912"
    base_commit = instance["base_commit"]
    test_files = get_modified_files(test_patch)
    if test_files:
        reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    else:
        reset_tests_command = "echo 'skip reset'"

    apply_test_patch_command = f"git apply -v - <<'{_HEREDOC_DELIMITER}'\n{test_patch}\n{_HEREDOC_DELIMITER}"
    test_cmd = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]["test_cmd"]
    test_command = " ".join([test_cmd, *get_test_directives(instance)])

    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",
        f"cd {repo_directory}",
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
        reset_tests_command,
    ]
    return eval_commands


@register_reward_spec("swe_bench")
class SWEBenchRewardSpec(AbstractRewardSpec):
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

        try:
            eval_script = make_eval_script(self.metadata, workdir="/testbed")
            eval_script_container = Path(f"/tmp/eval_script_{uuid.uuid4()}.sh")
            await self.env.write_file(eval_script_container, eval_script)

            execution_t0 = time.perf_counter()
            output = await self.env.communicate(
                f"bash {eval_script_container}", timeout=self.eval_timeout, check="ignore")
            execution_time = time.perf_counter() - execution_t0
            result["eval_completed"] = True
            result["eval_execution_time"] = execution_time

            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
            solved, eval_report = parse_eval_output(self.metadata, output)
            result["eval_report"] = eval_report
            self.logger.info(f"Eval report: {eval_report}")
            result["resolved"] = solved
        except Exception as e:
            self.logger.error(f"Failed to evaluate: {e}")
        return result["resolved"], result

    @auto_await
    async def _get_interaction_env_patch(self) -> str:
        """Get the current staged diff in /testbed (interaction env state) as a patch string."""
        try:
            env_patch_file = Path(f"/tmp/patch_{uuid.uuid4()}.diff")
            await self.env.communicate(
                f"cd /testbed && git add -A && git diff --no-color --cached > {env_patch_file.as_posix()}",
                check="ignore",
            )
            patch_content = await self.env.read_file(env_patch_file)
            return patch_content
        except Exception as e:
            self.logger.error(f"Failed to get interaction environment patch: {e}")
            return ""

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

    def _get_eval_report(self, eval_output: str):
        """Delegates to module-level :func:`parse_eval_output`."""
        _, report = parse_eval_output(self.metadata, eval_output)
        return report
