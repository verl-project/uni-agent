"""SWE-bench Multilingual evaluation and reward computation.

The benchmark spans C, Go, Java, JavaScript, PHP, Ruby, and Rust repositories.
Its published images contain the language-specific dependencies and may also
contain uncommitted build-time edits. Evaluation therefore resets only the test
files, preserving both those edits and the agent's solution.
"""

from __future__ import annotations

import logging
import time
import uuid

from swebench.harness.constants import (
    END_TEST_OUTPUT,
    FAIL_ONLY_REPOS,
    MAP_REPO_TO_EXT,
    MAP_REPO_VERSION_TO_SPECS,
    START_TEST_OUTPUT,
    EvalType,
    ResolvedStatus,
)
from swebench.harness.grading import get_eval_tests_report, get_resolution_status
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.javascript import get_download_img_commands
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import get_modified_files

logger = logging.getLogger(__name__)

# Heredoc delimiter used by the upstream harness to inline the test patch.
HEREDOC_DELIMITER = "EOF_114329324912"


def _as_commands(value) -> list[str]:
    """Normalize a harness command field to a list of shell commands."""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _make_eval_script_list(metadata: dict) -> list[str]:
    """Build the official multilingual test flow without resetting the solution."""
    repo = metadata["repo"]
    version = str(metadata["version"])
    repo_directory = "/testbed"
    base_commit = metadata["base_commit"]
    test_patch = metadata["test_patch"]
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]

    test_files = get_modified_files(test_patch)
    if test_files:
        reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    else:
        reset_tests_command = "echo 'skip reset'"

    apply_test_patch_command = (
        f"git apply --verbose --reject - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    )
    build_commands = _as_commands(specs.get("build"))
    test_commands = _as_commands(specs["test_cmd"])

    eval_commands = [
        "chmod 1777 /tmp 2>/dev/null || true",
        f"cd {repo_directory}",
        f"git config --global --add safe.directory {repo_directory}",
        f"cd {repo_directory}",
        reset_tests_command,
        apply_test_patch_command,
        *build_commands,
        f": '{START_TEST_OUTPUT}'",
        *test_commands,
        f": '{END_TEST_OUTPUT}'",
        reset_tests_command,
    ]

    # JavaScript instances may carry image fixtures downloaded immediately
    # before applying the test patch.
    if MAP_REPO_TO_EXT[repo] == "js":
        patch_index = eval_commands.index(apply_test_patch_command)
        eval_commands[patch_index:patch_index] = get_download_img_commands(metadata)

    return eval_commands


def _build_eval_script(metadata: dict) -> str:
    """Assemble the eval script without ``set -e`` so cleanup always runs."""
    return "\n".join(["#!/bin/bash", "set -uxo pipefail", *_make_eval_script_list(metadata)]) + "\n"


def _grade(test_spec, output: str) -> dict:
    """Parse test output and grade FAIL_TO_PASS/PASS_TO_PASS with swebench."""
    report = {
        "resolved": False,
        "found_eval_status": False,
        "test_status": None,
    }

    parser = MAP_REPO_TO_PARSER[test_spec.repo]
    if START_TEST_OUTPUT in output and END_TEST_OUTPUT in output:
        test_output = output.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]
    else:
        test_output = output

    status_map = parser(test_output, test_spec)
    # Some runners emit relevant output outside the markers, commonly on stderr.
    if not status_map and test_output != output:
        status_map = parser(output, test_spec)
    if not status_map:
        logger.warning(
            "SWE-bench Multilingual parser matched no tests for %s; output tail:\n%s",
            test_spec.instance_id,
            output[-3000:],
        )
        return report

    report["found_eval_status"] = True
    eval_ref = {
        "instance_id": test_spec.instance_id,
        "FAIL_TO_PASS": test_spec.FAIL_TO_PASS,
        "PASS_TO_PASS": test_spec.PASS_TO_PASS,
    }
    eval_type = EvalType.FAIL_ONLY if test_spec.repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL
    test_status = get_eval_tests_report(status_map, eval_ref, eval_type=eval_type)
    report["test_status"] = test_status
    report["resolved"] = get_resolution_status(test_status) == ResolvedStatus.FULL.value

    if not report["resolved"]:
        logger.warning(
            "SWE-bench Multilingual unresolved for %s: FAIL_TO_PASS failures=%s; PASS_TO_PASS failures=%s",
            test_spec.instance_id,
            test_status["FAIL_TO_PASS"]["failure"][:25],
            test_status["PASS_TO_PASS"]["failure"][:25],
        )

    return report


async def compute_reward(metadata: dict, sandbox, eval_timeout: float = 1800.0) -> dict:
    """Run the official multilingual evaluation flow inside ``sandbox``."""
    result = {
        "eval_completed": False,
        "eval_execution_time": None,
        "eval_report": None,
        "resolved": False,
    }

    test_spec = make_test_spec(metadata)
    eval_script_path = f"/tmp/sbm_eval_{uuid.uuid4().hex}.sh"
    await sandbox.write_file(eval_script_path, _build_eval_script(metadata))

    logger.info(
        "running SWE-bench Multilingual eval for %s (repo=%s, language=%s, timeout=%.0fs)",
        test_spec.instance_id,
        test_spec.repo,
        getattr(test_spec, "language", "?"),
        eval_timeout,
    )
    execution_t0 = time.perf_counter()

    # Piping through cat gives the language-specific runners a non-TTY output
    # stream, matching the official harness's docker-exec behavior.
    response = await sandbox.exec_shell(
        f"set -o pipefail; bash {eval_script_path} 2>&1 | cat",
        workdir="/testbed",
        timeout=eval_timeout,
    )
    execution_time = time.perf_counter() - execution_t0
    output = response.stdout
    if response.stderr:
        output = f"{output}\n{response.stderr}"

    result["eval_completed"] = response.exit_code == 0
    result["eval_execution_time"] = execution_time
    logger.info("multilingual eval finished in %.1fs (exit_code=%s)", execution_time, response.exit_code)

    eval_report = _grade(test_spec, output)
    result["eval_report"] = eval_report
    result["resolved"] = eval_report["resolved"]
    logger.info("reward for %s: resolved=%s", test_spec.instance_id, result["resolved"])

    return result
