"""SWE-rebench eval/reward."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid

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

logger = logging.getLogger(__name__)


def _make_eval_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch):
    """Apply the test patch and run the tests, using the row's own test_cmd/install."""
    _HEREDOC_DELIMITER = "EOF_114329324912"
    test_files = get_modified_files(test_patch)
    if test_files:
        reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    else:
        reset_tests_command = "echo 'skip reset'"
    apply_test_patch_command = f"git apply -v - <<'{_HEREDOC_DELIMITER}'\n{test_patch}\n{_HEREDOC_DELIMITER}"

    test_cmd = specs["test_cmd"]
    if isinstance(test_cmd, list):
        test_cmd = " ".join(test_cmd)
    test_command = " ".join([test_cmd, *get_test_directives(instance)])

    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if specs.get("eval_commands"):
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
    if specs.get("install"):
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


def parse_log_pytest(log: str) -> dict[str, str]:
    """Parse test logs from the PyTest framework into a test-case -> status map."""
    test_status_map = {}
    for line in log.split("\n"):
        if any(line.startswith(x.value) for x in TestStatus):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


def parse_log_pytest_v2(log: str) -> dict[str, str]:
    """Parse PyTest logs (later versions), stripping control codes first."""
    test_status_map = {}
    escapes = "".join([chr(char) for char in range(1, 32)])
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line)
        translator = str.maketrans("", "", escapes)
        line = line.translate(translator)
        if any(line.startswith(x.value) for x in TestStatus):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) >= 2:
                test_status_map[test_case[1]] = test_case[0]
        # Older pytest versions print the status at the end of the line.
        elif any(line.endswith(x.value) for x in TestStatus):
            test_case = line.split()
            if len(test_case) >= 2:
                test_status_map[test_case[0]] = test_case[1]
    return test_status_map


_LOG_PARSERS = {
    "parse_log_pytest": parse_log_pytest,
    "parse_log_pytest_v2": parse_log_pytest_v2,
}


def _as_list(value) -> list:
    """FAIL_TO_PASS / PASS_TO_PASS may arrive as a list or a JSON-encoded string."""
    if value is None:
        return []
    if isinstance(value, str):
        return json.loads(value) if value.strip() else []
    return list(value)


def _get_logs_eval(metadata, eval_output: str):
    parser_name = metadata["log_parser"]
    log_parser = _LOG_PARSERS.get(parser_name)
    if log_parser is None:
        raise NotImplementedError(f"log parser {parser_name!r} is not implemented for swe_rebench")
    if START_TEST_OUTPUT in eval_output and END_TEST_OUTPUT in eval_output:
        test_content = eval_output.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
        return log_parser(test_content), True
    return {}, False


def _get_eval_report(metadata, eval_output: str):
    eval_report = {
        "resolved": False,
        "found_eval_status": False,
        "test_status": None,
    }

    # step 1: get logs eval
    status_map, found = _get_logs_eval(metadata, eval_output)
    eval_report["found_eval_status"] = found
    if not found:
        return eval_report

    # step 2: get eval tests report
    eval_ref = {
        "instance_id": metadata["instance_id"],
        "FAIL_TO_PASS": _as_list(metadata.get("FAIL_TO_PASS")),
        "PASS_TO_PASS": _as_list(metadata.get("PASS_TO_PASS")),
    }
    repo = metadata["repo"]
    eval_type = EvalType.FAIL_ONLY if repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL
    report = get_eval_tests_report(status_map, eval_ref, eval_type=eval_type)
    eval_report["test_status"] = report
    if get_resolution_status(report) == ResolvedStatus.FULL.value:
        eval_report["resolved"] = True
    return eval_report


async def compute_reward(metadata, sandbox, eval_timeout: float = 300.0) -> dict:
    result = {
        "eval_completed": False,
        "eval_execution_time": None,
        "eval_report": None,
        "resolved": False,
    }

    # 1. eval script (test_cmd/install/eval_commands come from the row, not swebench specs)
    instance = metadata
    instance_id = instance.get("instance_id", "?")
    repo = instance["repo"]
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

    eval_script_container = f"/tmp/eval_script_{uuid.uuid4()}.sh"
    await sandbox.write_file(eval_script_container, eval_script)

    logger.info(f"running eval for {instance_id} (repo={repo}, timeout={eval_timeout:.0f}s)")
    execution_t0 = time.perf_counter()

    resp = await sandbox.exec_shell(f"bash {eval_script_container} 2>&1", workdir="/testbed", timeout=eval_timeout)
    output, exit_code = resp.stdout, resp.exit_code
    execution_time = time.perf_counter() - execution_t0
    result["eval_completed"] = exit_code == 0
    result["eval_execution_time"] = execution_time
    logger.info(f"eval finished in {execution_time:.1f}s (exit_code={exit_code})")

    # Drop ANSI colors / CRs so the pytest parsers see clean lines.
    output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)

    eval_report = _get_eval_report(metadata, output)
    result["eval_report"] = eval_report
    result["resolved"] = eval_report["resolved"]
    if not eval_report["found_eval_status"]:
        logger.warning(f"no parseable test output for {instance_id}; marking unresolved")
    logger.info(f"reward for {instance_id}: resolved={result['resolved']}")

    return result
