from __future__ import annotations

import json
import logging
import time
import uuid

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

logger = logging.getLogger(__name__)


def _make_eval_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch):
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


def _get_logs_eval(metadata, eval_output: str):
    instance = metadata
    repo = instance["repo"]
    log_parser = MAP_REPO_TO_PARSER[repo]
    if START_TEST_OUTPUT in eval_output and END_TEST_OUTPUT in eval_output:
        test_content = eval_output.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
        status_map = log_parser(test_content, None)
        return status_map, True
    else:
        status_map = {}
        return status_map, False


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
        "FAIL_TO_PASS": json.loads(metadata.get("FAIL_TO_PASS", "[]")),
        "PASS_TO_PASS": json.loads(metadata.get("PASS_TO_PASS", "[]")),
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

    # 1. eval script
    instance = metadata
    instance_id = instance.get("instance_id", "?")
    repo = instance["repo"]
    version = instance.get("version")
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
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

    # write eval script to container
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

    eval_report = _get_eval_report(metadata, output)
    result["eval_report"] = eval_report
    result["resolved"] = eval_report["resolved"]
    if not eval_report["found_eval_status"]:
        logger.warning(f"no parseable test output for {instance_id}; marking unresolved")
    logger.info(f"reward for {instance_id}: resolved={result['resolved']}")

    return result
