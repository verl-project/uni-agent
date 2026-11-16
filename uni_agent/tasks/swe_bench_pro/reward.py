"""SWE-bench Pro patch evaluation."""

from __future__ import annotations

import json
import logging
import shlex
import time
import uuid
from typing import Any

from ...sandbox import SandboxBackend

logger = logging.getLogger(__name__)

REPO_WORKDIR = "/app"


def _grade(metadata: dict[str, Any], output: Any) -> dict[str, Any]:
    if not isinstance(output, dict) or not isinstance(output.get("tests"), list):
        raise ValueError("SWE-bench Pro parser output must contain a 'tests' list")

    fail_to_pass = set(metadata["fail_to_pass"])
    pass_to_pass = set(metadata["pass_to_pass"])
    tests = output["tests"]
    passed = {
        test.get("name")
        for test in tests
        if isinstance(test, dict) and test.get("status") == "PASSED" and isinstance(test.get("name"), str)
    }
    missing_fail_to_pass = sorted(fail_to_pass - passed)
    missing_pass_to_pass = sorted(pass_to_pass - passed)
    return {
        "resolved": not missing_fail_to_pass and not missing_pass_to_pass,
        "found_eval_status": True,
        "test_status": tests,
        "required_count": len(fail_to_pass | pass_to_pass),
        "passed_count": len((fail_to_pass | pass_to_pass) & passed),
        "missing_fail_to_pass": missing_fail_to_pass,
        "missing_pass_to_pass": missing_pass_to_pass,
    }


def _selected_test_args(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError("selected_test_files_to_run must be a string or a list of strings")


async def _read_text(sandbox: SandboxBackend, path: str) -> str:
    try:
        return (await sandbox.read_file(path)).decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


async def compute_reward(
    metadata: dict[str, Any],
    sandbox: SandboxBackend,
    eval_timeout: float = 3600.0,
) -> dict[str, Any]:
    """Evaluate the worktree with the official per-instance test scripts."""
    instance_id = metadata["instance_id"]
    selected_tests = _selected_test_args(metadata["selected_test_files_to_run"])

    workspace = f"/tmp/swe_bench_pro_eval_{uuid.uuid4().hex}"
    paths = {
        "test_patch": f"{workspace}/test.patch",
        "run_script": f"{workspace}/run_script.sh",
        "parser": f"{workspace}/parser.py",
        "stdout": f"{workspace}/stdout.log",
        "stderr": f"{workspace}/stderr.log",
        "output": f"{workspace}/output.json",
    }
    await sandbox.write_file(paths["test_patch"], metadata["test_patch"])
    await sandbox.write_file(paths["run_script"], metadata["run_script"])
    await sandbox.write_file(paths["parser"], metadata["parser"])

    run_tests = shlex.join(["bash", paths["run_script"], *selected_tests])
    eval_script = "\n".join(
        [
            f"cd {shlex.quote(REPO_WORKDIR)}",
            f"git apply -v {shlex.quote(paths['test_patch'])} || exit $?",
            f"{run_tests} > {shlex.quote(paths['stdout'])} 2> {shlex.quote(paths['stderr'])}",
            shlex.join(["python", paths["parser"], paths["stdout"], paths["stderr"], paths["output"]]),
        ]
    )

    started = time.perf_counter()
    response = await sandbox.exec_shell(
        eval_script,
        workdir=REPO_WORKDIR,
        timeout=eval_timeout,
    )
    execution_time = time.perf_counter() - started

    result: dict[str, Any] = {
        "eval_completed": False,
        "eval_execution_time": execution_time,
        "eval_report": {
            "resolved": False,
            "found_eval_status": False,
            "eval_exit_code": response.exit_code,
            "error": None,
        },
        "resolved": False,
    }
    try:
        output = json.loads((await sandbox.read_file(paths["output"])).decode("utf-8"))
        report = _grade(metadata, output)
    except Exception as exc:
        details = [
            response.stderr.strip(),
            response.stdout.strip(),
            await _read_text(sandbox, paths["stderr"]),
            await _read_text(sandbox, paths["stdout"]),
        ]
        detail = next((item for item in details if item), "")
        result["eval_report"]["error"] = detail or f"{type(exc).__name__}: {exc}"
        logger.warning("SWE-bench Pro evaluation failed for %s: %s", instance_id, result["eval_report"]["error"])
        return result

    result["eval_report"].update(report)
    result["eval_completed"] = True
    result["resolved"] = report["resolved"]
    logger.info(
        "SWE-bench Pro reward for %s: resolved=%s required=%s passed=%s",
        instance_id,
        result["resolved"],
        report["required_count"],
        report["passed_count"],
    )
    return result
