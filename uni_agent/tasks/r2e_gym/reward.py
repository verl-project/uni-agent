"""R2E-Gym execution-based reward."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

EVAL_ASSETS_PATH = "/tmp/r2e_eval_assets.tar.gz"

# Puts the held-out runner and tests back in the layout ``run_tests.sh`` expects (``/root/r2e_tests``, linked from
# ``/testbed/r2e_tests``), replacing anything the agent left at those paths.
_RESTORE_SCRIPT = f"""
set -euo pipefail
stage="$(mktemp -d /tmp/r2e_eval_restore.XXXXXX)"
tar -xzf {EVAL_ASSETS_PATH} -C "$stage"
rm -rf /root/run_tests.sh /root/r2e_tests /testbed/r2e_tests
mv "$stage/run_tests.sh" /root/run_tests.sh
chmod +x /root/run_tests.sh
mv "$stage/r2e_tests" /root/r2e_tests
ln -s /root/r2e_tests /testbed/r2e_tests
rm -rf "$stage" {EVAL_ASSETS_PATH}
""".strip()

_RUN_TESTS = "export PATH=/testbed/.venv/bin:/root/.venv/bin:/root/.local/bin:$PATH\nbash /root/run_tests.sh 2>&1"

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m|\r")


def parse_log_pytest(log: str | None) -> dict[str, str]:
    """Parse the pytest short summary emitted by R2E-Gym's ``run_tests.sh``.

    This mirrors the small parser in the official R2E-Gym package, keeping
    reward workers independent of that package after preprocessing.
    """
    if not log or "short test summary info" not in log:
        return {}
    summary = log.split("short test summary info", 1)[1]
    status_map: dict[str, str] = {}
    for raw_line in summary.strip().splitlines():
        line = raw_line.strip()
        status = next((candidate for candidate in ("PASSED", "FAILED", "ERROR") if candidate in line), None)
        if status is None:
            continue
        try:
            test_name = ".".join(line.split("::")[1:])
        except IndexError:  # pragma: no cover - split always returns a list
            test_name = line
        if status in {"FAILED", "ERROR"}:
            test_name = test_name.split(" - ", 1)[0]
        if test_name:
            status_map[test_name] = status
    return status_map


def _status_map(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("expected_output_json must encode an object")
    return {
        _ANSI_ESCAPE.sub("", str(key)).split(" - ", 1)[0]: status
        for key, status in sorted(value.items(), key=lambda item: str(item[0]))
    }


def _get_eval_report(metadata: Mapping[str, Any], eval_output: str) -> dict[str, Any]:
    parsed_status = _status_map(parse_log_pytest(_ANSI_ESCAPE.sub("", eval_output)))
    expected_status = _status_map(metadata["expected_output_json"])
    found = bool(parsed_status)
    return {
        "resolved": found and parsed_status == expected_status,
        "found_eval_status": found,
        "test_status": {
            "parsed_status": parsed_status,
            "expected_status": expected_status,
        },
    }


async def compute_reward(
    metadata, sandbox, eval_timeout: float = 300.0, eval_assets: bytes | None = None
) -> dict[str, Any]:
    """Run the image-provided R2E tests and compare them with the expected map.

    ``eval_assets`` is the archive of held-out tests that ``R2EGymTask`` took out of the sandbox before the agent
    ran. It is written back and restored first; a restore failure is an infrastructure error and raises instead
    of scoring the sample as unresolved.
    """
    instance_id = metadata.get("instance_id", "?")
    logger.info(f"running R2E-Gym eval for {instance_id} (timeout={eval_timeout:.0f}s)")
    if eval_assets is not None:
        await sandbox.write_file(EVAL_ASSETS_PATH, eval_assets)
        restored = await sandbox.exec_shell(_RESTORE_SCRIPT, workdir="/testbed", timeout=eval_timeout)
        if restored.exit_code != 0:
            detail = restored.stderr.strip() or restored.stdout.strip() or f"exit code {restored.exit_code}"
            raise RuntimeError(f"failed to restore held-out R2E-Gym tests for {instance_id}: {detail}")
    execution_t0 = time.perf_counter()
    response = await sandbox.exec_shell(
        _RUN_TESTS,
        workdir="/testbed",
        timeout=eval_timeout,
    )
    execution_time = time.perf_counter() - execution_t0
    eval_report = _get_eval_report(metadata, response.stdout)
    result = {
        "eval_completed": response.exit_code == 0,
        "eval_exit_code": response.exit_code,
        "eval_execution_time": execution_time,
        "eval_report": eval_report,
        "resolved": eval_report["resolved"],
    }
    if not eval_report["found_eval_status"]:
        logger.warning(f"no parseable R2E-Gym test output for {instance_id}; marking unresolved")
    logger.info(f"R2E-Gym reward for {instance_id}: resolved={result['resolved']}")
    return result
