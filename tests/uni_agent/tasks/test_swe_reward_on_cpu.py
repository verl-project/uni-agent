import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from uni_agent.sandbox.base import ExecResult
from uni_agent.tasks.swe_bench import reward

UNRESOLVED_OUTPUT = "\n".join(
    (
        reward.START_TEST_OUTPUT,
        "FAILED test_requests.py::test_fix - AssertionError",
        "PASSED test_requests.py::test_existing",
        reward.END_TEST_OUTPUT,
    )
)
RESOLVED_OUTPUT = UNRESOLVED_OUTPUT.replace(
    "FAILED test_requests.py::test_fix - AssertionError", "PASSED test_requests.py::test_fix"
)
EMPTY_TEST_OUTPUT = f"{reward.START_TEST_OUTPUT}\n{reward.END_TEST_OUTPUT}"
UNRESOLVED_STATUS_MAP = {"test_requests.py::test_fix": "FAILED", "test_requests.py::test_existing": "PASSED"}
RESOLVED_STATUS_MAP = {**UNRESOLVED_STATUS_MAP, "test_requests.py::test_fix": "PASSED"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exit_code,output,found,status_map,resolved",
    [
        pytest.param(0, UNRESOLVED_OUTPUT, True, UNRESOLVED_STATUS_MAP, False, id="normal-unresolved"),
        pytest.param(1, UNRESOLVED_OUTPUT, True, UNRESOLVED_STATUS_MAP, False, id="nonzero-unresolved"),
        pytest.param(1, RESOLVED_OUTPUT, True, RESOLVED_STATUS_MAP, True, id="nonzero-resolved"),
        pytest.param(0, EMPTY_TEST_OUTPUT, True, {}, False, id="markers-without-tests"),
        pytest.param(1, "collection failed", False, {}, False, id="missing-markers"),
        pytest.param(-1, "", False, {}, False, id="sandbox-timeout-result"),
        pytest.param(127, "", False, {}, False, id="sandbox-exec-failure-result"),
    ],
)
async def test_swe_reward_preserves_execution_evidence_without_changing_resolution(
    exit_code, output, found, status_map, resolved
):
    metadata = {
        "instance_id": "psf__requests-test",
        "repo": "psf/requests",
        "version": "2.2",
        "base_commit": "0" * 40,
        "test_patch": "",
        "FAIL_TO_PASS": json.dumps(["test_requests.py::test_fix"]),
        "PASS_TO_PASS": json.dumps(["test_requests.py::test_existing"]),
    }
    sandbox = SimpleNamespace(
        write_file=AsyncMock(),
        exec_shell=AsyncMock(
            return_value=ExecResult(
                exit_code=exit_code,
                stdout=output,
                stderr="exec timed out after 600.0s" if exit_code == -1 else "",
            )
        ),
    )

    result = await reward.compute_reward(metadata, sandbox, eval_timeout=600.0)

    assert result["eval_exit_code"] == exit_code
    assert result["eval_completed"] is (exit_code == 0)
    assert result["eval_report"]["found_eval_status"] is found
    assert result["eval_report"]["status_map"] == status_map
    assert result["eval_report"]["resolved"] is resolved
    assert result["resolved"] is resolved
    if not found:
        assert result["eval_report"]["test_status"] is None
    assert sandbox.exec_shell.await_args.kwargs == {"workdir": "/testbed", "timeout": 600.0}
