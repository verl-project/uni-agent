from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pyarrow.parquet as pq
import pytest
from datasets import Dataset

from examples.inference.parallel_run_oracle import OracleEvalActor, _write_passed_dataset
from uni_agent.sandbox.base import ExecResult
from uni_agent.tasks.swe_bench import reward as swe_reward
from uni_agent.tasks.swe_bench.task import SWEBenchTask
from uni_agent.tasks.swe_rebench import reward as rebench_reward
from uni_agent.tasks.swe_rebench.task import SWEREBenchTask


@pytest.fixture(params=[(SWEBenchTask, swe_reward), (SWEREBenchTask, rebench_reward)], ids=["swe", "rebench"])
def oracle_task(request, monkeypatch):
    task_cls, reward_module = request.param
    sandbox = SimpleNamespace(
        write_file=AsyncMock(),
        exec=AsyncMock(),
        exec_shell=AsyncMock(return_value=ExecResult(exit_code=0, stdout="", stderr="")),
    )
    stopped = []

    @asynccontextmanager
    async def build_sandbox(self):
        try:
            yield sandbox
        finally:
            stopped.append(True)

    monkeypatch.setattr(task_cls, "build_sandbox", build_sandbox)
    agent = AsyncMock()
    monkeypatch.setattr(task_cls, "build_agent", agent)
    scorer = AsyncMock(return_value={"resolved": True, "eval_completed": True})
    monkeypatch.setattr(reward_module, "compute_reward", scorer)
    config = {
        "name": task_cls.name,
        "run_oracle_solution": True,
        "sandbox": {"provider": "local"},
        "metadata": {"instance_id": "gold-control", "base_commit": "a" * 40, "patch": "gold patch bytes"},
    }
    return config, sandbox, scorer, agent, stopped


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "apply_result",
    [
        ExecResult(exit_code=1, stdout="apply diagnostic", stderr="patch does not apply"),
        ExecResult(exit_code=2, stdout="stdout-only diagnostic", stderr=""),
        ExecResult(exit_code=-1, stdout="", stderr="exec timed out"),
        ConnectionError("sandbox transport unavailable"),
    ],
    ids=["rejected-patch", "stdout-only", "timeout-result", "transport-error"],
)
async def test_failed_gold_patch_cannot_reach_passing_verifier_or_export(oracle_task, apply_result, tmp_path):
    config, sandbox, scorer, agent, stopped = oracle_task
    if isinstance(apply_result, Exception):
        sandbox.exec.side_effect = apply_result
    else:
        sandbox.exec.return_value = apply_result

    # Use the real Task factory and Oracle actor. The scorer would report success
    # if called on the unpatched checkout, reproducing the false-admission risk.
    record = await OracleEvalActor(str(tmp_path / "logs"), 1).run_single(config)
    record["row_index"] = 0
    source = Dataset.from_list([{"extra_info": {"tools_kwargs": {"task": config}}}])
    output = tmp_path / "passed.parquet"
    _write_passed_dataset(source, [record], output)

    assert record["outcome"] == "execution_error"
    assert record["status"] == "error"
    assert record["solved"] is False
    assert pq.read_table(output).num_rows == 0
    log = (tmp_path / "logs" / record["log_id"] / "task.log").read_text()
    assert record["error"] in log
    scorer.assert_not_awaited()
    agent.assert_not_called()
    assert stopped == [True]
    if isinstance(apply_result, Exception):
        assert record["error"] == "ConnectionError: sandbox transport unavailable"
    else:
        assert "gold-control" in record["error"]
        assert f"exit_code={apply_result.exit_code}" in record["error"]
        assert apply_result.stdout in record["error"]
        assert apply_result.stderr in record["error"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("resolved", [True, False], ids=["passing-oracle", "failing-oracle"])
async def test_successful_gold_patch_preserves_verifier_result_and_export(oracle_task, resolved, tmp_path):
    config, sandbox, scorer, agent, stopped = oracle_task
    sandbox.exec.return_value = ExecResult(exit_code=0, stdout="applied", stderr="")
    scorer.return_value = {"resolved": resolved, "eval_completed": True}

    record = await OracleEvalActor(None, 1).run_single(config)
    record["row_index"] = 0
    source = Dataset.from_list([{"extra_info": {"tools_kwargs": {"task": config}}}])
    output = tmp_path / "passed.parquet"
    _write_passed_dataset(source, [record], output)

    assert record["outcome"] == ("passed" if resolved else "oracle_failed")
    assert record["status"] == "completed"
    assert record["solved"] is resolved
    assert record["reward"] == float(resolved)
    assert pq.read_table(output).to_pylist() == (source.to_list() if resolved else [])
    sandbox.write_file.assert_awaited_once_with("/tmp/gold_patch.patch", config["metadata"]["patch"])
    sandbox.exec.assert_awaited_once_with(
        ["git", "apply", "--whitespace=fix", "/tmp/gold_patch.patch"], workdir="/testbed"
    )
    scorer.assert_awaited_once_with(config["metadata"], sandbox, eval_timeout=600.0)
    agent.assert_not_called()
    assert stopped == [True]
