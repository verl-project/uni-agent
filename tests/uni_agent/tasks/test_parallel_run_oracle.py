import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pyarrow.parquet as pq
import pytest
from datasets import Dataset

from examples.inference import parallel_run_oracle as runner
from uni_agent.tasks.base import TaskResult

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result, expected",
    [
        (TaskResult(reward=1, finished=True, extra_info={"resolved": True}), "passed"),
        (TaskResult(reward=0, extra_info={"resolved": False, "eval_completed": True}), "oracle_failed"),
        # An installation/collection failure can be unresolved with eval_completed=True.
        # Keep the unresolved result; do not claim the gold patch itself is defective.
        (
            TaskResult(
                reward=0,
                extra_info={"resolved": False, "eval_completed": True, "eval_report": {"test_status": {}}},
            ),
            "oracle_failed",
        ),
        (RuntimeError("sandbox unavailable"), "execution_error"),
        (TaskResult(reward=1, extra_info={"resolved": True, "eval_completed": False}), "execution_error"),
        (TaskResult(reward=0, extra_info={"eval_report": {"found_eval_status": False}}), "execution_error"),
        (
            TaskResult(reward=1, extra_info={"agent": {"mode": "oracle", "exit_code": -1}}),
            "execution_error",
        ),
        (TaskResult(reward=1, finished=False), "execution_error"),
        (TaskResult(), "execution_error"),
        (TaskResult(reward=0.5), "oracle_failed"),
        (TaskResult(reward=1, extra_info={"resolved": False}), "oracle_failed"),
        (TaskResult(reward=1), "passed"),
    ],
)
async def test_actor_classifies_admission_and_retains_verifier_evidence(monkeypatch, result, expected):
    task = SimpleNamespace(run=AsyncMock())
    if isinstance(result, Exception):
        task.run.side_effect = result
    else:
        task.run.return_value = result
    monkeypatch.setattr(runner, "get_task", lambda config: task)

    record = await runner.OracleEvalActor(None, 1).run_single({"metadata": {"instance_id": "sample"}})

    assert record["outcome"] == expected
    assert record["solved"] is (expected == "passed")
    assert record["status"] == ("error" if expected == "execution_error" else "completed")
    if isinstance(result, TaskResult):
        assert record["extra_info"] == result.extra_info
        assert record["reward"] == result.reward
    else:
        assert record["error"] == "RuntimeError: sandbox unavailable"


def test_main_exports_only_tested_passing_rows_with_duplicate_ids(monkeypatch, tmp_path):
    rows = [
        {
            "prompt": [{"role": "user", "content": f"row {index}"}],
            "extra_info": {
                "tools_kwargs": {
                    "task": {
                        "name": "swe_rebench",
                        "sandbox": {"provider": "docker", "image": "original-image"},
                        "metadata": {"instance_id": "repeated-id", "index": index},
                    }
                }
            },
        }
        for index in range(4)
    ]
    source = Dataset.from_list(rows)
    source_path = tmp_path / "source.parquet"
    source.to_parquet(source_path)
    result_path = tmp_path / "results.json"
    passed_path = tmp_path / "passed.parquet"
    observed_configs = []

    def get_task(config):
        observed_configs.append(config)
        index = config["metadata"]["index"]
        run = AsyncMock()
        if index == 2:
            run.side_effect = RuntimeError("sandbox unavailable")
        else:
            run.return_value = TaskResult(reward=int(index == 0), extra_info={"resolved": index == 0})
        return SimpleNamespace(run=run)

    class Ref:
        def __init__(self, value):
            self.value = value

    def remote(actor_cls):
        def actor(log_dir, concurrency):
            instance = actor_cls(log_dir, concurrency)
            return SimpleNamespace(
                run_single=SimpleNamespace(remote=lambda config: Ref(asyncio.run(instance.run_single(config))))
            )

        return SimpleNamespace(remote=actor)

    monkeypatch.setattr(runner, "get_task", get_task)
    load_dataset = runner.load_dataset
    monkeypatch.setattr(
        runner, "load_dataset", lambda *args, **kwargs: load_dataset(*args, **kwargs, cache_dir=str(tmp_path / "cache"))
    )
    monkeypatch.setattr(runner.ray, "init", lambda: None)
    monkeypatch.setattr(runner.ray, "remote", remote)
    # Finish in reverse order to check source-row/result alignment.
    monkeypatch.setattr(runner.ray, "wait", lambda refs, **kwargs: ([refs[-1]], refs[:-1]))
    monkeypatch.setattr(runner.ray, "get", lambda ref: ref.value)
    monkeypatch.delenv("SANDBOX_PROVIDER", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parallel_run_oracle.py",
            "--data-path",
            str(source_path),
            "--result-path",
            str(result_path),
            "--passed-data-path",
            str(passed_path),
            "--limit",
            "3",
            "--num-workers",
            "1",
            "--concurrency",
            "1",
            "--log-dir",
            "",
        ],
    )

    runner.main()

    exported = Dataset(pq.read_table(passed_path))
    assert exported.to_list() == rows[:1]
    assert exported.features == source.features
    assert Dataset(pq.read_table(source_path)).to_list() == rows
    audit = json.loads(result_path.read_text())
    assert [row["row_index"] for row in audit["results"]] == [0, 1, 2]
    assert [row["outcome"] for row in audit["results"]] == ["passed", "oracle_failed", "execution_error"]
    assert audit["summary"]["status_counts"] == {"completed": 2, "error": 1}
    assert audit["summary"]["oracle_failed"] == 1
    assert audit["summary"]["solved"] == 1
    assert audit["summary"]["solved_rate"] == pytest.approx(1 / 3)
    assert all(config["run_oracle_solution"] is True for config in observed_configs)
    assert all(config["sandbox"]["image"] == "original-image" for config in observed_configs)
    assert all(row["sandbox"]["image"] == "original-image" for row in audit["results"])


def test_zero_pass_export_preserves_schema_and_refuses_overwrite(tmp_path):
    source = Dataset.from_list([{"instance_id": "failed", "metadata": {"patch": "unchanged"}}])
    output = tmp_path / "empty.parquet"
    runner._write_passed_dataset(source, [{"row_index": 0, "outcome": "oracle_failed"}], output)
    exported = Dataset(pq.read_table(output))
    assert len(exported) == 0
    assert exported.features == source.features
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        runner._write_passed_dataset(source, [{"row_index": 0, "outcome": "passed"}], output)
    assert output.read_bytes() == before


@pytest.mark.parametrize("options", [["--limit", "0"], ["--passed-data-path", "new.parquet"]])
def test_invalid_admission_arguments_fail_before_starting_ray(monkeypatch, options):
    init = AsyncMock()
    monkeypatch.setattr(runner.ray, "init", init)
    monkeypatch.setattr(sys, "argv", ["oracle", "--data-path", "unused.parquet", *options])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2
    init.assert_not_called()
