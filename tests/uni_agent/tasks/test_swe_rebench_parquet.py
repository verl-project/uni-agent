"""Real offline datasets/Arrow/Parquet I/O; synthetic rows, no benchmark download."""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from datasets import Dataset
from datasets import config as datasets_config

from uni_agent.tasks.swe_rebench.preprocess import build_swe_rebench, get_image_name

pytestmark = [pytest.mark.cpu, pytest.mark.level0]
ROOT = Path(__file__).resolve().parents[3]
IDS = ["NOhs__dkey-18", "EmilStenstrom__conllu-64", "example__unused-1"]


@pytest.fixture
def parquet(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache"))
    # datasets snapshots these environment variables at import time. Isolate
    # in-process cache writes as well as the CLI subprocess's environment.
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", tmp_path / "cache")
    monkeypatch.setattr(datasets_config, "HF_DATASETS_OFFLINE", True)
    rows = [
        {
            "instance_id": instance_id,
            "repo": "synthetic/repo",
            "base_commit": "synthetic-base",
            "patch": f"GOLD PATCH {index}",
            "test_patch": f"TEST PATCH {index}",
            "problem_statement": f"Synthetic problem {index} 中文",
            "FAIL_TO_PASS": '["test_fixed"]',
            "FAIL_TO_FAIL": "[]",
            "PASS_TO_PASS": '["test_existing"]',
            "PASS_TO_FAIL": "[]",
            "install_config": {"install": "echo install", "log_parser": "pytest", "test_cmd": "pytest"},
        }
        for index, instance_id in enumerate(IDS)
    ]
    path = tmp_path / "input.parquet"
    Dataset.from_list(rows).to_parquet(str(path))
    return path, rows


@pytest.mark.parametrize("selection,limit", [(None, None), ([IDS[1], IDS[0], IDS[1]], None), (None, 1)])
def test_preprocess_parquet_round_trip(parquet, tmp_path, selection, limit):
    source, rows = parquet
    result = build_swe_rebench(data_files=[str(source)], instance_ids=selection, max_instances=limit)
    output = tmp_path / "processed.parquet"
    result.to_parquet(str(output))
    restored = Dataset.from_parquet(str(output))
    wanted = list(dict.fromkeys(selection or IDS))[:limit]
    assert len(restored) == len(wanted)
    assert set(restored.column_names) == {"data_source", "prompt", "extra_info"}
    by_id = {row["instance_id"]: row for row in rows}
    for record, instance_id in zip(restored, wanted, strict=True):
        original = by_id[instance_id]
        task = record["extra_info"]["tools_kwargs"]["task"]
        assert task["metadata"]["instance_id"] == instance_id
        assert task["sandbox"]["image"] == get_image_name(instance_id)
        assert task["metadata"]["patch"] == original["patch"]
        assert task["metadata"]["test_patch"] == original["test_patch"]
        assert task["metadata"]["test_cmd"] == "pytest"
        assert record["prompt"] == [{"role": "user", "content": original["problem_statement"]}]
        assert "GOLD PATCH" not in str(record["prompt"])
        assert "TEST PATCH" not in str(record["prompt"])


@pytest.mark.parametrize("instance_ids,error", [(["missing"], "not found"), ([IDS[0]], "duplicate")])
def test_real_parquet_missing_and_duplicate_selection(parquet, tmp_path, instance_ids, error):
    _, rows = parquet
    path = tmp_path / "duplicate.parquet"
    Dataset.from_list([*rows, rows[0]]).to_parquet(str(path))
    with pytest.raises(ValueError, match=error):
        build_swe_rebench(data_files=[str(path)], instance_ids=instance_ids)


def test_readme_preprocess_command_is_runnable(parquet, tmp_path):
    source, _ = parquet
    readme = (ROOT / "examples/blackbox_recipes/opencode_swe_task/README.md").read_text()
    command = next(
        block
        for block in re.findall(r"```bash\n(.*?)```", readme, re.S)
        if "uni_agent.tasks.swe_rebench.preprocess" in block and "--data-files" in block
    )
    argv = shlex.split(command.replace("\\\n", ""))
    replacements = {
        "python": sys.executable,
        "/data/swe-rebench/filtered-00000-of-00001.parquet": str(source),
        "/data/uni_agent": str(tmp_path / "out"),
    }
    result = subprocess.run(
        [replacements.get(arg, arg) for arg in argv],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    output = Dataset.from_parquet(str(tmp_path / "out/swe_rebench_2task.parquet"))
    assert [row["extra_info"]["tools_kwargs"]["task"]["metadata"]["instance_id"] for row in output] == IDS[1::-1]
