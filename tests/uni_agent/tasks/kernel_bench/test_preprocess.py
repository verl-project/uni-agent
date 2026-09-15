from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from uni_agent.tasks.kernel_bench.preprocess import (
    SourceRecord,
    add_drkernel_input_groups,
    discover_drkernel_records,
    main,
    select_records,
    stable_uid,
)

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_minimal_drkernel_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train = tmp_path / "training_split.parquet"
    validation = tmp_path / "validation_level1.parquet"
    train.touch()
    validation.touch()
    output = tmp_path / "output"

    def load_dataset(_kind, *, data_files, split):
        return [{"code": "def get_inputs():\n    return []\n# " + Path(data_files).stem, "name": Path(data_files).stem}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocess",
            "--train-source",
            str(train),
            "--validation-source",
            str(validation),
            "--output-dir",
            str(output),
            "--format",
            "jsonl",
        ],
    )
    main()
    summary = json.loads((output / "dataset_summary.json").read_text())
    assert summary["dataset_kind"] == summary["dataset_name"] == "drkernel"
    rows = [json.loads((output / f"{split}.jsonl").read_text()) for split in ("train", "validation")]
    assert all(row["extra_info"]["tools_kwargs"]["task"]["name"] == "npu_triton_kernel" for row in rows)
    assert rows[0]["uid"] != rows[1]["uid"]


def test_drkernel_dynamic_input_groups_preserve_official_case_zero() -> None:
    source = """\
import torch
batch_size = 8

def get_inputs():
    return [torch.rand(batch_size, 4)]

class Model:
    def forward(self, value):
        return value
"""
    augmented, dynamic_batch_size = add_drkernel_input_groups(source, 5)

    assert dynamic_batch_size == 8
    assert "_TRITON_GENERATED_CASE_COUNT = 5" in augmented
    assert "_TRITON_DYNAMIC_CASE_SIZES = (8, 1, 2, 3, 7)" in augmented
    compile(augmented, "<test-drkernel>", "exec")


def test_seeded_max_rows_never_changes_record_uid() -> None:
    records = [
        SourceRecord(str(index), f"CODE={index}\n", {"cases.json": "{}\n"}, f"fingerprint-{index}")
        for index in range(5)
    ]
    full_uids = {record.source_id: stable_uid(record, dataset_name="bench") for record in records}
    chosen = select_records(records, seed=7, max_rows=3)

    assert [record.source_id for record in chosen] == ["4", "0", "3"]
    assert all(stable_uid(record, dataset_name="bench") == full_uids[record.source_id] for record in chosen)


def test_drkernel_loader_deduplicates_and_namespaces_cross_level_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    level1 = tmp_path / "validation_level1.parquet"
    level2 = tmp_path / "validation_level2.parquet"
    training = tmp_path / "training_split.parquet"
    for path in (level1, level2, training):
        path.touch()
    code_a = "def get_inputs():\n    return []\nclass Model:\n    pass\n"
    code_b = code_a + "# level two\n"
    rows = {
        level1.name: [
            {"code": code_a, "name": "18_Add", "extra_info": {"problem_id": 18}},
            {
                "code": code_a + "# invalid\n",
                "name": "66_Matmul_Dropout_Softmax",
                "extra_info": {"problem_id": 66},
            },
        ],
        level2.name: [{"code": code_b, "name": "18_Mul", "extra_info": {"problem_id": 18}}],
        training.name: [
            {"code": code_a, "name": "A", "extra_info": {"uuid": "duplicate-id"}},
            {"code": code_a, "name": "A", "extra_info": {"uuid": "duplicate-id"}},
            {"code": code_b, "name": "B", "extra_info": {"uuid": "duplicate-id"}},
        ],
    }

    def fake_load_dataset(_kind, *, data_files, split):
        assert split == "train"
        return rows[Path(data_files).name]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    validation = discover_drkernel_records(tmp_path, split="validation", num_cases=2)
    train = discover_drkernel_records(tmp_path, split="train", num_cases=2)

    assert len(validation) == 2
    assert {record.metadata["benchmark_level"] for record in validation} == {"1", "2"}
    assert len({record.source_id for record in validation}) == 2
    assert all("matmuldropoutsoftmax" not in record.source_id for record in validation)
    assert len(train) == 2
    assert len({record.source_id for record in train}) == 2
