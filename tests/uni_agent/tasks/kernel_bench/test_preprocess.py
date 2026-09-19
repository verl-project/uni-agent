from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from uni_agent.tasks.kernel_bench.preprocess import (
    SourceRecord,
    add_drkernel_input_groups,
    assert_disjoint,
    build_row,
    discover_drkernel_records,
    discover_records,
    filter_records,
    main,
    select_benchmark_levels,
    select_records,
    stable_uid,
)


pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def write_task(root: Path, name: str, code: str) -> None:
    path = root / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps({"cases": [{"shape": [1]}]}), encoding="utf-8")


def test_uid_is_stable_and_does_not_contain_shuffle_index(tmp_path: Path) -> None:
    root = tmp_path / "source"
    write_task(root, "level1/7_vector_add", "REFERENCE = 'add'\n")
    record = discover_records(root)[0]
    first = stable_uid(record, dataset_name="bench", dataset_revision="abc123")
    second = stable_uid(record, dataset_name="bench", dataset_revision="abc123")
    row = build_row(record, dataset_name="bench", dataset_revision="abc123", split="train")

    assert first == second == row["uid"]
    assert row["extra_info"]["uid"] == first
    assert row["extra_info"]["tools_kwargs"]["task"]["metadata"]["source_id"] == "level1/7_vector_add"
    metadata = row["extra_info"]["tools_kwargs"]["task"]["metadata"]
    assert metadata["support_files"] == [
        {"name": f"{metadata['op_name']}.json", "content": json.dumps({"cases": [{"shape": [1]}]})}
    ]
    assert metadata["arch"] == "ascend910b1"
    assert metadata["benchmark_level"] == "1"
    assert metadata["benchmark_problem_id"] == "7"
    assert metadata["benchmark_name"] == "vector_add"
    assert metadata["recipe_schema_version"] == "triton-agent-recipe-v2"
    assert len(metadata["emitted_semantics_sha256"]) == 64


def test_uid_changes_with_emitted_arch_semantics(tmp_path: Path) -> None:
    root = tmp_path / "source"
    write_task(root, "vector_add", "CASES = 'vector_add.json'\n")
    record = discover_records(root)[0]

    default = stable_uid(record, dataset_name="bench", dataset_revision="r1")
    different_arch = stable_uid(
        record,
        dataset_name="bench",
        dataset_revision="r1",
        arch="ascend910c",
    )

    assert default != different_arch


def test_minimal_cli_prepares_training_data_without_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = tmp_path / "train"
    validation = tmp_path / "validation"
    output = tmp_path / "output"
    write_task(train, "level1/add", "REFERENCE = 'add'\n")
    write_task(validation, "level1/mul", "REFERENCE = 'mul'\n")
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

    summary = json.loads((output / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["dataset_name"] == "npukernelbench"
    assert summary["dataset_revision"] == "local"
    assert "source_manifest_sha256" not in summary
    assert (output / "train.jsonl").is_file()
    assert (output / "validation.jsonl").is_file()


def test_split_check_rejects_renamed_content(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    val_root = tmp_path / "validation"
    write_task(train_root, "original", "REFERENCE = 'same'\n")
    write_task(val_root, "renamed", "REFERENCE = 'same'\n")

    with pytest.raises(ValueError, match="content_fingerprints"):
        assert_disjoint(discover_records(train_root), discover_records(val_root))


def test_missing_case_sidecar_fails_before_rollout(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "broken.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing case sidecar"):
        discover_records(root)


def test_jsonl_sidecar_is_preserved_and_hardcoded_name_is_rewritten(tmp_path: Path) -> None:
    root = tmp_path / "source"
    path = root / "10_Relu.py"
    path.parent.mkdir(parents=True)
    path.write_text('CASES = "10_Relu.json"\n', encoding="utf-8")
    raw_cases = '{"shape":[1]}\n{"shape":[2]}\n'
    path.with_suffix(".json").write_text(raw_cases, encoding="utf-8")

    record = discover_records(root)[0]
    row = build_row(record, dataset_name="bench", dataset_revision="r1", split="train")
    metadata = row["extra_info"]["tools_kwargs"]["task"]["metadata"]

    canonical = f"{metadata['op_name']}.json"
    assert metadata["support_files"] == [{"name": canonical, "content": raw_cases}]
    assert f'CASES = "{canonical}"' in metadata["task_code"] or f"CASES = '{canonical}'" in metadata["task_code"]


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


def test_level_filter_normalizes_level_spellings(tmp_path: Path) -> None:
    root = tmp_path / "source"
    write_task(root, "level_1/7_add", "REFERENCE = 'add'\n")
    write_task(root, "level2/8_mul", "REFERENCE = 'mul'\n")
    records = discover_records(root)

    assert [record.metadata["benchmark_name"] for record in select_benchmark_levels(records, {"level1"})] == ["add"]
    assert len(select_benchmark_levels(records, {"1", "level_2"})) == 2


def test_seeded_max_rows_never_changes_record_uid() -> None:
    records = [
        SourceRecord(str(index), f"CODE={index}\n", {"cases.json": "{}\n"}, f"fingerprint-{index}")
        for index in range(5)
    ]
    full_uids = {
        record.source_id: stable_uid(record, dataset_name="bench", dataset_revision="r1") for record in records
    }
    chosen = select_records(records, seed=7, max_rows=3)

    assert [record.source_id for record in chosen] == ["4", "0", "3"]
    assert all(
        stable_uid(record, dataset_name="bench", dataset_revision="r1") == full_uids[record.source_id]
        for record in chosen
    )


def test_static_shape_filter_preserves_unknowns_and_rejects_known_oversize() -> None:
    large = SourceRecord(
        "large",
        """\
import torch
M = 64
N = 64
def get_inputs():
    x = torch.randn(M, N)
    return [x]
""",
        {"cases.json": "{}\n"},
        "large-fingerprint",
    )
    dynamic = SourceRecord(
        "dynamic",
        "def get_inputs():\n    return make_dynamic_input()\n",
        {"cases.json": "{}\n"},
        "dynamic-fingerprint",
    )

    selected = filter_records(
        [large, dynamic],
        profile="none",
        max_input_elements=1024,
    )
    assert [record.source_id for record in selected] == ["dynamic"]


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
