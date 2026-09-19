"""Prepare deterministic, leakage-free KernelBench datasets.

This script consumes local source trees and emits datasets ready for training.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import random
import re
import tokenize
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

SYSTEM_PROMPT = "You are a coding agent optimizing Triton Ascend operators in an isolated NPU sandbox."
USER_PROMPT = """Implement the Ascend Triton operator in {workspace}/src/{op_name}.py.

Work only in the provided workspace. Put the implementation in
src/{op_name}_triton_ascend_impl.py. Use the image-provided verifier exactly as
documented in INSTRUCTIONS.md; do not alter verifier inputs or protected tools.
Correctness is required before latency optimization. Finish with the best fully
verified implementation in place.
"""

_UID_NAMESPACE = uuid.UUID("3dd30b37-4ce7-5257-9e15-048d7c0618d0")
_RECIPE_SCHEMA_VERSION = "triton-agent-recipe-v2"
_DRKERNEL_INVALID_VALIDATION_REFERENCES = {
    "66matmuldropoutsoftmax",
    "80gemmmaxsubtractgelu",
    "83conv3dgroupnormminclampdropout",
    "92cumsumexclusive",
}
_LEGACY_WARMUP_EXCLUDES = (
    "conv_transpose",
    "conv_transposed",
    "transpose3d",
    "transposed_3d",
    "conv3d",
    "3d_convolution",
    "attention",
    "transformer",
    "conv2d",
    "conv_standard",
)


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    code: str
    support_files: dict[str, str]
    fingerprint: str
    metadata: dict[str, Any] = field(default_factory=dict)


def discover_records(source_root: Path) -> list[SourceRecord]:
    """Read ``*.py`` tasks and their same-stem ``*.json`` case sidecars."""

    root = source_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {root}")
    records = []
    for python_file in sorted(root.rglob("*.py")):
        if python_file.name.endswith("_triton_ascend_impl.py"):
            continue
        sidecar = python_file.with_suffix(".json")
        if not sidecar.is_file():
            raise FileNotFoundError(f"missing case sidecar for {python_file}: {sidecar}")
        code = python_file.read_text(encoding="utf-8")
        cases = sidecar.read_text(encoding="utf-8")
        # Validate now so malformed cases do not fail after expensive rollout.
        _validate_json_or_jsonl(cases, path=sidecar)
        source_id = python_file.relative_to(root).with_suffix("").as_posix()
        support = {sidecar.name: cases}
        fingerprint = _fingerprint(code, support)
        level, problem_id, display_name = _npukernelbench_identity(source_id)
        records.append(
            SourceRecord(
                source_id,
                code,
                support,
                fingerprint,
                {
                    "dataset_kind": "npukernelbench",
                    "benchmark_level": level,
                    "benchmark_problem_id": problem_id,
                    "benchmark_name": display_name,
                },
            )
        )
    if not records:
        raise ValueError(f"no Python tasks found under {root}")
    return records


def assert_disjoint(train: list[SourceRecord], validation: list[SourceRecord]) -> None:
    """Reject both identity overlap and renamed content leakage."""

    train_ids = {record.source_id for record in train}
    validation_ids = {record.source_id for record in validation}
    duplicate_ids = sorted(train_ids & validation_ids)
    train_fingerprints = {record.fingerprint for record in train}
    validation_fingerprints = {record.fingerprint for record in validation}
    duplicate_content = sorted(train_fingerprints & validation_fingerprints)
    if duplicate_ids or duplicate_content:
        raise ValueError(
            "train/validation leakage detected: "
            f"source_ids={duplicate_ids[:10]} content_fingerprints={duplicate_content[:10]}"
        )


def stable_uid(
    record: SourceRecord,
    *,
    dataset_name: str,
    dataset_revision: str,
    workspace: str = "/workspace",
    arch: str = "ascend910b1",
) -> str:
    rendered = _render_record(record, workspace=workspace, arch=arch)
    identity = json.dumps(
        {
            "recipe_schema_version": _RECIPE_SCHEMA_VERSION,
            "dataset_name": dataset_name,
            "dataset_revision": dataset_revision,
            "source_id": record.source_id,
            "source_fingerprint": record.fingerprint,
            "emitted_semantics_sha256": rendered[4],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return str(uuid.uuid5(_UID_NAMESPACE, identity))


def build_row(
    record: SourceRecord,
    *,
    dataset_name: str,
    dataset_revision: str,
    split: str,
    workspace: str = "/workspace",
    arch: str = "ascend910b1",
) -> dict[str, Any]:
    op_name, task_code, support_files, prompt, semantics_digest = _render_record(
        record,
        workspace=workspace,
        arch=arch,
    )
    uid = stable_uid(
        record,
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
        workspace=workspace,
        arch=arch,
    )
    task_metadata = {
        "uid": uid,
        "op_name": op_name,
        "task_code": task_code,
        "support_files": support_files,
        "dataset_name": dataset_name,
        "dataset_revision": dataset_revision,
        "source_id": record.source_id,
        "source_fingerprint": record.fingerprint,
        "emitted_semantics_sha256": semantics_digest,
        "recipe_schema_version": _RECIPE_SCHEMA_VERSION,
        "split": split,
        "scenario": "npu_operator",
        "arch": arch,
        "entry_point": str(record.metadata.get("entry_point") or "Model"),
        "ability": str(record.metadata.get("ability") or "kernel_optimization"),
        "operator_backend": "triton_ascend",
        "dataset_kind": str(record.metadata.get("dataset_kind") or "npukernelbench"),
        "benchmark_name": str(record.metadata.get("benchmark_name") or record.source_id),
        "benchmark_problem_id": str(record.metadata.get("benchmark_problem_id") or record.source_id),
        "benchmark_level": str(record.metadata.get("benchmark_level") or ""),
        "source_metadata_json": json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
    }
    return {
        "uid": uid,
        "data_source": "npu_triton_kernel",
        "prompt": prompt,
        "reward_model": {"ground_truth": {"uid": uid}, "style": "rule"},
        "agent_name": "task",
        "extra_info": {
            "uid": uid,
            "tools_kwargs": {
                "task": {
                    "name": "npu_triton_kernel",
                    "prompt": prompt,
                    "metadata": task_metadata,
                }
            },
        },
    }


def _render_record(
    record: SourceRecord,
    *,
    workspace: str,
    arch: str,
) -> tuple[str, str, list[dict[str, str]], list[dict[str, str]], str]:
    """Render every UID-relevant task field without depending on the UID itself."""

    operator_seed = hashlib.sha256(
        f"{_RECIPE_SCHEMA_VERSION}\0{record.source_id}\0{record.fingerprint}".encode()
    ).hexdigest()[:12]
    op_name = _safe_operator_name(record.source_id, operator_seed)
    support_files = _canonical_support_files(record.support_files, op_name)
    primary_sidecar = sorted(name for name in record.support_files if name.endswith(".json"))[0]
    task_code = _rewrite_sidecar_literals(record.code, primary_sidecar, f"{op_name}.json")
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(workspace=workspace, op_name=op_name)},
    ]
    semantics = {
        "recipe_schema_version": _RECIPE_SCHEMA_VERSION,
        "workspace": workspace,
        "arch": arch,
        "op_name": op_name,
        "task_code": task_code,
        "support_files": support_files,
        "prompt": prompt,
        "record_metadata": record.metadata,
    }
    serialized = json.dumps(semantics, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return op_name, task_code, support_files, prompt, hashlib.sha256(serialized.encode()).hexdigest()


def build_splits(
    train_root: Path,
    validation_root: Path,
    *,
    dataset_name: str,
    dataset_revision: str,
    arch: str = "ascend910b1",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = discover_records(train_root)
    validation = discover_records(validation_root)
    _assert_unique_records(train, "train")
    _assert_unique_records(validation, "validation")
    assert_disjoint(train, validation)
    return rows_from_records(
        train,
        validation,
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
        arch=arch,
    )


def rows_from_records(
    train: list[SourceRecord],
    validation: list[SourceRecord],
    *,
    dataset_name: str,
    dataset_revision: str,
    arch: str = "ascend910b1",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        [
            build_row(
                record,
                dataset_name=dataset_name,
                dataset_revision=dataset_revision,
                split="train",
                arch=arch,
            )
            for record in train
        ],
        [
            build_row(
                record,
                dataset_name=dataset_name,
                dataset_revision=dataset_revision,
                split="validation",
                arch=arch,
            )
            for record in validation
        ],
    )


def discover_drkernel_records(
    source: Path,
    *,
    split: str,
    num_cases: int,
    validation_levels: set[int] | None = None,
) -> list[SourceRecord]:
    """Load the official DrKernel parquet layout and add deterministic input groups."""

    files = _drkernel_parquet_files(source, split=split, validation_levels=validation_levels)
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised in the training image
        raise RuntimeError("DrKernel parquet preparation requires the optional 'datasets' package") from exc
    records: list[SourceRecord] = []
    seen_fingerprints: set[str] = set()
    for parquet_file in files:
        dataset = load_dataset("parquet", data_files=str(parquet_file), split="train")
        file_level = _drkernel_file_level(parquet_file)
        for row in dataset:
            code = _drkernel_task_code(row)
            if not code.strip():
                continue
            augmented, dynamic_batch_size = add_drkernel_input_groups(code, num_cases)
            source_extra = _mapping(row.get("extra_info"))
            display_name = _drkernel_display_name(row, source_extra)
            if split == "validation" and _normalized_name(display_name) in _DRKERNEL_INVALID_VALIDATION_REFERENCES:
                continue
            level = _drkernel_level(row, source_extra, split, file_level=file_level)
            raw_id = str(
                source_extra.get("uuid")
                or source_extra.get("problem_id")
                or row.get("uuid")
                or row.get("problem_id")
                or hashlib.sha256(code.encode()).hexdigest()[:20]
            )
            # DrKernel reuses numeric problem IDs across validation levels and
            # contains a few UUID collisions. Namespace by level and add a
            # content-derived suffix so identity never depends on row order.
            source_id = (
                f"drkernel/{split}/level-{_normalized_name(level) or 'unknown'}/"
                f"{_normalized_name(raw_id) or 'task'}-{hashlib.sha256(code.encode()).hexdigest()[:12]}"
            )
            metadata = {
                "dataset_kind": "drkernel",
                "drkernel_num_cases": max(1, num_cases),
                "drkernel_dynamic_batch_size": dynamic_batch_size,
                "drkernel_split": split,
                "benchmark_name": display_name,
                "benchmark_problem_id": raw_id,
                "benchmark_level": level,
            }
            for key in (
                "difficulty_level",
                "difficulty_score",
                "entry_point",
                "has_3d",
                "heavy_ops",
                "level",
                "module_name",
                "num_ops",
                "ops",
                "repo_name",
                "type",
                "uuid",
            ):
                if key in source_extra:
                    metadata[f"drkernel_{key}"] = _json_value(source_extra[key])
            support = {"cases.json": "{}\n"}
            fingerprint = _fingerprint(augmented, support)
            # The official training parquet has exact duplicate rows. Stable
            # de-duplication is by content, never by mutable row position.
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            records.append(
                SourceRecord(
                    source_id=source_id,
                    code=augmented,
                    support_files=support,
                    fingerprint=fingerprint,
                    metadata=metadata,
                )
            )
    if not records:
        raise ValueError(f"no usable DrKernel rows in {files}")
    _assert_unique_records(records, split)
    return records


def _drkernel_parquet_files(
    source: Path,
    *,
    split: str,
    validation_levels: set[int] | None,
) -> list[Path]:
    path = source.resolve()
    if path.is_file():
        if path.suffix != ".parquet":
            raise ValueError(f"DrKernel source file must be parquet: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"DrKernel source does not exist: {path}")
    if split == "train":
        files = sorted(path.glob("training_*.parquet"))
        if len(files) != 1:
            raise ValueError(f"expected exactly one DrKernel training parquet in {path}, found {len(files)}")
        return files
    if split != "validation":
        raise ValueError(f"unsupported DrKernel split: {split}")
    files = sorted(path.glob("validation_*.parquet"))
    if validation_levels:
        files = [file for file in files if _drkernel_file_level(file) in validation_levels]
    if not files:
        raise ValueError(f"no DrKernel validation parquets selected under {path}")
    levels = [_drkernel_file_level(file) for file in files]
    if any(level is None for level in levels) or len(set(levels)) != len(levels):
        raise ValueError(f"DrKernel validation files must contain one unique level each: {files}")
    return files


def _drkernel_file_level(path: Path) -> int | None:
    match = re.search(r"level_?(\d+)", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _drkernel_task_code(row: Mapping[str, Any]) -> str:
    code = row.get("code") or row.get("reference_code")
    if code:
        return str(code)
    return str(_mapping(row.get("reward_model")).get("ground_truth") or "")


def _drkernel_display_name(row: Mapping[str, Any], extra: Mapping[str, Any]) -> str:
    for value in (row.get("name"), extra.get("name"), extra.get("module_name")):
        text = str(value or "").strip()
        if text and text.lower() not in {"model", "modelnew", "forward"}:
            return text
    ops = extra.get("ops")
    if isinstance(ops, str):
        try:
            ops = json.loads(ops)
        except json.JSONDecodeError:
            return ops.strip()
    if isinstance(ops, list | tuple):
        names = []
        for op in ops:
            if isinstance(op, Mapping):
                op = op.get("name") or op.get("op") or op.get("type")
            text = str(op or "").strip()
            if text and text not in names:
                names.append(text)
        return "_".join(names)
    return ""


def _drkernel_level(
    row: Mapping[str, Any],
    extra: Mapping[str, Any],
    split: str,
    *,
    file_level: int | None = None,
) -> str:
    match = re.search(r"level_?(\d+)", str(row.get("data_source", "")), flags=re.IGNORECASE)
    if match:
        return match.group(1)
    if extra.get("level") not in (None, ""):
        return str(extra["level"])
    if row.get("level") not in (None, ""):
        return str(row["level"])
    if file_level is not None:
        return str(file_level)
    return split


def _npukernelbench_identity(source_id: str) -> tuple[str, str, str]:
    """Extract review/filter metadata from ``levelN/ID_Name`` paths."""

    path = PurePosixPath(source_id)
    level_match = None
    for part in reversed(path.parts[:-1]):
        level_match = re.search(r"^level_?(\d+)$", part, flags=re.IGNORECASE)
        if level_match:
            break
    match = re.match(r"^(\d+)[_-](.+)$", path.name)
    level = level_match.group(1) if level_match else ""
    problem_id = match.group(1) if match else path.name
    display_name = match.group(2) if match else path.name
    return level, problem_id, display_name


def _normalized_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _eval_int_expr(node: ast.AST, names: dict[str, int]) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _eval_int_expr(node.operand, names)
        return -value if value is not None else None
    if isinstance(node, ast.BinOp):
        left = _eval_int_expr(node.left, names)
        right = _eval_int_expr(node.right, names)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv) and right:
            return left // right
    return None


def _loaded_names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}


def _drkernel_dynamic_batch_size(code: str) -> int | None:
    """Return a safely variable batch size, or None for value-only cases."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    assignments: dict[str, ast.AST] = {}
    int_names: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign | ast.AnnAssign):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Name) or node.value is None:
                continue
            assignments[target.id] = node.value
            value = _eval_int_expr(node.value, int_names)
            if value is not None:
                int_names[target.id] = value
    batch_size = int_names.get("batch_size")
    if batch_size is None or batch_size <= 1:
        return None
    dependent_names = {"batch_size"}
    changed = True
    while changed:
        changed = False
        for name, value in assignments.items():
            if name not in dependent_names and _loaded_names(value) & dependent_names:
                dependent_names.add(name)
                changed = True
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)}
    get_inputs = functions.get("get_inputs")
    get_init_inputs = functions.get("get_init_inputs")
    model = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model"), None)
    input_names = _loaded_names(get_inputs) if get_inputs is not None else set()
    if "batch_size" not in input_names or input_names & (dependent_names - {"batch_size"}):
        return None
    if get_init_inputs is not None and _loaded_names(get_init_inputs) & dependent_names:
        return None
    if model is not None and _loaded_names(model) & dependent_names:
        return None
    return batch_size


def _drkernel_case_sizes(batch_size: int, count: int) -> tuple[int, ...]:
    preferred = (
        batch_size,
        1,
        2,
        3,
        7,
        31,
        batch_size // 4,
        batch_size // 2,
        3 * batch_size // 4,
        batch_size - 1,
    )
    sizes: list[int] = []
    for value in (*preferred, *range(1, batch_size + 1)):
        value = max(1, min(int(value), batch_size))
        if value not in sizes:
            sizes.append(value)
        if len(sizes) >= count:
            break
    return tuple(sizes)


def _shape_numel(shape: tuple[int, ...] | None) -> int | None:
    if not shape:
        return None
    total = 1
    for value in shape:
        total *= value
    return total


def _torch_factory_shape(node: ast.AST, names: dict[str, int]) -> tuple[int, ...] | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in {"rand", "randn", "empty", "zeros", "ones"}:
        return None
    if not isinstance(func.value, ast.Name) or func.value.id != "torch":
        return None
    shape = tuple(_eval_int_expr(arg, names) for arg in node.args)
    if any(value is None for value in shape):
        return None
    return tuple(int(value) for value in shape if value is not None)


def _attr_name(node: ast.AST) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _matmul_output_shape(
    node: ast.AST,
    input_shapes: dict[str, tuple[int, ...]],
) -> tuple[int, ...] | None:
    left: ast.AST | None = None
    right: ast.AST | None = None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        left, right = node.left, node.right
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "matmul"
        and len(node.args) >= 2
    ):
        left, right = node.args[:2]
    if left is None or right is None:
        return None
    left_shape = input_shapes.get(_attr_name(left) or "")
    if isinstance(right, ast.Attribute) and right.attr == "T":
        base_shape = input_shapes.get(_attr_name(right.value) or "")
        right_shape = tuple(reversed(base_shape)) if base_shape else None
    else:
        right_shape = input_shapes.get(_attr_name(right) or "")
    if not left_shape or not right_shape or len(left_shape) != 2 or len(right_shape) != 2:
        return None
    return left_shape[0], right_shape[1]


def infer_static_shape_limits(code: str) -> tuple[int | None, int | None]:
    """Best-effort legacy static input/output element estimates.

    Unknown or dynamic shapes intentionally remain ``None`` and are not
    rejected; runtime limits still belong to the isolated verifier.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, None
    names: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = _eval_int_expr(node.value, names)
            if value is not None:
                names[node.targets[0].id] = value
    input_shapes: dict[str, tuple[int, ...]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_inputs":
            for statement in node.body:
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                ):
                    shape = _torch_factory_shape(statement.value, names)
                    if shape:
                        input_shapes[statement.targets[0].id] = shape
    max_input = max((_shape_numel(shape) or 0 for shape in input_shapes.values()), default=0) or None
    output_numel: int | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            for statement in ast.walk(node):
                if isinstance(statement, ast.Return):
                    output_numel = _shape_numel(_matmul_output_shape(statement.value, input_shapes))
                    if output_numel is not None:
                        return max_input, output_numel
    return max_input, output_numel


def add_drkernel_input_groups(code: str, num_cases: int) -> tuple[str, int | None]:
    """Add deterministic multi-case generation without changing official case zero."""

    num_cases = max(1, num_cases)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None
    if (
        any(
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "get_input_groups"
            for node in tree.body
        )
        or num_cases == 1
    ):
        return code, None
    batch_size = _drkernel_dynamic_batch_size(code)
    case_sizes = _drkernel_case_sizes(batch_size, num_cases) if batch_size is not None else ()
    generated = f"""

# Generated by DrKernel preprocessing. Case 0 preserves the official shape.
_TRITON_GENERATED_CASES = True
_TRITON_GENERATED_CASE_COUNT = {num_cases}
_TRITON_DYNAMIC_BATCH_SIZE = {batch_size!r}
_TRITON_DYNAMIC_CASE_SIZES = {case_sizes!r}


def get_input_groups():
    global batch_size
    if _TRITON_DYNAMIC_BATCH_SIZE is None:
        return [get_inputs() for _ in range(_TRITON_GENERATED_CASE_COUNT)]

    original_batch_size = batch_size
    groups = []
    try:
        for size in _TRITON_DYNAMIC_CASE_SIZES:
            batch_size = size
            groups.append(get_inputs())
        while len(groups) < _TRITON_GENERATED_CASE_COUNT:
            batch_size = original_batch_size
            groups.append(get_inputs())
    finally:
        batch_size = original_batch_size
    return groups
"""
    augmented = code.rstrip() + generated
    compile(augmented, "<drkernel-task>", "exec")
    return augmented, batch_size


def _assert_unique_records(records: list[SourceRecord], split: str) -> None:
    ids = [record.source_id for record in records]
    fingerprints = [record.fingerprint for record in records]
    duplicate_ids = sorted({value for value in ids if ids.count(value) > 1})
    duplicate_content = sorted({value for value in fingerprints if fingerprints.count(value) > 1})
    if duplicate_ids or duplicate_content:
        raise ValueError(
            f"duplicate records inside {split}: source_ids={duplicate_ids[:10]} "
            f"content_fingerprints={duplicate_content[:10]}"
        )


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    for attribute in ("item", "tolist"):
        method = getattr(value, attribute, None)
        if callable(method):
            try:
                return _json_value(method())
            except Exception:
                pass
    return str(value)


def filter_records(
    records: list[SourceRecord],
    *,
    profile: str,
    include_keywords: tuple[str, ...] = (),
    exclude_keywords: tuple[str, ...] = (),
    exclude_ops: tuple[str, ...] = (),
    max_code_chars: int | None = None,
    max_input_elements: int | None = None,
    max_output_elements: int | None = None,
) -> list[SourceRecord]:
    """Apply the old warmup keyword policy through explicit, reviewable CLI options."""

    if profile not in {"legacy-warmup", "none"}:
        raise ValueError(f"unknown filter profile: {profile}")
    default_excludes = _LEGACY_WARMUP_EXCLUDES if profile == "legacy-warmup" else ()
    excludes = tuple(value.lower() for value in (*default_excludes, *exclude_keywords))
    includes = tuple(value.lower() for value in include_keywords)
    normalized_exclude_ops = {_normalized_name(value) for value in exclude_ops}
    selected = []
    for record in records:
        if max_code_chars is not None and len(record.code) > max_code_chars:
            continue
        if max_input_elements is not None or max_output_elements is not None:
            input_elements, output_elements = infer_static_shape_limits(record.code)
            if max_input_elements is not None and input_elements is not None:
                if input_elements > max_input_elements:
                    continue
            if max_output_elements is not None and output_elements is not None:
                if output_elements > max_output_elements:
                    continue
        text = "\n".join(
            (record.source_id, record.code, json.dumps(record.metadata, ensure_ascii=False, sort_keys=True))
        ).lower()
        display_name = record.metadata.get("benchmark_name") or record.source_id
        if _normalized_name(display_name) in normalized_exclude_ops:
            continue
        if includes and not any(keyword in text for keyword in includes):
            continue
        if excludes and any(keyword in text for keyword in excludes):
            continue
        selected.append(record)
    if not selected:
        raise ValueError("sample filter removed every record")
    return selected


def select_records(records: list[SourceRecord], *, seed: int | None, max_rows: int | None) -> list[SourceRecord]:
    """Seeded selection that never changes per-record UID identity."""

    selected = list(records)
    if seed is not None:
        random.Random(seed).shuffle(selected)
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive when provided")
        selected = selected[:max_rows]
    if not selected:
        raise ValueError("record selection produced an empty split")
    return selected


def select_benchmark_levels(records: list[SourceRecord], levels: set[str] | None) -> list[SourceRecord]:
    if not levels:
        return records
    normalized = {_normalized_level(level) for level in levels}
    selected = [record for record in records if _normalized_level(record.metadata.get("benchmark_level")) in normalized]
    if not selected:
        raise ValueError(f"level selection {sorted(levels)} removed every record")
    return selected


def _normalized_level(value: Any) -> str:
    return re.sub(r"^level", "", _normalized_name(value))


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        return
    if path.suffix != ".parquet":
        raise ValueError("output paths must end in .parquet or .jsonl")
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - developer environments may use JSONL
        raise RuntimeError("writing parquet requires the optional 'datasets' package") from exc
    Dataset.from_list(rows).to_parquet(str(path))


def _fingerprint(code: str, support_files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    digest.update(code.encode("utf-8"))
    # Exclude filenames: renamed copies across splits are still leakage. The
    # stable UID already includes source_id, while this digest represents task
    # semantics for disjointness checks.
    for content in sorted(support_files.values()):
        digest.update(b"\0")
        digest.update(content.encode("utf-8"))
    return digest.hexdigest()


def _safe_operator_name(source_id: str, uid: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", source_id).strip("_")[:80] or "operator"
    if stem[0].isdigit():
        stem = f"op_{stem}"
    return f"{stem}_{uid.replace('-', '')[:12]}"


def _canonical_support_files(support_files: dict[str, str], op_name: str) -> list[dict[str, str]]:
    """Name the primary case sidecar exactly as the verifier contract expects."""

    json_files = sorted((name, content) for name, content in support_files.items() if name.endswith(".json"))
    if not json_files:
        raise ValueError("each operator requires a JSON case sidecar")
    primary_name, primary_content = json_files[0]
    canonical_name = f"{op_name}.json"
    result = [{"name": canonical_name, "content": primary_content}]
    for name, content in sorted(support_files.items()):
        if name == primary_name:
            continue
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError(f"support filename must be a basename: {name!r}")
        if name == canonical_name:
            raise ValueError(f"extra support file conflicts with canonical case sidecar: {name}")
        result.append({"name": name, "content": content})
    return result


def _validate_json_or_jsonl(text: str, *, path: Path | None = None) -> None:
    try:
        json.loads(text)
        return
    except json.JSONDecodeError as document_error:
        nonempty = [(index, line) for index, line in enumerate(text.splitlines(), start=1) if line.strip()]
        if not nonempty:
            raise ValueError(f"empty JSON/JSONL case sidecar: {path or '<memory>'}") from document_error
        for line_number, line in nonempty:
            try:
                json.loads(line)
            except json.JSONDecodeError as line_error:
                raise ValueError(
                    f"invalid JSONL in {path or '<memory>'} at line {line_number}: {line_error}"
                ) from line_error


def _rewrite_sidecar_literals(code: str, original_name: str, canonical_name: str) -> str:
    """Rewrite exact string-literal references while preserving source formatting."""

    rewritten: list[tokenize.TokenInfo] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        for token in tokens:
            if token.type == tokenize.STRING:
                try:
                    value = ast.literal_eval(token.string)
                except (SyntaxError, ValueError):
                    value = None
                if isinstance(value, str) and (value == original_name or value.endswith(f"/{original_name}")):
                    prefix = value[: -len(original_name)]
                    token = tokenize.TokenInfo(
                        token.type,
                        repr(f"{prefix}{canonical_name}"),
                        token.start,
                        token.end,
                        token.line,
                    )
            rewritten.append(token)
    except (tokenize.TokenError, IndentationError):
        return code
    return tokenize.untokenize(rewritten)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--validation-source", type=Path, required=True)
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-revision", default="local")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("parquet", "jsonl"), default="parquet")
    parser.add_argument("--arch", default="ascend910b1")
    parser.add_argument("--dataset-kind", choices=("npukernelbench", "drkernel"), default="npukernelbench")
    parser.add_argument(
        "--npukernelbench-levels",
        default="1,2",
        help="Comma-separated benchmark levels; blank explicitly selects all reviewed levels.",
    )
    parser.add_argument("--drkernel-num-cases", type=int, default=10)
    parser.add_argument(
        "--drkernel-validation-levels",
        default="",
        help="Comma-separated official validation levels; blank selects every validation parquet.",
    )
    parser.add_argument("--filter-profile", choices=("legacy-warmup", "none"), default="none")
    parser.add_argument("--include-keyword", action="append", default=[])
    parser.add_argument("--exclude-keyword", action="append", default=[])
    parser.add_argument("--exclude-op", action="append", default=[])
    parser.add_argument("--max-code-chars", type=int)
    parser.add_argument("--max-input-elements", type=int)
    parser.add_argument("--max-output-elements", type=int)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--validation-shuffle-seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-validation-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset_name or args.dataset_kind
    if args.dataset_kind == "drkernel":
        levels = {int(value.strip()) for value in args.drkernel_validation_levels.split(",") if value.strip()} or None
        train_records = discover_drkernel_records(
            args.train_source,
            split="train",
            num_cases=args.drkernel_num_cases,
        )
        validation_records = discover_drkernel_records(
            args.validation_source,
            split="validation",
            num_cases=args.drkernel_num_cases,
            validation_levels=levels,
        )
    else:
        train_records = discover_records(args.train_source)
        validation_records = discover_records(args.validation_source)
        _assert_unique_records(train_records, "train")
        _assert_unique_records(validation_records, "validation")
        levels = {value.strip() for value in args.npukernelbench_levels.split(",") if value.strip()} or None
        train_records = select_benchmark_levels(train_records, levels)
        validation_records = select_benchmark_levels(validation_records, levels)
    train_records = filter_records(
        train_records,
        profile=args.filter_profile,
        include_keywords=tuple(args.include_keyword),
        exclude_keywords=tuple(args.exclude_keyword),
        exclude_ops=tuple(args.exclude_op),
        max_code_chars=args.max_code_chars,
        max_input_elements=args.max_input_elements,
        max_output_elements=args.max_output_elements,
    )
    validation_records = filter_records(
        validation_records,
        profile=args.filter_profile,
        include_keywords=tuple(args.include_keyword),
        exclude_keywords=tuple(args.exclude_keyword),
        exclude_ops=tuple(args.exclude_op),
        max_code_chars=args.max_code_chars,
        max_input_elements=args.max_input_elements,
        max_output_elements=args.max_output_elements,
    )
    assert_disjoint(train_records, validation_records)
    train_records = select_records(
        train_records,
        seed=args.shuffle_seed,
        max_rows=args.max_train_rows,
    )
    validation_records = select_records(
        validation_records,
        seed=args.validation_shuffle_seed,
        max_rows=args.max_validation_rows,
    )
    train, validation = rows_from_records(
        train_records,
        validation_records,
        dataset_name=dataset_name,
        dataset_revision=args.dataset_revision,
        arch=args.arch,
    )
    suffix = f".{args.format}"
    train_output = args.output_dir / f"train{suffix}"
    validation_output = args.output_dir / f"validation{suffix}"
    write_rows(train, train_output)
    write_rows(validation, validation_output)
    summary = {
        "recipe_schema_version": _RECIPE_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "dataset_revision": args.dataset_revision,
        "dataset_kind": args.dataset_kind,
        "arch": args.arch,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_output_sha256": hashlib.sha256(train_output.read_bytes()).hexdigest(),
        "validation_output_sha256": hashlib.sha256(validation_output.read_bytes()).hexdigest(),
        "train_uid_sha256": hashlib.sha256("\n".join(sorted(row["uid"] for row in train)).encode()).hexdigest(),
        "validation_uid_sha256": hashlib.sha256(
            "\n".join(sorted(row["uid"] for row in validation)).encode()
        ).hexdigest(),
    }
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
