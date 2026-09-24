"""Download and split the DeepEyes visual-toolbox dataset.

Example::

    python -m uni_agent.tasks.deepeyes.preprocess \
        --local-save-dir ~/data/deepeyes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

DATASET_REPO = "ChenShawn/DeepEyes-Datasets-47k"
DATASET_FILENAME = "data_0.1.2_visual_toolbox_v2.parquet"
DEFAULT_VALIDATION_SIZE = 48
DEFAULT_BATCH_SIZE = 512
DEFAULT_SEED = 42


def validation_positions(*, row_count: int, validation_size: int, seed: int = DEFAULT_SEED) -> list[int]:
    """Choose reproducible random validation row positions."""

    import numpy as np

    if validation_size <= 0:
        raise ValueError("validation_size must be positive")
    if validation_size >= row_count:
        raise ValueError(f"validation_size must be smaller than source rows ({row_count})")
    positions = np.random.default_rng(seed).choice(row_count, validation_size, replace=False)
    return sorted(int(position) for position in positions)


def prepare_deepeyes(
    *,
    output_dir: str | Path,
    source_file: str | Path | None = None,
    repo_id: str = DATASET_REPO,
    filename: str = DATASET_FILENAME,
    revision: str = "main",
    validation_size: int = DEFAULT_VALIDATION_SIZE,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    compression: str | None = "snappy",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Prepare train/validation Parquet files and return their audit manifest."""

    import pyarrow.parquet as pq

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": destination / "train.parquet",
        "validation": destination / "val.parquet",
        "manifest": destination / "manifest.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Output already exists: {', '.join(map(str, existing))}. Use overwrite=True.")

    resolved_revision = None
    if source_file is None:
        source_path, resolved_revision = _download_source(
            output_dir=destination,
            repo_id=repo_id,
            filename=filename,
            revision=revision,
        )
    else:
        source_path = Path(source_file).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Source file does not exist: {source_path}")

    parquet_file = pq.ParquetFile(source_path)
    _validate_schema(parquet_file)
    row_count = parquet_file.metadata.num_rows
    selected_positions = validation_positions(
        row_count=row_count,
        validation_size=validation_size,
        seed=seed,
    )

    with tempfile.TemporaryDirectory(prefix="deepeyes-prepare-", dir=destination) as temporary_dir:
        temporary_root = Path(temporary_dir)
        temporary_train = temporary_root / "train.parquet"
        temporary_validation = temporary_root / "val.parquet"
        train_positions, actual_validation_positions = _write_split_files(
            source_path=source_path,
            train_path=temporary_train,
            validation_path=temporary_validation,
            validation_positions=set(selected_positions),
            batch_size=batch_size,
            compression=compression,
        )
        validation_indices = _read_extra_info_indices(temporary_validation)
        manifest = {
            "dataset_repo": repo_id if source_file is None else None,
            "dataset_file": filename if source_file is None else None,
            "requested_revision": revision if source_file is None else None,
            "resolved_revision": resolved_revision,
            "source_file": str(source_path),
            "sampler": "numpy.random.default_rng(seed).choice(row_count, validation_size, replace=False)",
            "seed": seed,
            "source_rows": row_count,
            "train_rows": len(train_positions),
            "validation_rows": len(actual_validation_positions),
            "validation_row_positions": selected_positions,
            "validation_indices": [_json_scalar(value) for value in validation_indices],
            "outputs": {
                "train": str(outputs["train"]),
                "validation": str(outputs["validation"]),
            },
        }
        temporary_manifest = temporary_root / "manifest.json"
        temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_train, outputs["train"])
        os.replace(temporary_validation, outputs["validation"])
        os.replace(temporary_manifest, outputs["manifest"])

    return manifest


def _download_source(*, output_dir: Path, repo_id: str, filename: str, revision: str) -> tuple[Path, str | None]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise RuntimeError("Install huggingface_hub or pass --source-file.") from error

    source_dir = output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    resolved_revision = None
    try:
        resolved_revision = HfApi().dataset_info(repo_id=repo_id, revision=revision).sha
    except Exception as error:  # noqa: BLE001 - the requested/cached revision may still download
        print(f"Warning: could not resolve Hugging Face revision {revision!r}: {error}", file=sys.stderr)
    source_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        revision=resolved_revision or revision,
        local_dir=source_dir,
    )
    return Path(source_path), resolved_revision


def _validate_schema(parquet_file: Any) -> None:
    required = {"prompt", "images", "reward_model", "extra_info"}
    missing = required.difference(parquet_file.schema_arrow.names)
    if missing:
        raise ValueError(f"DeepEyes Parquet is missing required columns: {sorted(missing)}")
    extra_info_type = parquet_file.schema_arrow.field("extra_info").type
    if not hasattr(extra_info_type, "get_field_index") or extra_info_type.get_field_index("index") < 0:
        raise ValueError("DeepEyes Parquet extra_info must contain an index field")


def _set_split_label(table: Any, *, label: str):
    import pyarrow as pa

    column_index = table.schema.get_field_index("extra_info")
    extra_info = table.column(column_index).combine_chunks()
    split_index = extra_info.type.get_field_index("split")
    if split_index < 0:
        return table
    children = [extra_info.field(index) for index in range(extra_info.type.num_fields)]
    children[split_index] = pa.array([label] * len(table), type=extra_info.type[split_index].type)
    replacement = pa.StructArray.from_arrays(children, type=extra_info.type)
    return table.set_column(column_index, "extra_info", replacement)


def _take_rows(table: Any, positions: list[int]):
    import pyarrow as pa

    if not positions:
        return None
    return table.take(pa.array(positions, type=pa.int64()))


def _write_split_files(
    *,
    source_path: Path,
    train_path: Path,
    validation_path: Path,
    validation_positions: set[int],
    batch_size: int,
    compression: str | None,
) -> tuple[list[int], list[int]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(source_path)
    train_writer = None
    validation_writer = None
    train_positions: list[int] = []
    selected_validation_positions: list[int] = []
    source_position = 0
    try:
        for record_batch in parquet_file.iter_batches(batch_size=batch_size, use_threads=False):
            table = pa.Table.from_batches([record_batch])
            local_train = []
            local_validation = []
            for local_position in range(table.num_rows):
                absolute_position = source_position + local_position
                if absolute_position in validation_positions:
                    local_validation.append(local_position)
                    selected_validation_positions.append(absolute_position)
                else:
                    local_train.append(local_position)
                    train_positions.append(absolute_position)
            source_position += table.num_rows

            train_table = _take_rows(table, local_train)
            if train_table is not None:
                train_table = _set_split_label(train_table, label="train")
                if train_writer is None:
                    train_writer = pq.ParquetWriter(train_path, train_table.schema, compression=compression)
                train_writer.write_table(train_table)

            validation_table = _take_rows(table, local_validation)
            if validation_table is not None:
                validation_table = _set_split_label(validation_table, label="validation")
                if validation_writer is None:
                    validation_writer = pq.ParquetWriter(
                        validation_path,
                        validation_table.schema,
                        compression=compression,
                    )
                validation_writer.write_table(validation_table)
    finally:
        if train_writer is not None:
            train_writer.close()
        if validation_writer is not None:
            validation_writer.close()

    if source_position != parquet_file.metadata.num_rows:
        raise RuntimeError(f"Read {source_position} rows, expected {parquet_file.metadata.num_rows}")
    return train_positions, selected_validation_positions


def _read_extra_info_indices(path: Path) -> list[Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["extra_info"])
    return [row["index"] for row in table["extra_info"].to_pylist()]


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-save-dir",
        "--output-dir",
        dest="output_dir",
        type=Path,
        required=True,
        help="Directory for source/, train.parquet, val.parquet, and manifest.json.",
    )
    parser.add_argument("--source-file", type=Path, help="Use a local Parquet instead of downloading.")
    parser.add_argument("--repo-id", default=DATASET_REPO)
    parser.add_argument("--filename", default=DATASET_FILENAME)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--validation-size", type=int, default=DEFAULT_VALIDATION_SIZE)
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Random seed for the validation split (default: 42)."
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--compression",
        choices=("snappy", "gzip", "brotli", "zstd", "lz4", "none"),
        default="snappy",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = prepare_deepeyes(
        output_dir=args.output_dir,
        source_file=args.source_file,
        repo_id=args.repo_id,
        filename=args.filename,
        revision=args.revision,
        validation_size=args.validation_size,
        seed=args.seed,
        batch_size=args.batch_size,
        compression=None if args.compression == "none" else args.compression,
        overwrite=args.overwrite,
    )
    print(f"Train:      {manifest['outputs']['train']} ({manifest['train_rows']} rows)")
    print(f"Validation: {manifest['outputs']['validation']} ({manifest['validation_rows']} rows)")
    print(f"Manifest:   {Path(args.output_dir).expanduser().resolve() / 'manifest.json'}")


if __name__ == "__main__":
    main()
