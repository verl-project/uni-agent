#!/usr/bin/env python3
"""Select one already-preprocessed SWE-bench row without mutating the source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def _instance_id(row: dict) -> str | None:
    try:
        return row["extra_info"]["tools_kwargs"]["task"]["metadata"]["instance_id"]
    except (KeyError, TypeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--metadata-output", required=True)
    args = parser.parse_args()
    if args.index < 0:
        raise SystemExit("--index must be non-negative")

    input_path = str(Path(args.input).expanduser())
    dataset = load_dataset("parquet", data_files=input_path, split="train")
    rows = dataset.to_list()
    if args.instance_id:
        matches = [index for index, row in enumerate(rows) if _instance_id(row) == args.instance_id]
        if len(matches) != 1:
            raise SystemExit(f"expected exactly one row for instance_id={args.instance_id!r}, got {len(matches)}")
        selected_index = matches[0]
    else:
        if args.index >= len(rows):
            raise SystemExit(f"--index {args.index} is outside the {len(rows)}-row dataset")
        selected_index = args.index

    selected = rows[selected_index]
    instance_id = _instance_id(selected)
    if not isinstance(instance_id, str) or not instance_id:
        raise SystemExit("selected row has no extra_info.tools_kwargs.task.metadata.instance_id")
    task = selected["extra_info"]["tools_kwargs"]["task"]
    if task.get("name") != "swe_bench":
        raise SystemExit(f"selected row task name must be swe_bench, got {task.get('name')!r}")
    sandbox = task.get("sandbox")
    sandbox_image = sandbox.get("image") if isinstance(sandbox, dict) else None
    if not isinstance(sandbox_image, str) or not sandbox_image:
        raise SystemExit("selected row task has no sandbox.image")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.select([selected_index]).to_parquet(str(output))
    metadata = {
        "instance_id": instance_id,
        "source_index": selected_index,
        "source_path": input_path,
        "sandbox_image": sandbox_image,
    }
    metadata_path = Path(args.metadata_output)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, separators=(",", ":")))


if __name__ == "__main__":
    main()
