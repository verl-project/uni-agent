"""Download or index Harbor tasks as Uni-Agent parquet rows."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from datasets import Dataset

TASK_NAME = "harbor"


def download_dataset(dataset_ref: str, output_dir: Path | str) -> Path:
    """Download and export a Harbor dataset to a persistent local directory."""
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    command = ["harbor", "download", dataset_ref, "--export", "--output-dir", str(output_root)]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Harbor CLI executable was not found; install Harbor 0.16.0 or later") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"failed to download Harbor dataset {dataset_ref!r}") from exc

    dataset_dir = output_root / dataset_ref.split("@", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    if not dataset_dir.is_dir():
        raise RuntimeError(f"Harbor dataset was not exported to {dataset_dir}")
    return dataset_dir


def build_harbor_dataset(
    task_root: Path | str,
    *,
    dataset_name: str | None = None,
    max_instances: int | None = None,
) -> Dataset:
    """Build a dataset that points to Harbor tasks on the shared local filesystem."""
    root = Path(task_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Harbor task root does not exist: {root}")
    name = dataset_name or root.name
    if not name:
        raise ValueError("dataset_name must be non-empty")
    if max_instances is not None and max_instances < 0:
        raise ValueError("max_instances must be non-negative")

    task_dirs = sorted(
        {task_file.parent.resolve() for task_file in root.rglob("task.toml")},
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not task_dirs:
        raise ValueError(f"no Harbor tasks containing task.toml found under {root}")
    if max_instances is not None:
        task_dirs = task_dirs[:max_instances]

    rows = []
    for task_dir in task_dirs:
        relative_path = task_dir.relative_to(root)
        relative_id = task_dir.name if relative_path == Path(".") else relative_path.as_posix()
        instance_id = f"{name}/{relative_id}"
        instruction_path = task_dir / "instruction.md"
        prompt = (
            [{"role": "user", "content": instruction_path.read_text(encoding="utf-8").strip()}]
            if instruction_path.is_file()
            else []
        )
        metadata = {
            "instance_id": instance_id,
            "dataset": name,
            "task_path": str(task_dir),
        }
        task_config = {"name": TASK_NAME, "metadata": metadata}
        rows.append(
            {
                "data_source": name,
                "prompt": prompt,
                "extra_info": {"tools_kwargs": {"task": task_config}},
            }
        )
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess a Harbor dataset for Uni-Agent.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-ref", help="Harbor dataset reference to download, such as org/name@version.")
    source.add_argument("--task-root", help="Existing local directory containing Harbor tasks.")
    parser.add_argument("--local-save-dir", default="~/data/uni_agent")
    parser.add_argument("--max-instances", type=int, default=None, help="Keep only the first N tasks.")
    args = parser.parse_args()

    save_dir = Path(args.local_save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset_ref:
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.dataset_ref).strip("._")
        task_root = download_dataset(args.dataset_ref, save_dir / "harbor" / filename)
        dataset_name = args.dataset_ref.split("@", 1)[0]
    else:
        task_root = Path(args.task_root).expanduser().resolve()
        dataset_name = task_root.name
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name).strip("._")

    dataset = build_harbor_dataset(
        task_root,
        dataset_name=dataset_name,
        max_instances=args.max_instances,
    )
    output_path = save_dir / f"harbor_{filename or 'tasks'}.parquet"
    dataset.to_parquet(output_path)
    print(f"Wrote {len(dataset)} instances to {output_path}", flush=True)


if __name__ == "__main__":
    main()
