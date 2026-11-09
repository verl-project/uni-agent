"""Build a Uni-Agent Parquet index for DeepSWE v1.1."""

import argparse
from pathlib import Path

from datasets import Dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", required=True, help="Path to deep-swe/tasks.")
    parser.add_argument("--output", required=True, help="Output Parquet path.")
    args = parser.parse_args()

    task_root = Path(args.task_root).expanduser().resolve()
    task_dirs = sorted(path.parent for path in task_root.rglob("task.toml"))
    if len(task_dirs) != 113:
        raise ValueError(f"expected 113 DeepSWE v1.1 tasks, found {len(task_dirs)}")

    rows = []
    for task_dir in task_dirs:
        instruction = (task_dir / "instruction.md").read_text(encoding="utf-8").strip()
        instance_id = f"datacurve/deep-swe-1-1/{task_dir.relative_to(task_root).as_posix()}"
        rows.append(
            {
                "data_source": "datacurve/deep-swe-1-1",
                "prompt": [{"role": "user", "content": instruction}],
                "extra_info": {
                    "tools_kwargs": {
                        "task": {
                            "name": "harbor",
                            "metadata": {
                                "instance_id": instance_id,
                                "dataset": "datacurve/deep-swe-1-1",
                                "task_path": str(task_dir),
                            },
                        }
                    }
                },
            }
        )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(rows)
    dataset.to_parquet(output)
    print(f"Wrote {len(dataset)} tasks to {output}")


if __name__ == "__main__":
    main()
