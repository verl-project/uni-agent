"""Build DeepSWE or MiMo Parquet rows from local Harbor task directories."""

import argparse
from pathlib import Path

import yaml
from datasets import Dataset

HARNESS_VERSIONS = {
    "mini-swe-agent": "2.4.6",
    "claude-code": "2.1.148",
}


def load_tasks(task_name: str, task_root: Path, harnesses: list[str] | None = None) -> list[dict]:
    root = task_root.expanduser().resolve()
    if (root / "tasks").is_dir():
        root = root / "tasks"
    task_dirs = sorted(path.parent for path in root.rglob("task.toml"))
    if task_name == "mimo":
        source = "mimo-v26-rl-oss-code-terminal-harbor"
    else:
        source = "datacurve/deep-swe-1-1"
        if len(task_dirs) != 113:
            raise ValueError(f"expected 113 DeepSWE v1.1 tasks, found {len(task_dirs)}")

    rows = []
    for task_dir in task_dirs:
        relative_id = task_dir.relative_to(root).as_posix() if task_dir != root else task_dir.name
        metadata = {"instance_id": f"{task_name}/{relative_id}", "dataset": task_name, "task_path": str(task_dir)}
        instruction = (task_dir / "instruction.md").read_text(encoding="utf-8").strip()
        for harness in harnesses or [None]:
            task = {"name": "harbor", "metadata": metadata}
            if harness is not None:
                task["harbor_env"] = "e2b"
                task["agent"] = {
                    "name": harness,
                    "timeout_sec": 18000,
                    "kwargs": {"version": HARNESS_VERSIONS[harness]},
                }
            rows.append(
                {
                    "data_source": source,
                    "prompt": [{"role": "user", "content": instruction}],
                    "extra_info": {"tools_kwargs": {"task": task}},
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True, choices=("mimo", "deepswe"), help="Dataset to prepare.")
    parser.add_argument("--task-root", required=True, type=Path, help="Root directory containing Harbor tasks.")
    parser.add_argument("--output", required=True, type=Path, help="Output Parquet path.")
    parser.add_argument("--harness", type=yaml.safe_load, help="Quoted YAML list of harness names.")
    args = parser.parse_args()

    rows = load_tasks(args.task_name, args.task_root, args.harness)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(output)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
