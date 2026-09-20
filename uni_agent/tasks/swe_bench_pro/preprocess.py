"""Preprocess SWE-bench Pro into Uni-Agent's native task format."""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
from typing import Any

from datasets import DownloadManager, load_dataset

DATA_SOURCE = "ScaleAI/SWE-bench_Pro"
RUN_SCRIPTS_ARCHIVE = "https://github.com/scaleapi/SWE-bench_Pro-os/archive/refs/heads/main.tar.gz"


def _parse_string_list(value: object, field_name: str) -> list[str]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a serialized list of strings") from exc
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return value


def _resolve_run_scripts_dir(scripts_dir: str | None) -> Path:
    if scripts_dir:
        run_scripts_dir = Path(scripts_dir).expanduser()
    else:
        archive = Path(DownloadManager().download_and_extract(RUN_SCRIPTS_ARCHIVE))
        candidates = [archive / "run_scripts", *archive.glob("*/run_scripts")]
        run_scripts_dir = next((path for path in candidates if path.is_dir()), archive / "run_scripts")

    if not run_scripts_dir.is_dir():
        raise FileNotFoundError(f"SWE-bench Pro run_scripts directory not found: {run_scripts_dir}")
    return run_scripts_dir


def build_swe_bench_pro(max_instances: int | None = None, *, scripts_dir: str | None = None):
    """Load and convert the public SWE-bench Pro test split."""
    run_scripts_dir = _resolve_run_scripts_dir(scripts_dir)

    def process(example: dict[str, Any]) -> dict[str, Any]:
        requirements = example.get("requirements") or ""
        interface = example.get("interface") or ""
        problem_statement = (
            f"{example['problem_statement']}\n\n"
            f"Requirements:\n{requirements}\n\n"
            f"New interfaces introduced:\n{interface}"
        )
        fail_to_pass = _parse_string_list(example["fail_to_pass"], "fail_to_pass")
        pass_to_pass = _parse_string_list(example["pass_to_pass"], "pass_to_pass")
        selected_tests = _parse_string_list(
            example["selected_test_files_to_run"],
            "selected_test_files_to_run",
        )
        instance_scripts_dir = run_scripts_dir / example["instance_id"]

        metadata = {
            "repo": example["repo"],
            "instance_id": example["instance_id"],
            "base_commit": example["base_commit"],
            "patch": example["patch"],
            "test_patch": example["test_patch"],
            "problem_statement": problem_statement,
            "repo_language": example["repo_language"],
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
            "selected_test_files_to_run": ",".join(selected_tests),
            "run_script": (instance_scripts_dir / "run_script.sh").read_text(encoding="utf-8"),
            "parser": (instance_scripts_dir / "parser.py").read_text(encoding="utf-8"),
        }
        task_config = {
            "name": "swe_bench_pro",
            "sandbox": {"image": f"jefzda/sweap-images:{example['dockerhub_tag']}"},
            "metadata": metadata,
        }
        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": problem_statement}],
            "extra_info": {"tools_kwargs": {"task": task_config}},
        }

    print(f"Loading {DATA_SOURCE} from Hugging Face...", flush=True)
    dataset = load_dataset(DATA_SOURCE, split="test")
    print(f"Loaded {len(dataset)} raw instances", flush=True)

    if max_instances is not None and max_instances >= 0:
        dataset = dataset.select(range(min(max_instances, len(dataset))))
        print(f"Capped to {len(dataset)} instances", flush=True)

    return dataset.map(process, remove_columns=dataset.column_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SWE-bench Pro for Uni-Agent.")
    parser.add_argument("--local-save-dir", default="~/data/swe_agent")
    parser.add_argument("--scripts-dir", default=None, help="Optional local official run_scripts directory.")
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Optional cap on the number of instances kept (smoke testing).",
    )
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    dataset = build_swe_bench_pro(max_instances=args.max_instances, scripts_dir=args.scripts_dir)
    output_path = os.path.join(save_dir, "swe_bench_pro.parquet")
    dataset.to_parquet(output_path)
    print(f"Wrote {len(dataset)} instances to {output_path}", flush=True)
