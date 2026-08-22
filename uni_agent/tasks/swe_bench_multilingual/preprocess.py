# ruff: noqa: E501
"""Preprocess SWE-bench Multilingual into the new-framework SWE task format.

The generated rows are provider-agnostic: they carry the canonical public
``swebench/sweb.eval.x86_64.<id>`` image reference and leave provider-specific
image mapping and resource configuration to the sandbox at run time.

Example::

    python -m uni_agent.tasks.swe_bench_multilingual.preprocess \
        --local-save-dir ~/data/swe_agent
"""

import argparse
import os

from datasets import load_dataset
from swebench.harness.constants import MAP_REPO_TO_EXT

EXT_TO_LANGUAGE = {
    "c": "C",
    "go": "Go",
    "java": "Java",
    "js": "JavaScript",
    "php": "PHP",
    "rb": "Ruby",
    "rs": "Rust",
}


def get_image_name(instance_id: str) -> str:
    """Return the canonical image ref, mirroring swebench's instance image key."""
    return f"swebench/sweb.eval.x86_64.{instance_id.lower().replace('__', '_1776_')}"


def build_swe_bench_multilingual(max_instances: int | None = None):
    def process(example):
        instance_id = example["instance_id"]
        repo = example["repo"]

        metadata = {
            "instance_id": instance_id,
            "repo": repo,
            "version": str(example["version"]),
            "base_commit": example["base_commit"],
            "patch": example["patch"],
            "test_patch": example["test_patch"],
            "problem_statement": example["problem_statement"],
            "language": EXT_TO_LANGUAGE.get(MAP_REPO_TO_EXT[repo], "the project's"),
            "FAIL_TO_PASS": example["FAIL_TO_PASS"],
            "PASS_TO_PASS": example["PASS_TO_PASS"],
        }
        task_config = {
            "name": "swe_bench_multilingual",
            "sandbox": {"image": get_image_name(instance_id)},
            "metadata": metadata,
        }

        return {
            "data_source": "SWE-bench/SWE-bench_Multilingual",
            "prompt": [{"role": "user", "content": example["problem_statement"]}],
            "extra_info": {
                "tools_kwargs": {"task": task_config},
            },
        }

    data_source = "SWE-bench/SWE-bench_Multilingual"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="test")
    print(f"Loaded {len(dataset)} raw instances", flush=True)

    if max_instances is not None and max_instances >= 0:
        dataset = dataset.select(range(min(max_instances, len(dataset))))
        print(f"Capped to {len(dataset)} instances", flush=True)

    dataset = dataset.map(process, remove_columns=dataset.column_names)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-save-dir", default="~/data/swe_agent")
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Optional cap on the number of instances kept (smoke testing).",
    )
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    dataset = build_swe_bench_multilingual(max_instances=args.max_instances)
    out_path = f"{save_dir}/swe_bench_multilingual.parquet"
    dataset.to_parquet(out_path)
    print(f"Wrote {len(dataset)} instances to {out_path}", flush=True)
