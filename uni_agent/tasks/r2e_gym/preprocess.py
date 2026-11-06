# ruff: noqa: E501
"""Preprocess the official R2E-Gym subset into Uni-Agent Task rows.

The historical integration used the legacy ``tools_kwargs.env`` /
``tools_kwargs.reward`` schema. This preprocessor emits the current serialized
Task Config and uses the official Docker image recorded in each dataset row.

Example::

    python -m uni_agent.tasks.r2e_gym.preprocess --local-save-dir ~/data/uni_agent
"""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset

from .utils import get_gold_patch

DEFAULT_DATA_SOURCE = "dyyyyyyyy/r2e-gym-subset-filtered"


def build_r2e_gym_subset(
    max_instances: int | None = None,
    data_source: str = DEFAULT_DATA_SOURCE,
):
    """Build the training split in the Task Config format used by the framework."""

    def process(example):
        repo_name = str(example["repo_name"])
        base_commit = str(example["commit_hash"])
        instance_id = f"{repo_name}__{base_commit[:10]}"
        problem_statement = str(example["problem_statement"])
        docker_image = f"namanjain12/{repo_name}_final:{base_commit}"

        metadata = {
            "repo": repo_name,
            "instance_id": instance_id,
            "base_commit": base_commit,
            "patch": get_gold_patch(example),
            "expected_output_json": example["expected_output_json"],
            "problem_statement": problem_statement,
        }
        task_config = {
            "name": "r2e_gym",
            "sandbox": {"image": docker_image},
            "metadata": metadata,
        }
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": problem_statement}],
            "extra_info": {"tools_kwargs": {"task": task_config}},
        }

    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="train")
    print(f"Loaded {len(dataset)} raw instances", flush=True)

    if max_instances is not None and max_instances >= 0:
        dataset = dataset.select(range(min(max_instances, len(dataset))))
        print(f"Capped to {len(dataset)} instances", flush=True)

    return dataset.map(process, remove_columns=dataset.column_names)


def build_r2e_gym_verified():
    """Backward-compatible name used by the pre-refactor preprocessor."""
    return build_r2e_gym_subset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-save-dir", default="~/data/uni_agent")
    parser.add_argument("--data-source", default=DEFAULT_DATA_SOURCE)
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Optional cap on the number of instances kept (smoke testing).",
    )
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    output = build_r2e_gym_subset(max_instances=args.max_instances, data_source=args.data_source)
    out_path = f"{save_dir}/r2e_gym_subset.parquet"
    output.to_parquet(out_path)
    print(f"Wrote {len(output)} instances to {out_path}", flush=True)
