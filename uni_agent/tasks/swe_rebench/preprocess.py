# ruff: noqa: E501
"""Preprocess SWE-rebench into the new-framework SWE task format.

Example::

    python -m uni_agent.tasks.swe_rebench.preprocess --local-save-dir ~/data/swe_agent
"""

import argparse
import os

from datasets import load_dataset


def get_image_name(instance_id: str) -> str:
    """Canonical open-source image ref for a swe-rebench instance.

    Published under the ``swerebench`` org (mirrors the modal image naming); a
    provider maps this to its own registry at run time.
    """
    return f"swerebench/sweb.eval.x86_64.{instance_id.lower().replace('__', '_1776_')}"


def build_swe_rebench(max_instances: int | None = None):
    def process(example):
        instance_id = example["instance_id"]

        install_config = example["install_config"]
        metadata = {
            "instance_id": instance_id,
            "repo": example["repo"],
            "base_commit": example["base_commit"],
            "patch": example["patch"],
            "test_patch": example["test_patch"],
            "problem_statement": example["problem_statement"],
            "FAIL_TO_PASS": example["FAIL_TO_PASS"],
            "FAIL_TO_FAIL": example["FAIL_TO_FAIL"],
            "PASS_TO_PASS": example["PASS_TO_PASS"],
            "PASS_TO_FAIL": example["PASS_TO_FAIL"],
            "install": install_config["install"],
            "log_parser": install_config["log_parser"],
            "test_cmd": install_config["test_cmd"],
        }
        task_config = {
            "name": "swe_rebench",
            "sandbox": {"image": get_image_name(instance_id)},
            "metadata": metadata,
        }

        return {
            "data_source": "nebius/SWE-rebench",
            "prompt": [{"role": "user", "content": example["problem_statement"]}],
            "extra_info": {
                "tools_kwargs": {"task": task_config},
            },
        }

    data_source = "nebius/SWE-rebench"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="filtered")
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

    dataset = build_swe_rebench(max_instances=args.max_instances)
    out_path = f"{save_dir}/swe_rebench_filtered.parquet"
    dataset.to_parquet(out_path)
    print(f"Wrote {len(dataset)} instances to {out_path}", flush=True)
