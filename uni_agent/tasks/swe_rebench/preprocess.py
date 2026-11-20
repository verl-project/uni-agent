# ruff: noqa: E501
"""Preprocess SWE-rebench into the new-framework SWE task format.

Example::

    python -m uni_agent.tasks.swe_rebench.preprocess --local-save-dir ~/data/swe_agent

Offline, from a downloaded copy of the ``filtered`` split, keeping selected instances::

    python -m uni_agent.tasks.swe_rebench.preprocess \
        --data-files filtered-00000-of-00001.parquet \
        --instance-ids 0b01001001__spectree-64,owner__repo-1 \
        --output-name swe_rebench_smoke.parquet
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


def build_swe_rebench(
    max_instances: int | None = None,
    data_files: list[str] | None = None,
    instance_ids: list[str] | None = None,
):
    if data_files is not None and not data_files:
        raise ValueError("data_files must contain at least one parquet file")
    if instance_ids is not None and (not instance_ids or any(not item.strip() for item in instance_ids)):
        raise ValueError("instance_ids must contain non-empty instance ids")

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
    if data_files is not None:
        print(f"Loading the {data_source} filtered split from {data_files}...", flush=True)
        dataset = load_dataset("parquet", data_files=data_files, split="train")
    else:
        print(f"Loading the {data_source} dataset from huggingface...", flush=True)
        dataset = load_dataset(data_source, split="filtered")
    print(f"Loaded {len(dataset)} raw instances", flush=True)

    if instance_ids is not None:
        wanted = list(dict.fromkeys(instance_ids))
        wanted_set = set(wanted)
        positions = {}
        for index, instance_id in enumerate(dataset["instance_id"]):
            if instance_id in wanted_set:
                if instance_id in positions:
                    raise ValueError(f"duplicate instance id in the dataset: {instance_id}")
                positions[instance_id] = index
        missing = [instance_id for instance_id in wanted if instance_id not in positions]
        if missing:
            raise ValueError(f"instance ids not found in the dataset: {missing}")
        dataset = dataset.select([positions[instance_id] for instance_id in wanted])
        print(f"Selected {len(dataset)} requested instances", flush=True)

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
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=None,
        help="Local parquet file(s) of the filtered split; skips the Hugging Face download.",
    )
    parser.add_argument(
        "--instance-ids",
        default=None,
        help="Comma-separated instance ids to keep, in this order.",
    )
    parser.add_argument("--output-name", default="swe_rebench_filtered.parquet")
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    instance_ids = (
        [item.strip() for item in args.instance_ids.split(",") if item.strip()]
        if args.instance_ids is not None
        else None
    )
    dataset = build_swe_rebench(
        max_instances=args.max_instances,
        data_files=[os.path.expanduser(path) for path in args.data_files] if args.data_files else None,
        instance_ids=instance_ids,
    )
    out_path = f"{save_dir}/{args.output_name}"
    dataset.to_parquet(out_path)
    print(f"Wrote {len(dataset)} instances to {out_path}", flush=True)
