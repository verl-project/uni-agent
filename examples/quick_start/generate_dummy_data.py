# ruff: noqa: E501
"""Generate a tiny dummy dataset for the quick_start example.

Produces 8 identical trivial "fix the bug" samples. There is no real task,
no real reward signal, and no training benefit -- this only exercises the
data -> agent loop -> trajectory -> reward -> training path end to end.
"""

import argparse
import os

from datasets import Dataset

SYSTEM_PROMPT = """
You are a coding assistant. Use bash to explore and fix the bug. When done, call submit.
""".strip()

USER_PROMPT = """
Fix the bug: the function add(a, b) should return a + b, but it returns a - b instead. Find and fix it.
""".strip()


def build_dummy_dataset(num_samples: int = 8):
    def make_sample(index):
        # `prompt` MUST be a list of chat messages (list[dict]), not a plain
        # string -- the agent loop passes it straight to the chat template.
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            "agent_name": "swe_agent",
            "extra_info": {
                "index": index,
                "task_id": f"dummy-{index}",
                # tools_kwargs must be non-empty (Parquet cannot store an empty struct).
                "tools_kwargs": {"dummy": "placeholder"},
            },
        }

    return Dataset.from_list([make_sample(i) for i in range(num_samples)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-save-dir", default="~/data/swe_agent")
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    dataset = build_dummy_dataset(args.num_samples)
    dataset.to_parquet(os.path.join(save_dir, "dummy_agent_train.parquet"))
    print(f"Generated {len(dataset)} samples", flush=True)
