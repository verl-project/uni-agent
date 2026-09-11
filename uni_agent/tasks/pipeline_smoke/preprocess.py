"""Create the tiny deterministic parquet files used by the training quickstart."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

DATA_SOURCE = "pipeline_smoke"

EXAMPLES: tuple[tuple[str, str], ...] = (
    ("Reverse the three-letter string 'lup'.", "pul"),
    ("Return the third word in: amber cobalt maple orbit.", "maple"),
    ("What is 17 minus 9? Return only the number.", "8"),
    ("Convert the word 'NOVA' to lowercase.", "nova"),
    ("Return the word between the markers: START quartz END.", "quartz"),
    ("What is 6 multiplied by 7? Return only the number.", "42"),
    ("Remove the first letter from 'plane'.", "lane"),
    ("Return the last word in: red green blue.", "blue"),
    ("What is 12 divided by 3? Return only the number.", "4"),
    ("Return the first word in: cedar silver meadow.", "cedar"),
    ("What is 5 plus 6? Return only the number.", "11"),
    ("Return the middle letter of the word 'cat'.", "a"),
    ("Reverse the three-letter string 'sun'.", "nus"),
    ("Return the second word in: alpha beta gamma.", "beta"),
    ("What is 20 minus 7? Return only the number.", "13"),
    ("Convert the word 'ORBIT' to lowercase.", "orbit"),
    ("Return the word between the markers: START willow END.", "willow"),
    ("What is 9 multiplied by 3? Return only the number.", "27"),
    ("Remove the first letter from 'stone'.", "tone"),
    ("Return the last word in: square triangle circle.", "circle"),
)

SYSTEM_PROMPT = (
    "You are completing a short deterministic training smoke-test task. "
    "Solve it yourself, then make a real call to the finish tool with only the final answer. "
    "Do not merely describe or imitate the tool call in plain text. "
    "Do not run or describe shell commands. "
    'Use the tool-call format: <tool_call>\n{"name": "finish", "arguments": {"answer": "YOUR_ANSWER"}}\n</tool_call>. '
    "Replace YOUR_ANSWER with your solution."
)


def build_rows(size: int, *, offset: int = 0) -> list[dict[str, Any]]:
    """Build deterministic task rows without downloading an external dataset."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    if offset + size > len(EXAMPLES):
        raise ValueError(
            f"requested examples [{offset}, {offset + size}) exceed the {len(EXAMPLES)} unique bundled examples"
        )
    rows: list[dict[str, Any]] = []
    for row_index in range(size):
        example_index = offset + row_index
        question, expected_answer = EXAMPLES[example_index]
        instance_id = f"pipeline-smoke-{offset + row_index:04d}"
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": prompt,
                "extra_info": {
                    "tools_kwargs": {
                        "task": {
                            "name": "pipeline_smoke",
                            "expected_answer": expected_answer,
                            "metadata": {"instance_id": instance_id},
                        }
                    }
                },
            }
        )
    return rows


def write_dataset(output_dir: str | Path, *, train_size: int, val_size: int) -> tuple[Path, Path]:
    """Write train and validation parquet files and return their paths."""
    from datasets import Dataset

    train_rows = build_rows(train_size)
    val_rows = build_rows(val_size, offset=train_size)
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    train_path = destination / "train.parquet"
    val_path = destination / "val.parquet"
    Dataset.from_list(train_rows).to_parquet(train_path)
    Dataset.from_list(val_rows).to_parquet(val_path)
    return train_path, val_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/pipeline_smoke")
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--val-size", type=int, default=4)
    args = parser.parse_args()
    train_path, val_path = write_dataset(
        args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
    )
    print(f"Wrote training data to {train_path}")
    print(f"Wrote validation data to {val_path}")


if __name__ == "__main__":
    main()
