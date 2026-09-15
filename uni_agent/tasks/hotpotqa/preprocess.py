"""Preprocess HotpotQA into the Task format consumed by Uni-Agent.

Example::

    python -m uni_agent.tasks.hotpotqa.preprocess \
        --tokenizer-path Qwen/Qwen3-8B \
        --local-save-dir ~/data/uni_agent
"""

from __future__ import annotations

import argparse
import os
from typing import Any

DATA_SOURCE = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
DEFAULT_CONTEXT_CHUNK_SIZE = 5_000


def context_to_text(context: Any) -> str:
    """Normalize a HotpotQA context, while retaining document titles."""

    if context is None:
        return ""
    if isinstance(context, str):
        return context
    if isinstance(context, dict):
        titles = context.get("title")
        sentences = context.get("sentences")
        if isinstance(titles, list) and isinstance(sentences, list) and len(titles) == len(sentences):
            sections = []
            for title, document_sentences in zip(titles, sentences, strict=True):
                document = context_to_text(document_sentences)
                sections.append(f"{title}\n{document}" if document else str(title))
            return "\n\n".join(sections)
        return "\n".join(f"{key}: {context_to_text(value)}" for key, value in context.items())
    if isinstance(context, list | tuple):
        return "\n".join(part for value in context if (part := context_to_text(value)))
    return str(context)


def split_context_into_token_chunks(context: Any, *, tokenizer: Any, chunk_size: int) -> list[str]:
    """Split a context into decoded, token-bounded chunks."""

    if chunk_size <= 0:
        raise ValueError(f"context_chunk_size must be positive, got {chunk_size}")
    context_ids = tokenizer.encode(context_to_text(context), add_special_tokens=False)
    return [
        tokenizer.decode(context_ids[offset : offset + chunk_size], skip_special_tokens=True)
        for offset in range(0, len(context_ids), chunk_size)
    ]


def process_example(example: dict[str, Any], *, tokenizer: Any, chunk_size: int) -> dict[str, Any]:
    """Convert one canonical HotpotQA example to a serialized Task Config."""

    question = str(example["question"])
    answer = str(example["answer"])
    prompt = [{"role": "user", "content": question}]
    metadata = {
        "instance_id": example.get("id"),
        "type": example.get("type"),
        "level": example.get("level"),
        "chunks": split_context_into_token_chunks(
            example.get("context"),
            tokenizer=tokenizer,
            chunk_size=chunk_size,
        ),
    }
    task_config = {
        "name": "hotpotqa",
        "prompt": prompt,
        "ground_truth": [answer],
        "metadata": metadata,
    }
    return {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "extra_info": {"tools_kwargs": {"task": task_config}},
    }


def build_hotpotqa(
    *,
    tokenizer_path: str,
    split: str,
    context_chunk_size: int = DEFAULT_CONTEXT_CHUNK_SIZE,
    max_instances: int | None = None,
):
    """Load and preprocess one HotpotQA split."""

    from datasets import load_dataset
    from transformers import AutoTokenizer

    if context_chunk_size <= 0:
        raise ValueError(f"context_chunk_size must be positive, got {context_chunk_size}")

    print(f"Loading {DATA_SOURCE}/{DATASET_CONFIG} split={split}...", flush=True)
    dataset = load_dataset(DATA_SOURCE, DATASET_CONFIG, split=split)
    print(f"Loaded {len(dataset)} raw instances", flush=True)

    if max_instances is not None and max_instances >= 0:
        dataset = dataset.select(range(min(max_instances, len(dataset))))
        print(f"Capped to {len(dataset)} instances", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return dataset.map(
        lambda example: process_example(example, tokenizer=tokenizer, chunk_size=context_chunk_size),
        remove_columns=dataset.column_names,
        desc=f"Preprocessing HotpotQA {split}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--local-save-dir", default="~/data/uni_agent")
    parser.add_argument("--context-chunk-size", type=int, default=DEFAULT_CONTEXT_CHUNK_SIZE)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Optional cap applied independently to every requested split.",
    )
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    for requested_split in args.splits:
        processed = build_hotpotqa(
            tokenizer_path=args.tokenizer_path,
            split=requested_split,
            context_chunk_size=args.context_chunk_size,
            max_instances=args.max_instances,
        )
        output_split = "dev" if requested_split == "validation" else requested_split
        out_path = os.path.join(save_dir, f"hotpotqa_{output_split}.parquet")
        processed.to_parquet(out_path)
        print(f"Wrote {len(processed)} instances to {out_path}", flush=True)
