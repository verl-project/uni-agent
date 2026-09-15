"""Runtime dataset adapter for MemAgent training on raw HotpotQA parquet files."""

from __future__ import annotations

from typing import Any

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from uni_agent.tasks.hotpotqa.preprocess import (
    DEFAULT_CONTEXT_CHUNK_SIZE,
    split_context_into_token_chunks,
)
from verl.utils.dataset.rl_dataset import RLHFDataset

_REQUIRED_COLUMNS = ("data_source", "prompt", "context", "reward_model", "extra_info")


def _ground_truths(reward_model: Any) -> list[str]:
    """Normalize the raw parquet reward-model payload into accepted answers."""

    if not isinstance(reward_model, dict):
        return []
    ground_truth = reward_model.get("ground_truth", [])
    if isinstance(ground_truth, str):
        return [ground_truth]
    if isinstance(ground_truth, list | tuple):
        return [str(answer) for answer in ground_truth]
    if ground_truth is None:
        return []
    return [str(ground_truth)]


def build_task_config(row: dict[str, Any], *, tokenizer: Any, chunk_size: int) -> dict[str, Any]:
    """Build the serialized HotpotQA Task expected by ``run_task``."""

    prompt = row.get("raw_prompt") or row.get("prompt") or []
    source_extra_info = row.get("extra_info")
    metadata = dict(source_extra_info) if isinstance(source_extra_info, dict) else {}
    metadata["chunks"] = split_context_into_token_chunks(
        row.get("context"),
        tokenizer=tokenizer,
        chunk_size=chunk_size,
    )
    return {
        "name": "hotpotqa",
        "prompt": prompt,
        "ground_truth": _ground_truths(row.get("reward_model")),
        "metadata": metadata,
    }


class HotpotQAMemAgentDataset(RLHFDataset):
    """Read the raw 32k parquet safely and adapt each row at access time.

    Hugging Face ``load_dataset('parquet')`` fails on the source file's unused
    nested score columns.  Reading only the five training columns with PyArrow
    avoids that conversion path and also avoids producing a converted parquet.
    """

    def _read_files_and_tokenize(self) -> None:
        dataframes = []
        for parquet_file in self.data_files:
            if not str(parquet_file).endswith(".parquet"):
                raise ValueError(f"HotpotQAMemAgentDataset only supports parquet files: {parquet_file}")

            parquet_schema = pq.ParquetFile(parquet_file).schema_arrow
            missing = sorted(set(_REQUIRED_COLUMNS) - set(parquet_schema.names))
            if missing:
                raise ValueError(f"HotpotQA parquet is missing required columns {missing}: {parquet_file}")

            table = pq.read_table(parquet_file, columns=list(_REQUIRED_COLUMNS), memory_map=True)
            # pyarrow's `string` uses 32-bit offsets and overflows when
            # concatenating many long contexts; cast to `large_string`.
            if any(pa.types.is_string(f.type) for f in table.schema):
                table = table.cast(
                    pa.schema(
                        [
                            pa.field(
                                f.name,
                                pa.large_string() if pa.types.is_string(f.type) else f.type,
                                f.nullable,
                            )
                            for f in table.schema
                        ]
                    )
                )
            dataframes.append(datasets.Dataset(table))

        self.dataframe = datasets.concatenate_datasets(dataframes)
        total = len(self.dataframe)
        print(f"dataset len: {total}")

        if 0 < self.max_samples < total:
            if self.shuffle:
                rng = np.random.default_rng(self.seed)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = super().__getitem__(item)
        chunk_size = int(self.config.get("context_chunk_size", DEFAULT_CONTEXT_CHUNK_SIZE))

        tools_kwargs = dict(row.get("tools_kwargs") or {})
        tools_kwargs["task"] = build_task_config(row, tokenizer=self.tokenizer, chunk_size=chunk_size)
        row["tools_kwargs"] = tools_kwargs

        # The Task payload owns these values now.  In particular, never send the
        # original 100k-character context through the rollout path a second time.
        row.pop("context", None)
        return row
