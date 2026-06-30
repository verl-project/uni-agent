# ruff: noqa: E501
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preprocess ASearcher JSON/JSONL dataset into Parquet for uni-agent training.

Adapted from asearcher/preprocess_asearcher.py for the uni-agent framework:
  - agent_name is set to "search_agent" (matching agent_config.yaml)
  - ground_truth is placed in tools_kwargs["reward"] for SearchRewardSpec
  - system prompt instructs the model to use search + finish tools

If ``--input_json`` is omitted, the filtered ASearcher dataset
(``aidenjhwu/ASearcher_en_no-math_Qwen3-8B-reject-sample``) is downloaded from
HuggingFace automatically. That dataset is derived from the original
``inclusionAI/ASearcher-train-data`` with Chinese samples and math problems
removed and reject sampling applied, and it already ships in the nested
``extra_info`` schema this script expects. To preprocess the original
(unfiltered) ASearcher data instead, write a separate preprocessing script: its
records use a different, flat schema (top-level ``question``/``answer``) that
this script does not handle.
"""

import argparse
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Filtered ASearcher dataset used by default when no local --input_json is given.
HF_FILTERED_REPO = "aidenjhwu/ASearcher_en_no-math_Qwen3-8B-reject-sample"
HF_FILTERED_FILE = "ASearcher_en_nomath_rejectsample.json"

DEFAULT_SYSTEM_CONTENT = (
    "You are an expert research assistant. Your goal is to answer the user's question by thoroughly "
    "researching it. You must follow a structured process of reasoning and tool use.\n\n"
    "# Core Instructions\n"
    "Reasoning First: Before any action, you must analyze the request, break down the problem, "
    "and plan your next steps.\n"
    "Iterative Research: The research process is iterative. You may need to use the tools multiple "
    "times to gather sufficient information.\n"
    "# Workflow\n"
    "1. Plan: Understand the user's question and formulate initial search queries.\n"
    "2. Search: Search to get a list of potential sources.\n"
    "3. Evaluate & Plan Next Step: Review the search results. If the summaries from search are "
    "sufficient to answer the question, proceed to generate the final answer. If the summaries "
    "are insufficient but some URLs look promising, use the crawler to extract in-depth "
    "information.\n"
    "4. Crawl (if necessary): Crawl the whole pages of the selected URLs.\n"
    "5. Repeat or Answer: If you still lack information, repeat the process. Otherwise, call "
    "`finish` with your final answer.\n\n"
)

DEFAULT_USER_CONTENT_PREFIX = "Question: "


def process_single_row(row, current_split_name, row_index, system_content, user_content_prefix):
    extra_info_in = row.get("extra_info", {})
    if not isinstance(extra_info_in, dict):
        extra_info_in = {}

    question = extra_info_in.get("question")
    user_content = user_content_prefix.rstrip("\n") + question
    prompt = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    raw_gt = extra_info_in.get("ground_truth")
    if isinstance(raw_gt, dict) and "target" in raw_gt:
        target = raw_gt.get("target")
    else:
        target = raw_gt

    if target is None:
        target_list = []
    elif isinstance(target, list):
        target_list = target
    else:
        target_list = [target]

    ground_truth = {"target": target_list}

    data_source_raw = extra_info_in.get("data_source") or "asearcher"
    data_source = str(data_source_raw)

    tools_kwargs = {
        "reward": {
            "ground_truth": ground_truth,
        },
    }

    extra_info = dict(extra_info_in)
    extra_info.update(
        {
            "index": row_index,
            "question": question,
            "split": current_split_name,
            "tools_kwargs": tools_kwargs,
        }
    )

    return pd.Series(
        {
            "data_source": data_source,
            "prompt": prompt,
            "reward_model": {"ground_truth": ground_truth, "style": "rule"},
            "extra_info": extra_info,
            "agent_name": "search_agent",
        }
    )


def _read_input_as_dataframe(json_path: str) -> pd.DataFrame:
    try:
        return pd.read_json(json_path)
    except ValueError:
        return pd.read_json(json_path, lines=True)


def _load_filtered_dataset() -> pd.DataFrame:
    """Download the filtered ASearcher dataset from HuggingFace as a DataFrame."""
    from datasets import load_dataset

    logger.info(f"Downloading {HF_FILTERED_REPO}:{HF_FILTERED_FILE} from HuggingFace...")
    dataset = load_dataset(HF_FILTERED_REPO, data_files=HF_FILTERED_FILE, split="train")
    return dataset.to_pandas()


def main():
    parser = argparse.ArgumentParser(description="Preprocess ASearcher JSON/JSONL dataset for uni-agent training.")
    parser.add_argument(
        "--input_json",
        default=None,
        help="Path to raw ASearcher JSON or JSONL file. If omitted, the filtered dataset "
        f"({HF_FILTERED_REPO}) is downloaded from HuggingFace.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="./asearcher_processed",
        help="Local directory to save processed Parquet files.",
    )
    parser.add_argument(
        "--train_rows",
        type=int,
        default=8192,
        help="Number of rows for the train split.",
    )
    parser.add_argument(
        "--test_rows",
        type=int,
        default=100,
        help="Number of rows for the test split.",
    )
    args = parser.parse_args()

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    if args.input_json:
        input_json_path = os.path.expanduser(args.input_json)
        df_raw = _read_input_as_dataframe(input_json_path)
        logger.info(f"Loaded {len(df_raw)} records from {input_json_path}")
    else:
        df_raw = _load_filtered_dataset()
        logger.info(f"Loaded {len(df_raw)} records from {HF_FILTERED_REPO}")

    total_needed = args.train_rows + args.test_rows
    if len(df_raw) < total_needed:
        raise ValueError(f"Not enough rows: len={len(df_raw)}, but train_rows+test_rows={total_needed}")

    system_content = DEFAULT_SYSTEM_CONTENT
    user_content_prefix = DEFAULT_USER_CONTENT_PREFIX

    df_train = df_raw.iloc[: args.train_rows].reset_index(drop=True)
    df_test = df_raw.iloc[args.train_rows : args.train_rows + args.test_rows].reset_index(drop=True)

    train_processed = df_train.apply(
        lambda row: process_single_row(row, "train", row.name, system_content, user_content_prefix),
        axis=1,
    )
    test_processed = df_test.apply(
        lambda row: process_single_row(row, "test", row.name, system_content, user_content_prefix),
        axis=1,
    )

    train_path = os.path.join(local_save_dir, "train.parquet")
    test_path = os.path.join(local_save_dir, "test.parquet")

    train_processed.to_parquet(train_path, index=False)
    test_processed.to_parquet(test_path, index=False)

    logger.info(f"Saved {len(train_processed)} train rows to {train_path}")
    logger.info(f"Saved {len(test_processed)} test rows to {test_path}")


if __name__ == "__main__":
    main()
