"""Run MemAgent directly on the long-context HotpotQA JSON evaluation files.

The files used by the MemAgent benchmark are raw JSON lists (``context``,
``input`` and ``answers``), not the parquet Task payload accepted by the generic
inference example.  This runner adapts one row at a time in memory, so no
converted dataset needs to be generated on disk.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from uni_agent.tasks import TaskConfigResolver, get_task
from uni_agent.tasks.hotpotqa.preprocess import DEFAULT_CONTEXT_CHUNK_SIZE, split_context_into_token_chunks

logger = logging.getLogger(__name__)


def _answers(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple):
        return [str(answer) for answer in value]
    if value is None:
        return []
    return [str(value)]


def _sample_key(sample: dict[str, Any], source_index: int) -> str:
    sample_id = sample.get("id")
    sample_index = sample.get("index", source_index)
    return f"{sample_id if sample_id is not None else '<no-id>'}:{sample_index}"


def _load_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list) or not all(isinstance(sample, dict) for sample in raw):
        raise ValueError(f"expected a JSON list of objects in {path}")
    required = {"context", "input", "answers"}
    for index, sample in enumerate(raw):
        missing = sorted(required - sample.keys())
        if missing:
            raise ValueError(f"sample {index} in {path} is missing fields: {missing}")
    return raw


def _load_latest_results(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path} at line {line_number}: {exc}") from exc
            key = record.get("sample_key")
            if isinstance(key, str):
                latest[key] = record
    return latest


def _contains_answer(response: str, answers: list[str]) -> bool:
    normalized_response = response.lower()
    return any(answer.strip() and answer.lower() in normalized_response for answer in answers)


def _public_metadata(sample: dict[str, Any], source_index: int) -> dict[str, Any]:
    ignored = {"context", "input", "answers"}
    metadata = {key: value for key, value in sample.items() if key not in ignored}
    metadata.setdefault("instance_id", sample.get("id", source_index))
    metadata["source_index"] = source_index
    return metadata


async def _run_one(
    *,
    sample: dict[str, Any],
    source_index: int,
    resolver: TaskConfigResolver,
    runtime_model: dict[str, Any],
    tokenizer: Any,
    chunk_size: int,
) -> dict[str, Any]:
    key = _sample_key(sample, source_index)
    started_at = time.perf_counter()
    answers = _answers(sample.get("answers"))
    base_record: dict[str, Any] = {
        "sample_key": key,
        "source_index": source_index,
        "id": sample.get("id"),
        "index": sample.get("index", source_index),
        "target_length": sample.get("target_length"),
        "actual_length": sample.get("actual_length"),
        "question": str(sample.get("input", "")),
        "answers": answers,
    }

    try:
        split = partial(
            split_context_into_token_chunks,
            sample.get("context"),
            tokenizer=tokenizer,
            chunk_size=chunk_size,
        )
        chunks = await asyncio.to_thread(split)
        num_chunks = len(chunks)
        if not num_chunks:
            raise ValueError("context produced zero token chunks")

        metadata = _public_metadata(sample, source_index)
        metadata["chunks"] = chunks
        sample_config = {
            "name": "hotpotqa",
            "prompt": [{"role": "user", "content": str(sample["input"])}],
            "ground_truth": answers,
            "metadata": metadata,
            # Process every chunk, including the ~210 chunks in a 1M example,
            # and reserve one final model call for the boxed answer.
            "agent": {
                "max_chunks": num_chunks,
                "max_steps": num_chunks + 1,
            },
        }
        resolved = resolver.resolve(sample_config, runtime_model=runtime_model)
        result = await get_task(resolved).run()
        info = result.extra_info or {}
        response = str(info.get("response", ""))
        reward = float(result.reward)
        base_record.update(
            {
                "num_chunks": num_chunks,
                "num_contexts": info.get("num_contexts"),
                "total_steps": info.get("total_steps"),
                "response": response,
                "reward": reward,
                "exact_match": reward >= 1.0 - 1e-12,
                "answer_contained": _contains_answer(response, answers),
                "thinking_detected": bool(info.get("thinking_detected", False)),
                "thinking_turns": int(info.get("thinking_turns", 0)),
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "error": None,
            }
        )
    except Exception as exc:  # Keep a durable row so a long benchmark is diagnosable.
        logger.exception("failed sample %s", key)
        base_record.update(
            {
                "response": "",
                "reward": 0.0,
                "exact_match": False,
                "answer_contained": False,
                "thinking_detected": False,
                "thinking_turns": 0,
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return base_record


def _write_summary(
    *,
    summary_path: Path,
    data_path: Path,
    output_path: Path,
    selected_keys: list[str],
    latest: dict[str, dict[str, Any]],
    wall_seconds: float,
) -> dict[str, Any]:
    records = [latest[key] for key in selected_keys if key in latest]
    successful = [record for record in records if not record.get("error")]
    errors = [record for record in records if record.get("error")]
    exact_matches = sum(bool(record.get("exact_match")) for record in successful)
    contained_answers = sum(bool(record.get("answer_contained")) for record in successful)
    thinking_detected = sum(bool(record.get("thinking_detected")) for record in successful)
    score = sum(float(record.get("reward", 0.0)) for record in successful) / len(successful) if successful else 0.0
    summary = {
        "data_path": str(data_path),
        "output_path": str(output_path),
        "score_metric": "mean_boxed_answer_token_lcs",
        "score": score,
        "selected": len(selected_keys),
        "completed": len(successful),
        "errors": len(errors),
        "exact_match": exact_matches,
        "exact_match_rate": exact_matches / len(successful) if successful else 0.0,
        "answer_contained": contained_answers,
        "answer_contained_rate": contained_answers / len(successful) if successful else 0.0,
        "thinking_detected": thinking_detected,
        "thinking_free_rate": 1.0 - thinking_detected / len(successful) if successful else 0.0,
        # Kept for compatibility with the first version of this runner.
        "average_reward": score,
        "wall_seconds": round(wall_seconds, 3),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary


async def _run(args: argparse.Namespace, tokenizer: Any) -> int:
    data_path = Path(args.data_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    summary_path = Path(args.summary_path).expanduser().resolve()
    samples = _load_samples(data_path)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        logger.warning("no samples selected from %s", data_path)
        return 0

    indexed_samples = list(enumerate(samples))
    selected_keys = [_sample_key(sample, source_index) for source_index, sample in indexed_samples]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        output_path.unlink(missing_ok=True)
    latest = _load_latest_results(output_path)
    pending = [
        (source_index, sample)
        for source_index, sample in indexed_samples
        if latest.get(_sample_key(sample, source_index), {}).get("error") is not None
        or _sample_key(sample, source_index) not in latest
    ]

    resolver = TaskConfigResolver.from_file(args.task_config)
    runtime_model = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model_name": args.model,
    }
    logger.info(
        "loaded=%d pending=%d resumed=%d concurrency=%d data=%s",
        len(indexed_samples),
        len(pending),
        len(indexed_samples) - len(pending),
        args.concurrency,
        data_path,
    )

    started_at = time.perf_counter()
    write_lock = asyncio.Lock()
    counter_lock = asyncio.Lock()
    completed_now = 0
    queue: asyncio.Queue[tuple[int, dict[str, Any]] | None] = asyncio.Queue()
    for work_item in pending:
        queue.put_nowait(work_item)
    worker_count = min(args.concurrency, max(1, len(pending)))
    for _ in range(worker_count):
        queue.put_nowait(None)

    async def worker() -> None:
        nonlocal completed_now
        while True:
            work_item = await queue.get()
            try:
                if work_item is None:
                    return
                source_index, sample = work_item
                record = await _run_one(
                    sample=sample,
                    source_index=source_index,
                    resolver=resolver,
                    runtime_model=runtime_model,
                    tokenizer=tokenizer,
                    chunk_size=args.chunk_size,
                )
                serialized = json.dumps(record, ensure_ascii=False)
                async with write_lock:
                    with output_path.open("a", encoding="utf-8") as handle:
                        handle.write(serialized + "\n")
                        handle.flush()
                    latest[record["sample_key"]] = record
                async with counter_lock:
                    completed_now += 1
                    status = "ERROR" if record.get("error") else "OK"
                    logger.info(
                        "[%d/%d] %s %s chunks=%s reward=%.4f elapsed=%.1fs",
                        completed_now,
                        len(pending),
                        status,
                        record["sample_key"],
                        record.get("num_chunks", "?"),
                        float(record.get("reward", 0.0)),
                        float(record.get("elapsed_seconds", 0.0)),
                    )
            finally:
                queue.task_done()

    if pending:
        workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
        await queue.join()
        await asyncio.gather(*workers)

    summary = _write_summary(
        summary_path=summary_path,
        data_path=data_path,
        output_path=output_path,
        selected_keys=selected_keys,
        latest=latest,
        wall_seconds=time.perf_counter() - started_at,
    )
    logger.info("summary: %s", json.dumps(summary, ensure_ascii=False))
    return 2 if summary["errors"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="Served model name sent to the OpenAI-compatible endpoint.")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CONTEXT_CHUNK_SIZE)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    # MemAgent logs full 5k-token prompts at INFO; benchmark progress is useful,
    # but duplicating every source context into the console and log files is not.
    logging.getLogger("uni_agent.agents.mem_agent.agent").setLevel(logging.WARNING)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    # The tokenizer only encodes the source so it can be split; no model request
    # ever contains the full 128k-1M token sequence.
    tokenizer.model_max_length = 10**12
    raise SystemExit(asyncio.run(_run(args, tokenizer)))


if __name__ == "__main__":
    main()
