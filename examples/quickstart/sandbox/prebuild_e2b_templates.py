"""Build the E2B templates a dataset needs before running it.

With ``provider: e2b`` and an ``image``, the E2B sandbox builds a template from
that image the first time it starts one. For datasets with one image per
sample (SWE-bench and similar), run this script first so rollouts start from
ready templates instead of waiting for builds:

    python examples/quickstart/sandbox/prebuild_e2b_templates.py \\
        --data-path ~/data/uni_agent/swe_bench_verified.parquet \\
        --task-config examples/quickstart/oracle/task_config_oracle.yaml \\
        --limit 3

Each sample is resolved exactly as the inference and oracle runners resolve it
(Task Config defaults, ``sandbox.image_map``, ``sandbox_kwargs``), so the
templates built here are the ones ``E2BSandbox.start()`` looks up.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from uni_agent.sandbox import SandboxConfig
from uni_agent.sandbox.e2b import E2BSandbox, ensure_template
from uni_agent.tasks import TaskConfigResolver


def collect_templates(
    samples: list[dict[str, Any]], resolver: TaskConfigResolver
) -> list[tuple[str, int, int, str | None]]:
    """Distinct ``(image, cpu_count, memory_mb, proxy)`` the E2B sandboxes of ``samples`` will need."""
    needed: dict[tuple[str, int, int, str | None], None] = {}
    for sample in samples:
        task_config = resolver.resolve(sample["extra_info"]["tools_kwargs"]["task"])
        sandbox_config = SandboxConfig.model_validate({**task_config["sandbox"], "provider": "e2b"})
        sandbox = E2BSandbox.from_config(sandbox_config)
        if sandbox.template is None and sandbox.image is not None:
            needed[(sandbox.image, sandbox.cpu_count, sandbox.memory_mb, sandbox.proxy)] = None
    return list(needed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-path", required=True, help="Preprocessed parquet file.")
    parser.add_argument("--task-config", default=None, help="Run-level Task Config YAML used for the run.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N samples, as in the runners.")
    parser.add_argument("--concurrency", type=int, default=4, help="Templates built in parallel.")
    args = parser.parse_args()

    from datasets import load_dataset

    samples = load_dataset("parquet", data_files=args.data_path, split="train").to_list()[: args.limit]
    resolver = TaskConfigResolver.from_file(args.task_config) if args.task_config else TaskConfigResolver()
    templates = collect_templates(samples, resolver)
    print(f"{len(samples)} samples need {len(templates)} E2B templates", flush=True)

    def build(spec: tuple[str, int, int, str | None]) -> tuple[str, float]:
        begin = time.monotonic()
        return ensure_template(*spec), time.monotonic() - begin

    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {pool.submit(build, spec): spec for spec in templates}
        for future in as_completed(futures):
            image = futures[future][0]
            try:
                name, seconds = future.result()
                print(f"ready   {name}  ({image}, {seconds:.0f}s)", flush=True)
            except Exception as exc:
                failed += 1
                print(f"failed  {image}: {type(exc).__name__}: {exc}", flush=True)
    print(f"{len(templates) - failed} ready, {failed} failed", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
