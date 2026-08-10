"""Download Terminal-Bench releases with Harbor and convert them into Uni-Agent parquet rows."""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import re
import shlex
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from datasets import Dataset

try:
    tomllib = import_module("tomllib")
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10
    try:
        tomllib = import_module("tomli")
    except ModuleNotFoundError:
        tomllib = None


TASK_NAME = "terminal_bench"
DEFAULT_VERSION = "2.1"
# Allow extra wall-clock time for model-serving contention during concurrent rollouts.
AGENT_TIMEOUT_MULTIPLIER = 4.0


@dataclass(frozen=True)
class BenchmarkSpec:
    version: str
    dataset: str
    harbor_ref: str
    expected_tasks: int


BENCHMARKS = {
    "2.0": BenchmarkSpec(
        version="2.0",
        dataset="terminal-bench/terminal-bench-2",
        harbor_ref="terminal-bench@2.0",
        expected_tasks=89,
    ),
    "2.1": BenchmarkSpec(
        version="2.1",
        dataset="terminal-bench/terminal-bench-2-1",
        harbor_ref=(
            "terminal-bench/terminal-bench-2-1@sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"
        ),
        expected_tasks=89,
    ),
}


SYSTEM_PROMPT = """
You are a helpful assistant that can interact with a computer to solve tasks.
""".strip()

USER_PROMPT = """
You are working in a Linux environment.

Complete the task below using the available tools.
Inspect the environment, create or modify files, run commands, and verify your work as needed.

Task instruction:

{instruction}
""".strip()


def _load_toml(path: Path) -> dict[str, Any]:
    if tomllib is None:
        raise RuntimeError("Python 3.10 requires `pip install tomli` to preprocess Terminal-Bench")
    with path.open("rb") as file:
        return tomllib.load(file)


def parse_dockerfile_workdir(dockerfile: Path) -> str | None:
    """Return the final literal WORKDIR declared by a Dockerfile."""
    _FROM_RE = re.compile(r"^\s*FROM(?:\s|$)", re.IGNORECASE)
    _WORKDIR_RE = re.compile(r"^\s*WORKDIR\s+(.+?)\s*$", re.IGNORECASE)

    if not dockerfile.is_file():
        return None

    workdir: str | None = None
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        if _FROM_RE.match(line):
            workdir = None
            continue
        match = _WORKDIR_RE.match(line)
        if not match:
            continue
        parts = shlex.split(match.group(1), comments=True)
        if parts:
            workdir = parts[0]

    return workdir


def pack_directory_base64(source_dir: Path) -> str:
    """Create a deterministic base64 tarball while preserving executable bits."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"required task directory does not exist: {source_dir}")

    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", dereference=False) as archive:
        paths = [source_dir, *sorted(source_dir.rglob("*"), key=lambda path: path.as_posix())]
        for path in paths:
            arcname = "." if path == source_dir else path.relative_to(source_dir).as_posix()
            info = archive.gettarinfo(str(path), arcname=arcname)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            if info.isfile():
                with path.open("rb") as file:
                    archive.addfile(info, file)
            else:
                archive.addfile(info)

    return base64.b64encode(gzip.compress(raw.getvalue(), mtime=0)).decode("ascii")


def _load_task(task_dir: Path) -> tuple[str, dict[str, Any]]:
    config = _load_toml(task_dir / "task.toml")
    name = config.get("task", {}).get("name") or f"terminal-bench/{task_dir.name}"
    return name, config


def _size_mb(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    units = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
    return int(float(value[:-1]) * units[value[-1].upper()])


def discover_task_dirs(dataset_dir: Path | str, *, expected_task_count: int | None = None) -> list[Path]:
    """Return task directories exported under a dataset root."""
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Terminal-Bench dataset directory does not exist: {root}")

    task_dirs = sorted(path for path in root.iterdir() if path.is_dir() and (path / "task.toml").is_file())

    if expected_task_count is not None and len(task_dirs) != expected_task_count:
        raise ValueError(f"expected {expected_task_count} Terminal-Bench tasks, found {len(task_dirs)} under {root}")
    return task_dirs


def build_task_row(
    task_dir: Path,
    *,
    benchmark: BenchmarkSpec = BENCHMARKS[DEFAULT_VERSION],
) -> dict[str, Any]:
    """Build one provider-agnostic Uni-Agent sample."""
    task_name, config = _load_task(task_dir)
    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8").strip()
    environment = config["environment"]
    verifier = config["verifier"]
    agent = config["agent"]
    solution = config.get("solution", {})
    workdir = environment.get("workdir") or parse_dockerfile_workdir(task_dir / "environment" / "Dockerfile")

    agent_timeout = float(agent["timeout_sec"]) * AGENT_TIMEOUT_MULTIPLIER
    verifier_timeout = float(verifier["timeout_sec"])
    runtime_timeout = agent_timeout + verifier_timeout + 600.0
    memory = environment["memory_mb"] if "memory_mb" in environment else environment["memory"]
    storage = environment["storage_mb"] if "storage_mb" in environment else environment["storage"]

    environment_spec = {
        "workdir": workdir,
        "cpus": environment["cpus"],
        "memory_mb": _size_mb(memory),
        "storage_mb": _size_mb(storage),
        "gpus": environment.get("gpus", 0),
        "gpu_types": environment.get("gpu_types", []),
        "allow_internet": environment.get("allow_internet", True),
        "env": environment.get("env", {}),
    }

    solution_dir = task_dir / "solution"
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(instruction=instruction)},
    ]
    metadata = {
        "instance_id": task_name,
        "task_id": task_name,
        "dataset": benchmark.dataset,
        "dataset_version": benchmark.version,
        "dataset_ref": benchmark.harbor_ref,
        "agent_timeout": agent_timeout,
        "verifier_timeout": verifier_timeout,
        "environment_json": json.dumps(environment_spec, sort_keys=True),
        "verifier_env_json": json.dumps(verifier.get("env", {}), sort_keys=True),
        "solution_env_json": json.dumps(solution.get("env", {}), sort_keys=True),
        "tests_archive": pack_directory_base64(task_dir / "tests"),
        "solution_archive": pack_directory_base64(solution_dir),
    }
    task_config = {
        "name": TASK_NAME,
        "sandbox": {
            "image": environment["docker_image"],
            "runtime_timeout": runtime_timeout,
        },
        "prompt": prompt,
        "metadata": metadata,
    }
    return {
        "data_source": benchmark.dataset,
        "prompt": prompt,
        "extra_info": {"tools_kwargs": {"task": task_config}},
    }


def _download_benchmark(benchmark: BenchmarkSpec, output_dir: Path) -> Path:
    ref = benchmark.harbor_ref
    print(f"Downloading Terminal-Bench {benchmark.version} ({ref})", flush=True)
    try:
        subprocess.run(
            [
                "harbor",
                "download",
                ref,
                "--export",
                "--output-dir",
                str(output_dir),
            ],
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Terminal-Bench preprocessing requires the Harbor CLI; install it with `pip install harbor`"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"failed to download {ref}") from exc

    dataset_name = ref.split("@", 1)[0].rsplit("/", 1)[-1]
    dataset_dir = output_dir / dataset_name
    if not dataset_dir.is_dir():
        raise RuntimeError(f"dataset was not exported to {dataset_dir}")
    return dataset_dir


def build_terminal_bench(
    version: str = DEFAULT_VERSION,
    *,
    max_instances: int | None = None,
):
    """Build a Hugging Face Dataset for a configured Terminal-Bench release."""
    benchmark = BENCHMARKS[version]

    with tempfile.TemporaryDirectory(prefix=f"terminal-bench-{version}-") as temp_dir:
        dataset_dir = _download_benchmark(benchmark, Path(temp_dir))
        task_dirs = discover_task_dirs(dataset_dir, expected_task_count=benchmark.expected_tasks)

        if max_instances is not None and max_instances >= 0:
            task_dirs = task_dirs[:max_instances]

        rows = []
        for index, task_dir in enumerate(task_dirs, start=1):
            print(f"[{index}/{len(task_dirs)}] packing {task_dir.name}", flush=True)
            rows.append(build_task_row(task_dir, benchmark=benchmark))

    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Terminal-Bench for Uni-Agent.")
    parser.add_argument("--version", choices=sorted(BENCHMARKS), default=DEFAULT_VERSION)
    parser.add_argument("--local-save-dir", default="~/data/uni_agent")
    parser.add_argument("--max-instances", type=int, default=None, help="Keep only the first N tasks.")
    args = parser.parse_args()

    save_dir = Path(args.local_save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_terminal_bench(
        args.version,
        max_instances=args.max_instances,
    )
    version_slug = args.version.replace(".", "_")
    output_path = save_dir / f"terminal_bench_{version_slug}.parquet"
    dataset.to_parquet(output_path)
    print(f"Wrote {len(dataset)} instances to {output_path}", flush=True)


if __name__ == "__main__":
    main()
