# ruff: noqa: E501
import argparse
import io
import os
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

import tomllib
from datasets import Dataset

impl = os.getenv("DEPLOYMENT", "modal").lower()
if impl != "modal":
    raise ValueError("Terminal-Bench v2 preprocessing only supports modal deployment now")


TERMINAL_BENCH_V2_GIT_URL = "https://github.com/laude-institute/terminal-bench-2.git"
TERMINAL_BENCH_V2_GIT_COMMIT = "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c"
TERMINAL_BENCH_V2_CACHE_DIR = Path("~/.cache/uni-agent/terminal-bench-2")

# TODO(terminal-bench-v2): re-enable the tasks below once Modal sandbox creation works for them.
SKIP_TASK_IDS: frozenset[str] = frozenset(
    {
        "qemu-alpine-ssh",
        "qemu-startup",
    }
)


SYSTEM_PROMPT = """
You are a helpful assistant that can interact with a computer to solve tasks.
""".strip()

USER_PROMPT = """
You are working inside a terminal benchmark environment. The task files and dependencies are already present in the container, and your commands run in the task working directory.

Task instruction:

{instruction}

Use the available tools to inspect the environment, create or modify files, run commands, and verify your work. The hidden verifier will be run after you submit. When you are confident the task is complete, call the submit tool.
""".strip()


def download_terminal_bench_v2() -> Path:
    tasks_dir = TERMINAL_BENCH_V2_CACHE_DIR.expanduser().resolve()
    tasks_dir.parent.mkdir(parents=True, exist_ok=True)
    if tasks_dir.exists():
        shutil.rmtree(tasks_dir)

    print(f"Cloning Terminal-Bench v2 into {tasks_dir}", flush=True)
    subprocess.run(["git", "clone", TERMINAL_BENCH_V2_GIT_URL, str(tasks_dir)], check=True)
    subprocess.run(["git", "-C", str(tasks_dir), "checkout", TERMINAL_BENCH_V2_GIT_COMMIT], check=True)
    return tasks_dir


def _tar_gz_bytes(source_dir: Path) -> bytes:
    """Pack ``source_dir`` contents into a deterministic gzipped tar blob.

    Entries are written relative to ``source_dir`` (so the archive does not
    include the top-level directory name), sorted for byte-stable output, and
    stripped of host-specific metadata (uid/gid/mtime).
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Expected directory, not found: {source_dir}")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(source_dir.rglob("*")):
            arcname = path.relative_to(source_dir).as_posix()
            info = tar.gettarinfo(str(path), arcname=arcname)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            if info.isfile():
                with path.open("rb") as f:
                    tar.addfile(info, f)
            else:
                tar.addfile(info)
    return buf.getvalue()


def _resolve_workdir(task_dir: Path) -> str | None:
    """Parse ``environment/Dockerfile`` to find the final ``WORKDIR`` value."""
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        return None

    workdir: str | None = None
    for raw_line in dockerfile.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("WORKDIR "):
            value = line.split(None, 1)[1].strip()
            if value.startswith("/"):
                workdir = value
            elif workdir is not None:
                workdir = str(PurePosixPath(workdir) / value)
            else:
                workdir = str(PurePosixPath("/") / value)
    return workdir


def _build_sample(task_dir: Path) -> dict:
    def _load_toml(path: Path) -> dict:
        with path.open("rb") as f:
            return tomllib.load(f)

    task_id = task_dir.name
    config = _load_toml(task_dir / "task.toml")
    instruction = (task_dir / "instruction.md").read_text()
    environment_config = config["environment"]
    verifier_config = config.get("verifier", {})
    agent_timeout_sec = float(config.get("agent", {}).get("timeout_sec", 600.0))
    verifier_timeout_sec = float(verifier_config.get("timeout_sec", 600.0))
    # runtime_timeout caps a single in-container command; cover the longer of
    # the agent (solve.sh) and verifier (test.sh) budgets so neither gets
    # truncated below task.toml's declared limits.
    runtime_timeout_sec = max(agent_timeout_sec, verifier_timeout_sec)
    env_kwargs = {
        "deployment": {
            "type": "modal",
            "image": environment_config["docker_image"],
            "startup_timeout": float(environment_config["build_timeout_sec"]),
            "runtime_timeout": runtime_timeout_sec,
            "deployment_timeout": max(3600.0, agent_timeout_sec),
            "modal_sandbox_kwargs": {
                "cpu": environment_config["cpus"],
                "memory": int(environment_config["memory"].removesuffix("G")) * 1024,
            },
        },
        "env_variables": {
            "PIP_PROGRESS_BAR": "off",
            "PIP_CACHE_DIR": "~/.cache/pip",
            "PAGER": "cat",
            "MANPAGER": "cat",
            "LESS": "-R",
            "TQDM_DISABLE": "1",
            "GIT_PAGER": "cat",
        },
        "post_setup_cmd": "",
    }
    reward_metadata = {
        "task_id": task_id,
        "task_config": config,
        "workdir": _resolve_workdir(task_dir),
        "solution_archive": _tar_gz_bytes(task_dir / "solution"),
        "tests_archive": _tar_gz_bytes(task_dir / "tests"),
        "solve_relpath": "solve.sh",
        "test_relpath": "test.sh",
    }

    # ``tools_kwargs`` is a full agent-loop config (mode 2 in
    # ``UniAgentLoop._init_config``): every field that varies per task lives
    # here, so terminal-bench parquets are self-contained. The agent-loop YAML
    # only needs to provide truly global knobs (``_target_``, ``name``,
    # ``log_dir``, ``concurrency``, ``mask_abnormal_exit_traj``, ...).
    tools_kwargs = {
        "env": env_kwargs,
        "reward": {
            "name": "terminal_bench_v2",
            "metadata": reward_metadata,
            "eval_timeout": verifier_timeout_sec,
        },
        "interaction": {
            # ``action_timeout`` caps a single in-container shell command; we
            # keep it well below ``runtime_timeout`` so the swerex layer can
            # return a clean timeout error rather than tripping a deployment-
            # level cancel.
            "action_timeout": agent_timeout_sec,
            # ``max_turns`` is a safety net; the binding constraint should be
            # ``agent.timeout_sec`` from ``task.toml``, not the turn cap.
            "max_turns": 100000,
        },
        "tools": [
            {"name": "execute_bash"},
            {"name": "str_replace_editor"},
            {"name": "submit"},
        ],
    }

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(instruction=instruction)},
        ],
        "agent_name": "swe_agent",
        "extra_info": {
            "task_id": task_id,
            "data_source": "terminal-bench-v2",
            "tools_kwargs": tools_kwargs,
        },
    }


def build_terminal_bench_v2(tasks_dir: Path) -> Dataset:
    tasks_dir = tasks_dir.expanduser().resolve()
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Terminal-Bench v2 task directory not found: {tasks_dir}")

    all_task_dirs = sorted(
        task_dir for task_dir in tasks_dir.iterdir() if task_dir.is_dir() and (task_dir / "task.toml").exists()
    )
    assert len(all_task_dirs) == 89, f"Expected 89 Terminal-Bench v2 tasks, found {len(all_task_dirs)} in {tasks_dir}"

    task_dirs = [task_dir for task_dir in all_task_dirs if task_dir.name not in SKIP_TASK_IDS]
    skipped = sorted(SKIP_TASK_IDS & {task_dir.name for task_dir in all_task_dirs})
    if skipped:
        print(f"Skipping {len(skipped)} task(s) flagged in SKIP_TASK_IDS: {skipped}", flush=True)

    samples = [_build_sample(task_dir) for task_dir in task_dirs]
    print(f"Loaded {len(samples)} Terminal-Bench v2 tasks from {tasks_dir}", flush=True)
    return Dataset.from_list(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-save-dir", default="~/data/swe_agent")

    args = parser.parse_args()

    tasks_dir = download_terminal_bench_v2()
    dataset = build_terminal_bench_v2(tasks_dir)

    output_dir = Path(args.local_save_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"terminal_bench_v2_{impl}.parquet"
    output_path = output_dir / output_name
    dataset.to_parquet(str(output_path))


if __name__ == "__main__":
    main()
