# ruff: noqa: E501
import argparse
import asyncio
import json
import logging
import os
import platform
import re
import shlex
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uni_agent.async_logging import add_file_handler, cleanup_handlers
from uni_agent.interaction import AgentEnv, AgentEnvConfig
from uni_agent.reward import load_reward_spec

logger = logging.getLogger(__file__)
logger.setLevel("INFO")

DEFAULT_LOCAL_COMMAND = (
    "python3 -m pip install -q swe-rex && "
    "python3 -m swerex.server --host 0.0.0.0 --port {port} --auth-token {token}"
)
UNI_AGENT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = UNI_AGENT_ROOT.parent
DEFAULT_RESULTS_DIR = WORKSPACE_ROOT / "data" / "swe_agent" / "gold_patch_verification_logs"
DEFAULT_LOCAL_SIF_IMAGE_DIR = WORKSPACE_ROOT / "data" / "swe_agent" / "arch_specific_images" / "images"
DEFAULT_CACHE_DIR = str(WORKSPACE_ROOT / "tmp" / "uni-agent-hf-cache")


def _host_swe_image_arch() -> str:
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        return "arm64"
    return "x86_64"


def _safe_sif_name(image: str) -> str:
    image = image.replace(":", "_").replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", image).strip("_")


def _resolve_local_sif(image: str, image_dir: str | os.PathLike[str] | None) -> str | None:
    if image_dir is None:
        return None
    image_dir_text = str(image_dir)
    if image_dir_text.lower() in {"", "none", "false", "off"}:
        return None
    sif_path = Path(image_dir_text).expanduser() / f"{_safe_sif_name(image)}.sif"
    if sif_path.is_file():
        return str(sif_path)
    return None


def resolve_local_image(
    image: str,
    image_arch: str | None = None,
    image_dir: str | os.PathLike[str] | None = None,
) -> str:
    image_arch = (image_arch or os.getenv("LOCAL_DEPLOYMENT_IMAGE_ARCH", "auto")).lower()
    if image_arch in {"keep", "none", "false"}:
        return image
    if image_arch == "auto":
        image_arch = _host_swe_image_arch()
    if image_arch not in {"x86_64", "arm64"}:
        raise ValueError("LOCAL_DEPLOYMENT_IMAGE_ARCH must be one of: auto, keep, x86_64, arm64")
    resolved_image = image.replace("sweb.eval.x86_64.", f"sweb.eval.{image_arch}.")
    if image_dir is None:
        image_dir = os.getenv("LOCAL_DEPLOYMENT_IMAGE_DIR", str(DEFAULT_LOCAL_SIF_IMAGE_DIR))
    return _resolve_local_sif(resolved_image, image_dir) or resolved_image


def build_deployment_config(impl: str, instance_env: dict) -> dict:
    if impl == "local":
        startup_timeout = os.getenv("LOCAL_DEPLOYMENT_STARTUP_TIMEOUT")
        deployment_config = {
            "type": "local",
            "image": resolve_local_image(instance_env["image"]),
            "command": os.getenv("LOCAL_DEPLOYMENT_COMMAND", DEFAULT_LOCAL_COMMAND),
            "timeout": 600.0,
            "startup_timeout": float(startup_timeout) if startup_timeout else 180.0,
        }
        local_runtime = os.getenv("LOCAL_CONTAINER_RUNTIME")
        local_network = os.getenv("LOCAL_DEPLOYMENT_NETWORK")
        local_host = os.getenv("LOCAL_DEPLOYMENT_HOST")
        local_port = os.getenv("LOCAL_DEPLOYMENT_PORT")
        local_extra_args = os.getenv("LOCAL_DEPLOYMENT_EXTRA_ARGS")
        if local_runtime:
            deployment_config["container_runtime"] = local_runtime
        if local_network:
            deployment_config["network"] = local_network
        if local_host:
            deployment_config["host"] = local_host
        if local_port:
            deployment_config["published_port"] = int(local_port)
        if local_extra_args:
            deployment_config["extra_run_args"] = shlex.split(local_extra_args)
        return deployment_config

    if impl == "vefaas":
        return {
            "type": "vefaas",
            "image": instance_env["image"],
            "command": "curl -fsSL https://vefaas-swe.tos-cn-beijing.ivolces.com/swe-rex/install_1.4.0.sh | bash -s -- {token}",
            "timeout": 600.0,
            "startup_timeout": 180.0,
            "function_id": os.getenv("VEFAAS_FUNCTION_ID"),
            "function_route": os.getenv("VEFAAS_FUNCTION_ROUTE"),
        }

    if impl == "modal":
        return {
            "type": "modal",
            "image": instance_env["image"],
            "startup_timeout": 600.0,
            "runtime_timeout": 600.0,
            "deployment_timeout": 3600.0,
        }

    if impl == "":
        raise ValueError("DEPLOYMENT must be set")
    raise ValueError(f"Invalid environment implementation: {impl}")


def resolve_execution_defaults(impl: str, num_workers: int | None, worker_concurrency: int | None) -> tuple[int, int]:
    if num_workers is None:
        num_workers = 1 if impl == "local" else 8
    if worker_concurrency is None:
        worker_concurrency = 1 if impl == "local" else 8
    if num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    if worker_concurrency < 1:
        raise ValueError("--worker-concurrency must be >= 1")
    return num_workers, worker_concurrency


def resolve_start_max_retries(impl: str, start_max_retries: int | None) -> int:
    if start_max_retries is None:
        env_value = os.getenv("LOCAL_DEPLOYMENT_MAX_RETRIES" if impl == "local" else "DEPLOYMENT_MAX_RETRIES")
        start_max_retries = int(env_value) if env_value else (1 if impl == "local" else 5)
    if start_max_retries < 1:
        raise ValueError("--start-max-retries must be >= 1")
    return start_max_retries


def load_samples(data_path: str, cache_dir: str | None, dataset_split: str, max_samples: int) -> list[dict]:
    data_path = os.path.expanduser(data_path)
    cache_dir = os.path.expanduser(cache_dir) if cache_dir else None
    logger.info(f"Loading dataset from: {data_path}")
    dataset = load_dataset("parquet", data_files={dataset_split: data_path}, split=dataset_split, cache_dir=cache_dir)
    samples = dataset.to_list()
    if max_samples > 0:
        samples = samples[:max_samples]
        logger.info(f"Using first {len(samples)} samples (--max-samples={max_samples})")
    return samples


def instance_metadata(sample: dict) -> dict:
    return sample["extra_info"]["tools_kwargs"]["reward"]["metadata"]


def instance_name(sample: dict) -> str:
    return instance_metadata(sample)["instance_id"]


def classify_result(result: dict) -> str:
    if result.get("resolved"):
        return "success"
    if result.get("eval_completed"):
        return "fail_wa"
    return "fail_tle_or_error"


def create_result_run_dir(results_dir: str, run_name: str | None = None) -> Path:
    root = Path(results_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    if run_name is None:
        run_name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (root / "latest_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    return run_dir


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _summary_is_compatible(summary: dict, *, data_path: str, deployment: str, args: argparse.Namespace) -> bool:
    return (
        summary.get("data_path") == os.path.expanduser(data_path)
        and summary.get("deployment") == deployment
        and summary.get("dataset_split") == args.dataset_split
        and summary.get("max_samples") == args.max_samples
    )


def _load_summary(run_dir: Path) -> dict | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _latest_run_dir(results_dir: str) -> Path | None:
    latest_path = Path(results_dir).expanduser() / "latest_run.txt"
    if not latest_path.is_file():
        return None
    latest = latest_path.read_text(encoding="utf-8").strip()
    if not latest:
        return None
    return Path(latest).expanduser()


def resolve_result_run_dir(
    *,
    results_dir: str,
    run_name: str | None,
    auto_resume: bool,
    data_path: str,
    deployment: str,
    args: argparse.Namespace,
) -> tuple[Path, list[dict], float]:
    if run_name:
        return create_result_run_dir(results_dir, run_name), [], 0.0

    if auto_resume:
        latest_run_dir = _latest_run_dir(results_dir)
        if latest_run_dir and latest_run_dir.is_dir():
            summary = _load_summary(latest_run_dir)
            if summary and _summary_is_compatible(summary, data_path=data_path, deployment=deployment, args=args):
                total_samples = int(summary.get("total_samples") or 0)
                all_num = int(summary.get("all_num") or 0)
                if total_samples and all_num < total_samples:
                    existing_records = read_jsonl(latest_run_dir / "results.jsonl")
                    logger.info(
                        "Resuming incomplete verification run %s (%s/%s records).",
                        latest_run_dir,
                        len(existing_records),
                        total_samples,
                    )
                    return latest_run_dir, existing_records, float(summary.get("execution_time") or 0.0)

    return create_result_run_dir(results_dir), [], 0.0


def select_pending_samples(samples: list[dict], existing_records: list[dict]) -> tuple[list[int], list[dict]]:
    completed_instance_ids = {
        record["instance_id"]
        for record in existing_records
        if record.get("instance_id")
    }
    pending_indices = []
    pending_samples = []
    for index, sample in enumerate(samples):
        if instance_name(sample) in completed_instance_ids:
            continue
        pending_indices.append(index)
        pending_samples.append(sample)
    return pending_indices, pending_samples


def build_result_record(index: int, sample: dict, result: dict) -> dict:
    instance = sample["extra_info"]["tools_kwargs"]
    metadata = instance["reward"]["metadata"]
    outcome = result.get("outcome") or classify_result(result)
    return {
        "index": index,
        "instance_id": result.get("instance_id") or metadata.get("instance_id"),
        "outcome": outcome,
        "resolved": bool(result.get("resolved")),
        "eval_completed": bool(result.get("eval_completed")),
        "eval_execution_time": result.get("eval_execution_time"),
        "error": result.get("error"),
        "run_id": result.get("run_id"),
        "eval_log_path": result.get("eval_log_path"),
        "source_image": result.get("source_image") or instance["env"].get("image"),
        "resolved_image": result.get("resolved_image"),
        "reward_name": instance["reward"].get("name"),
        "repo": metadata.get("repo"),
        "base_commit": metadata.get("base_commit"),
        "version": metadata.get("version"),
        "eval_report": result.get("eval_report"),
    }


def summarize_records(records: list[dict], execution_time: float) -> dict:
    success_instances = [record["instance_id"] for record in records if record["outcome"] == "success"]
    fail_wa_instances = [record["instance_id"] for record in records if record["outcome"] == "fail_wa"]
    fail_tle_instances = [record["instance_id"] for record in records if record["outcome"] == "fail_tle_or_error"]
    execution_times = [record["eval_execution_time"] for record in records if record["eval_execution_time"] is not None]
    return {
        "all_num": len(records),
        "success_num": len(success_instances),
        "fail_wa_num": len(fail_wa_instances),
        "fail_tle_num": len(fail_tle_instances),
        "execution_time": execution_time,
        "avg_execution_time": (sum(execution_times) / len(execution_times)) if execution_times else None,
        "success_instances": success_instances,
        "failed_instances": fail_wa_instances + fail_tle_instances,
        "fail_wa_instances": fail_wa_instances,
        "fail_tle_or_error_instances": fail_tle_instances,
    }


def write_verification_artifacts(
    samples: list[dict],
    results: list[dict],
    *,
    execution_time: float,
    run_dir: Path,
    eval_log_dir: str,
    deployment: str,
    args: argparse.Namespace,
) -> dict:
    records = [build_result_record(index, sample, result) for index, (sample, result) in enumerate(zip(samples, results, strict=False))]
    summary = summarize_records(records, execution_time)
    summary.update(
        {
            "total_samples": len(samples),
            "deployment": deployment,
            "data_path": os.path.expanduser(args.data_path),
            "dataset_split": args.dataset_split,
            "max_samples": args.max_samples,
            "num_workers": args.num_workers,
            "worker_concurrency": args.worker_concurrency,
            "eval_timeout": args.eval_timeout,
            "start_max_retries": args.start_max_retries,
            "eval_log_dir": str(Path(eval_log_dir).expanduser()),
            "run_dir": str(run_dir),
        }
    )

    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (run_dir / "success_instances.txt").write_text("\n".join(summary["success_instances"]) + ("\n" if summary["success_instances"] else ""), encoding="utf-8")
    failed_lines = [f"{record['instance_id']}\t{record['outcome']}" for record in records if record["outcome"] != "success"]
    (run_dir / "failed_instances.txt").write_text("\n".join(failed_lines) + ("\n" if failed_lines else ""), encoding="utf-8")
    return summary


class VerificationRecorder:
    def __init__(
        self,
        *,
        samples: list[dict],
        run_dir: Path,
        eval_log_dir: str,
        deployment: str,
        args: argparse.Namespace,
        existing_records: list[dict] | None = None,
        execution_time_offset: float = 0.0,
    ) -> None:
        self.samples = samples
        self.run_dir = run_dir
        self.eval_log_dir = eval_log_dir
        self.deployment = deployment
        self.args = args
        self.records: list[dict] = list(existing_records or [])
        self.execution_time_offset = execution_time_offset
        if existing_records:
            self._write_artifacts(execution_time=self.execution_time_offset)
        else:
            (self.run_dir / "results.jsonl").write_text("", encoding="utf-8")
            self._write_artifacts(execution_time=0.0)

    def record_result(self, index: int, sample: dict, result: dict, *, execution_time: float) -> None:
        record = build_result_record(index, sample, result)
        self.records.append(record)
        with (self.run_dir / "results.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self._write_artifacts(execution_time=self.execution_time_offset + execution_time)

    def finalize(self, *, execution_time: float) -> dict:
        return self._write_artifacts(execution_time=self.execution_time_offset + execution_time)

    def _write_artifacts(self, *, execution_time: float) -> dict:
        summary = summarize_records(self.records, execution_time)
        summary.update(
            {
                "total_samples": len(self.samples),
                "deployment": self.deployment,
                "data_path": os.path.expanduser(self.args.data_path),
                "dataset_split": self.args.dataset_split,
                "max_samples": self.args.max_samples,
                "num_workers": self.args.num_workers,
                "worker_concurrency": self.args.worker_concurrency,
                "eval_timeout": self.args.eval_timeout,
                "start_max_retries": self.args.start_max_retries,
                "eval_log_dir": str(Path(self.eval_log_dir).expanduser()),
                "run_dir": str(self.run_dir),
            }
        )
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        (self.run_dir / "success_instances.txt").write_text(
            "\n".join(summary["success_instances"]) + ("\n" if summary["success_instances"] else ""),
            encoding="utf-8",
        )
        failed_lines = [f"{record['instance_id']}\t{record['outcome']}" for record in self.records if record["outcome"] != "success"]
        (self.run_dir / "failed_instances.txt").write_text(
            "\n".join(failed_lines) + ("\n" if failed_lines else ""),
            encoding="utf-8",
        )
        return summary


async def run_sample(
    sample,
    *,
    eval_log_dir: str = "/tmp/eval_gold_patch",
    eval_timeout: float = 600.0,
    start_max_retries: int = 5,
):
    run_id = str(uuid.uuid4())
    instance = sample["extra_info"]["tools_kwargs"]
    metadata = instance["reward"]["metadata"]
    sample_instance_id = metadata.get("instance_id")
    impl = os.getenv("DEPLOYMENT", "vefaas").lower()
    log_dir = Path(eval_log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.log"
    add_file_handler(log_path, run_id)

    deployment_config: dict = {}
    env = None
    result = {
        "eval_completed": False,
        "eval_execution_time": None,
        "eval_report": None,
        "resolved": False,
    }
    try:
        deployment_config = build_deployment_config(impl, instance["env"])
        env_config = {
            "deployment": deployment_config,
            "env_variables": {
                "PIP_PROGRESS_BAR": "off",
                "PIP_CACHE_DIR": "~/.cache/pip",
                "PAGER": "cat",
                "MANPAGER": "cat",
                "LESS": "-R",
                "TQDM_DISABLE": "1",
                "GIT_PAGER": "cat",
            },
            "post_setup_cmd": instance["env"]["post_setup_cmd"],
        }
        env_config = AgentEnvConfig(**env_config)
        env = AgentEnv(run_id=run_id, env_config=env_config)

        reward_config = {
            "name": instance["reward"]["name"],
            "run_id": run_id,
            "metadata": instance["reward"]["metadata"],
            "env": env,
            "eval_timeout": eval_timeout,
        }
        reward_spec = load_reward_spec(reward_config)

        await env.start(max_retries=start_max_retries)
        await reward_spec.apply_gold_patch()
        _, result = await reward_spec.compute_reward()
    except Exception as exc:
        logger.exception("Failed to verify sample %s: %s", sample_instance_id, exc)
        result["error"] = str(exc)
    finally:
        try:
            if env is not None:
                await env.close()
        finally:
            cleanup_handlers(run_id)
    if not isinstance(result, dict):
        result = {"eval_completed": False, "eval_execution_time": None, "eval_report": None, "resolved": False, "raw_result": repr(result)}
    result.setdefault("eval_completed", False)
    result.setdefault("eval_execution_time", None)
    result.setdefault("eval_report", None)
    result.setdefault("resolved", False)
    result.update(
        {
            "instance_id": sample_instance_id,
            "run_id": run_id,
            "source_image": instance["env"].get("image"),
            "resolved_image": deployment_config.get("image"),
            "eval_log_path": str(log_path),
        }
    )
    result["outcome"] = classify_result(result)
    return result


async def run_samples_locally(
    samples: list[dict],
    *,
    sample_indices: list[int] | None = None,
    concurrency: int,
    eval_log_dir: str,
    eval_timeout: float,
    start_max_retries: int,
    on_result=None,
):
    semaphore = asyncio.Semaphore(concurrency)
    started_at = time.time()
    if sample_indices is None:
        sample_indices = list(range(len(samples)))
    if len(sample_indices) != len(samples):
        raise ValueError("sample_indices must match samples length")

    async def run_single(local_index: int, sample: dict):
        async with semaphore:
            result = await run_sample(
                sample,
                eval_log_dir=eval_log_dir,
                eval_timeout=eval_timeout,
                start_max_retries=start_max_retries,
            )
            return local_index, result

    results: list[dict | None] = [None] * len(samples)
    tasks = [asyncio.create_task(run_single(index, sample)) for index, sample in enumerate(samples)]
    for task in asyncio.as_completed(tasks):
        local_index, result = await task
        results[local_index] = result
        if on_result:
            on_result(sample_indices[local_index], samples[local_index], result, execution_time=time.time() - started_at)
    return [result for result in results if result is not None]


def run_samples_with_ray(
    samples: list[dict],
    *,
    num_workers: int,
    worker_concurrency: int,
    eval_log_dir: str,
    eval_timeout: float,
    start_max_retries: int,
):
    try:
        import ray
    except ImportError as exc:
        raise RuntimeError("Ray is required when --num-workers is greater than 1. Install ray or use --num-workers 1.") from exc

    @ray.remote
    class TestEvalActor:
        def __init__(self, concurrency: int, eval_log_dir: str, eval_timeout: float, start_max_retries: int):
            self._semaphore = asyncio.Semaphore(concurrency)
            self._eval_log_dir = eval_log_dir
            self._eval_timeout = eval_timeout
            self._start_max_retries = start_max_retries

        async def run_batch(self, batch):
            tasks = [self.run_single(sample) for sample in batch]
            return await asyncio.gather(*tasks)

        async def run_single(self, sample):
            async with self._semaphore:
                return await run_sample(
                    sample,
                    eval_log_dir=self._eval_log_dir,
                    eval_timeout=self._eval_timeout,
                    start_max_retries=self._start_max_retries,
                )

    ray.init()
    worker_count = min(num_workers, len(samples)) if samples else 1
    workers = [
        TestEvalActor.remote(worker_concurrency, eval_log_dir, eval_timeout, start_max_retries)
        for _ in range(worker_count)
    ]
    futures = []
    chunk_size = (len(samples) - 1) // len(workers) + 1 if samples else 1
    for i, worker in enumerate(workers):
        chunk = samples[i * chunk_size : (i + 1) * chunk_size]
        if chunk:
            futures.append(worker.run_batch.remote(chunk))
    results_chunk = ray.get(futures) if futures else []
    return [item for chunk in results_chunk for item in chunk]


def log_summary(samples: list[dict], results: list[dict], execution_time: float) -> None:
    records = [build_result_record(index, sample, result) for index, (sample, result) in enumerate(zip(samples, results, strict=False))]
    summary = summarize_records(records, execution_time)
    logger.info(f"time cost: {execution_time:.2f}s")
    logger.info(
        f"all_num: {summary['all_num']}, success_num: {summary['success_num']}, fail_wa_num: {summary['fail_wa_num']}, fail_tle_num: {summary['fail_tle_num']}"
    )
    if summary["avg_execution_time"] is not None:
        exec_times = [record["eval_execution_time"] for record in records if record["eval_execution_time"] is not None]
        logger.info(f"avg_execution_time: {summary['avg_execution_time']:.2f}s (n={len(exec_times)})")
    else:
        logger.info("avg_execution_time: n/a (n=0)")

    logger.info(f"success instance names: {summary['success_instances']}")
    logger.info(f"fail_wa instance names: {summary['fail_wa_instances']}")
    logger.info(f"fail_tle instance names: {summary['fail_tle_or_error_instances']}")


def main(args):
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
    impl = os.getenv("DEPLOYMENT", "vefaas").lower()
    num_workers, worker_concurrency = resolve_execution_defaults(impl, args.num_workers, args.worker_concurrency)
    start_max_retries = resolve_start_max_retries(impl, args.start_max_retries)
    args.num_workers = num_workers
    args.worker_concurrency = worker_concurrency
    args.start_max_retries = start_max_retries
    samples = load_samples(args.data_path, args.cache_dir, args.dataset_split, args.max_samples)
    result_run_dir, existing_records, execution_time_offset = resolve_result_run_dir(
        results_dir=args.results_dir,
        run_name=args.run_name,
        auto_resume=not getattr(args, "no_resume_latest", False),
        data_path=args.data_path,
        deployment=impl,
        args=args,
    )
    pending_indices, pending_samples = select_pending_samples(samples, existing_records)
    eval_log_dir = args.eval_log_dir or str(result_run_dir / "raw_logs")
    recorder = VerificationRecorder(
        samples=samples,
        run_dir=result_run_dir,
        eval_log_dir=eval_log_dir,
        deployment=impl,
        args=args,
        existing_records=existing_records,
        execution_time_offset=execution_time_offset,
    )
    if not pending_samples:
        summary = recorder.finalize(execution_time=0.0)
        logger.info("No pending samples remain for verification run: %s", result_run_dir)
        logger.info("per-instance verification results written to: %s", result_run_dir)
        logger.info("success instances: %s", summary["success_instances"])
        logger.info("failed instances: %s", summary["failed_instances"])
        return

    begin_time = time.time()
    if num_workers == 1:
        results = asyncio.run(
            run_samples_locally(
                pending_samples,
                sample_indices=pending_indices,
                concurrency=worker_concurrency,
                eval_log_dir=eval_log_dir,
                eval_timeout=args.eval_timeout,
                start_max_retries=start_max_retries,
                on_result=recorder.record_result,
            )
        )
    else:
        results = run_samples_with_ray(
            pending_samples,
            num_workers=num_workers,
            worker_concurrency=worker_concurrency,
            eval_log_dir=eval_log_dir,
            eval_timeout=args.eval_timeout,
            start_max_retries=start_max_retries,
        )
    end_time = time.time()
    execution_time = end_time - begin_time
    log_summary(pending_samples, results, execution_time)
    if num_workers == 1:
        summary = recorder.finalize(execution_time=execution_time)
    else:
        summary = write_verification_artifacts(
            samples,
            results,
            execution_time=execution_time,
            run_dir=result_run_dir,
            eval_log_dir=eval_log_dir,
            deployment=impl,
            args=args,
        )
    logger.info("per-instance verification results written to: %s", result_run_dir)
    logger.info("success instances: %s", summary["success_instances"])
    logger.info("failed instances: %s", summary["failed_instances"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/home/tiger/data/swe_agent/swe_bench_verified_modal.parquet")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--worker-concurrency", type=int, default=None)
    parser.add_argument("--eval-timeout", type=float, default=600.0)
    parser.add_argument("--start-max-retries", type=int, default=None)
    parser.add_argument("--eval-log-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=os.getenv("UNI_AGENT_GOLD_PATCH_RESULTS_DIR", str(DEFAULT_RESULTS_DIR)))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-resume-latest", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=os.getenv("UNI_AGENT_HF_CACHE_DIR", DEFAULT_CACHE_DIR))
    parser.add_argument("--dataset-split", type=str, default="test")
    args = parser.parse_args()

    main(args)
