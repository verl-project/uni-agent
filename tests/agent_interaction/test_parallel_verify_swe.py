import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERIFY_SCRIPT = _REPO_ROOT / "examples" / "agent_interaction" / "parallel_verify_swe.py"


def _load_verify_module():
    module_name = "_parallel_verify_swe_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, _VERIFY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sample(instance_id="astropy__astropy-12907"):
    return {
        "extra_info": {
            "tools_kwargs": {
                "env": {
                    "image": "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
                    "post_setup_cmd": "cd /testbed && git checkout abc123",
                },
                "reward": {
                    "name": "swe_bench",
                    "metadata": {
                        "instance_id": instance_id,
                        "patch": "diff --git a/a.py b/a.py\n",
                    },
                },
            }
        }
    }


def test_default_cache_dir_is_workspace_local():
    module = _load_verify_module()

    assert module.DEFAULT_CACHE_DIR == str(module.WORKSPACE_ROOT / "tmp" / "uni-agent-hf-cache")


def test_local_deployment_config_uses_sample_image_and_env_overrides(monkeypatch):
    module = _load_verify_module()
    monkeypatch.setattr(module.platform, "machine", lambda: "aarch64")
    monkeypatch.setenv("LOCAL_DEPLOYMENT_IMAGE_DIR", "none")
    monkeypatch.setenv("LOCAL_CONTAINER_RUNTIME", "/opt/apptainer/bin/apptainer")
    monkeypatch.setenv("LOCAL_DEPLOYMENT_COMMAND", "server --port {port} --auth-token {token}")
    monkeypatch.setenv("LOCAL_DEPLOYMENT_NETWORK", "host")
    monkeypatch.setenv("LOCAL_DEPLOYMENT_HOST", "http://127.0.0.1")
    monkeypatch.setenv("LOCAL_DEPLOYMENT_PORT", "4567")
    monkeypatch.setenv("LOCAL_DEPLOYMENT_EXTRA_ARGS", "--bind /host:/mnt --contain")

    config = module.build_deployment_config("local", _sample()["extra_info"]["tools_kwargs"]["env"])

    assert config == {
        "type": "local",
        "image": "swebench/sweb.eval.arm64.astropy_1776_astropy-12907:latest",
        "command": "server --port {port} --auth-token {token}",
        "timeout": 600.0,
        "startup_timeout": 180.0,
        "container_runtime": "/opt/apptainer/bin/apptainer",
        "network": "host",
        "host": "http://127.0.0.1",
        "published_port": 4567,
        "extra_run_args": ["--bind", "/host:/mnt", "--contain"],
    }


def test_local_image_arch_can_be_kept(monkeypatch):
    module = _load_verify_module()
    monkeypatch.setenv("LOCAL_DEPLOYMENT_IMAGE_ARCH", "keep")

    image = module.resolve_local_image("swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest")

    assert image == "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"


def test_local_image_prefers_existing_arch_specific_sif(monkeypatch, tmp_path):
    module = _load_verify_module()
    monkeypatch.setattr(module.platform, "machine", lambda: "aarch64")
    sif = tmp_path / "swebench_sweb.eval.arm64.astropy_1776_astropy-12907_latest.sif"
    sif.write_text("fake sif", encoding="utf-8")

    image = module.resolve_local_image(
        "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
        image_dir=tmp_path,
    )

    assert image == str(sif)


def test_local_image_falls_back_to_arch_docker_image_when_sif_missing(monkeypatch, tmp_path):
    module = _load_verify_module()
    monkeypatch.setattr(module.platform, "machine", lambda: "aarch64")

    image = module.resolve_local_image(
        "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
        image_dir=tmp_path,
    )

    assert image == "swebench/sweb.eval.arm64.astropy_1776_astropy-12907:latest"


def test_execution_defaults_are_conservative_for_local():
    module = _load_verify_module()

    assert module.resolve_execution_defaults("local", None, None) == (1, 1)
    assert module.resolve_execution_defaults("vefaas", None, None) == (8, 8)
    assert module.resolve_execution_defaults("local", 2, 3) == (2, 3)
    assert module.resolve_start_max_retries("local", None) == 1
    assert module.resolve_start_max_retries("vefaas", None) == 5
    assert module.resolve_start_max_retries("local", 2) == 2


def test_load_samples_uses_explicit_cache_dir_and_max_samples(monkeypatch, tmp_path):
    module = _load_verify_module()
    calls = {}

    class FakeDataset:
        def to_list(self):
            return [{"id": 1}, {"id": 2}, {"id": 3}]

    def fake_load_dataset(name, data_files, split, cache_dir):
        calls.update({"name": name, "data_files": data_files, "split": split, "cache_dir": cache_dir})
        return FakeDataset()

    monkeypatch.setattr(module, "load_dataset", fake_load_dataset)

    samples = module.load_samples(str(tmp_path / "data.parquet"), str(tmp_path / "cache"), "test", max_samples=2)

    assert samples == [{"id": 1}, {"id": 2}]
    assert calls == {
        "name": "parquet",
        "data_files": {"test": str(tmp_path / "data.parquet")},
        "split": "test",
        "cache_dir": str(tmp_path / "cache"),
    }


def test_log_summary_handles_no_execution_times():
    module = _load_verify_module()

    module.log_summary([_sample()], [{"resolved": False, "eval_completed": False}], execution_time=0.1)


def test_write_verification_artifacts_persists_success_and_failure_lists(tmp_path):
    module = _load_verify_module()
    run_dir = module.create_result_run_dir(str(tmp_path), "manual-run")
    samples = [_sample("astropy__success"), _sample("astropy__wa"), _sample("astropy__timeout")]
    results = [
        {
            "resolved": True,
            "eval_completed": True,
            "eval_execution_time": 1.0,
            "eval_report": {"resolved": True},
            "run_id": "run-success",
            "eval_log_path": "/logs/run-success.log",
        },
        {
            "resolved": False,
            "eval_completed": True,
            "eval_execution_time": 2.0,
            "eval_report": {"resolved": False},
            "run_id": "run-wa",
            "error": "wrong answer",
        },
        {
            "resolved": False,
            "eval_completed": False,
            "eval_execution_time": None,
            "eval_report": None,
            "run_id": "run-timeout",
            "error": "timeout",
        },
    ]
    args = SimpleNamespace(
        data_path="data.parquet",
        dataset_split="test",
        max_samples=3,
        num_workers=1,
        worker_concurrency=1,
        eval_timeout=60.0,
        start_max_retries=1,
    )

    summary = module.write_verification_artifacts(
        samples,
        results,
        execution_time=3.5,
        run_dir=run_dir,
        eval_log_dir=str(run_dir / "raw_logs"),
        deployment="local",
        args=args,
    )

    records = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    persisted_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert (tmp_path / "latest_run.txt").read_text(encoding="utf-8").strip() == str(run_dir)
    assert [record["instance_id"] for record in records] == ["astropy__success", "astropy__wa", "astropy__timeout"]
    assert [record["outcome"] for record in records] == ["success", "fail_wa", "fail_tle_or_error"]
    assert (run_dir / "success_instances.txt").read_text(encoding="utf-8") == "astropy__success\n"
    assert (run_dir / "failed_instances.txt").read_text(encoding="utf-8") == (
        "astropy__wa\tfail_wa\n"
        "astropy__timeout\tfail_tle_or_error\n"
    )
    assert summary["success_instances"] == ["astropy__success"]
    assert persisted_summary["failed_instances"] == ["astropy__wa", "astropy__timeout"]
    assert persisted_summary["avg_execution_time"] == 1.5


def test_main_creates_durable_results_run(monkeypatch, tmp_path):
    module = _load_verify_module()
    monkeypatch.setenv("DEPLOYMENT", "local")
    samples = [_sample("astropy__main-success")]

    def fake_load_samples(data_path, cache_dir, dataset_split, max_samples):
        assert data_path == "data.parquet"
        assert dataset_split == "test"
        assert max_samples == 1
        return samples

    async def fake_run_samples_locally(
        samples_arg,
        *,
        sample_indices,
        concurrency,
        eval_log_dir,
        eval_timeout,
        start_max_retries,
        on_result,
    ):
        assert samples_arg == samples
        assert sample_indices == [0]
        assert concurrency == 1
        assert eval_log_dir == str(tmp_path / "runs" / "fixed-run" / "raw_logs")
        assert eval_timeout == 60.0
        assert start_max_retries == 1
        result = {
            "resolved": True,
            "eval_completed": True,
            "eval_execution_time": 4.0,
            "eval_report": {"resolved": True},
            "run_id": "main-run-id",
            "eval_log_path": str(Path(eval_log_dir) / "main-run-id.log"),
        }
        on_result(0, samples_arg[0], result, execution_time=0.25)
        records_path = tmp_path / "runs" / "fixed-run" / "results.jsonl"
        assert "astropy__main-success" in records_path.read_text(encoding="utf-8")
        return [result]

    monkeypatch.setattr(module, "load_samples", fake_load_samples)
    monkeypatch.setattr(module, "run_samples_locally", fake_run_samples_locally)
    args = SimpleNamespace(
        data_path="data.parquet",
        max_samples=1,
        num_workers=None,
        worker_concurrency=None,
        eval_timeout=60.0,
        eval_log_dir=None,
        results_dir=str(tmp_path / "runs"),
        run_name="fixed-run",
        no_resume_latest=False,
        cache_dir=None,
        dataset_split="test",
        start_max_retries=None,
    )

    module.main(args)

    run_dir = tmp_path / "runs" / "fixed-run"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["success_instances"] == ["astropy__main-success"]
    assert summary["failed_instances"] == []
    assert (tmp_path / "runs" / "latest_run.txt").read_text(encoding="utf-8").strip() == str(run_dir)


def test_main_resumes_incomplete_latest_run(monkeypatch, tmp_path):
    module = _load_verify_module()
    monkeypatch.setenv("DEPLOYMENT", "local")
    samples = [_sample("astropy__done"), _sample("astropy__pending")]
    run_dir = tmp_path / "runs" / "previous-run"
    run_dir.mkdir(parents=True)
    (tmp_path / "runs" / "latest_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    existing_record = {
        "index": 0,
        "instance_id": "astropy__done",
        "outcome": "success",
        "resolved": True,
        "eval_completed": True,
        "eval_execution_time": 1.0,
        "error": None,
    }
    (run_dir / "results.jsonl").write_text(json.dumps(existing_record) + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "all_num": 1,
                "total_samples": 2,
                "data_path": "data.parquet",
                "deployment": "local",
                "dataset_split": "test",
                "max_samples": -1,
                "execution_time": 10.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_load_samples(data_path, cache_dir, dataset_split, max_samples):
        return samples

    async def fake_run_samples_locally(
        samples_arg,
        *,
        sample_indices,
        concurrency,
        eval_log_dir,
        eval_timeout,
        start_max_retries,
        on_result,
    ):
        assert samples_arg == [samples[1]]
        assert sample_indices == [1]
        result = {
            "resolved": False,
            "eval_completed": False,
            "eval_execution_time": None,
            "eval_report": None,
            "run_id": "resumed-run-id",
            "error": "missing image",
        }
        on_result(1, samples_arg[0], result, execution_time=2.0)
        return [result]

    monkeypatch.setattr(module, "load_samples", fake_load_samples)
    monkeypatch.setattr(module, "run_samples_locally", fake_run_samples_locally)
    args = SimpleNamespace(
        data_path="data.parquet",
        max_samples=-1,
        num_workers=None,
        worker_concurrency=None,
        eval_timeout=60.0,
        eval_log_dir=None,
        results_dir=str(tmp_path / "runs"),
        run_name=None,
        no_resume_latest=False,
        cache_dir=None,
        dataset_split="test",
        start_max_retries=None,
    )

    module.main(args)

    records = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert [record["instance_id"] for record in records] == ["astropy__done", "astropy__pending"]
    assert records[1]["index"] == 1
    assert summary["all_num"] == 2
    assert summary["total_samples"] == 2
    assert summary["execution_time"] >= 10.0


def test_run_sample_closes_env_and_returns_reward_result(monkeypatch, tmp_path):
    module = _load_verify_module()
    monkeypatch.setenv("DEPLOYMENT", "local")
    created_envs = []

    class FakeEnv:
        def __init__(self, run_id, env_config):
            self.run_id = run_id
            self.env_config = env_config
            self.closed = False
            self.start_max_retries = None
            created_envs.append(self)

        async def start(self, max_retries=5):
            self.start_max_retries = max_retries
            return None

        async def close(self):
            self.closed = True

    class FakeReward:
        def __init__(self):
            self.applied = False

        async def apply_gold_patch(self):
            self.applied = True

        async def compute_reward(self):
            return True, {
                "eval_completed": True,
                "eval_execution_time": 1.25,
                "eval_report": {"resolved": True},
                "resolved": True,
            }

    reward = FakeReward()
    monkeypatch.setattr(module, "AgentEnvConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(module, "AgentEnv", FakeEnv)
    monkeypatch.setattr(module, "load_reward_spec", lambda config: reward)

    result = asyncio.run(
        module.run_sample(
            _sample(),
            eval_log_dir=str(tmp_path),
            eval_timeout=12.0,
            start_max_retries=2,
        )
    )

    assert result["resolved"]
    assert result["instance_id"] == "astropy__astropy-12907"
    assert result["outcome"] == "success"
    assert result["source_image"] == "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
    assert result["eval_log_path"].endswith(".log")
    assert reward.applied
    assert created_envs[0].closed
    assert created_envs[0].start_max_retries == 2
    assert created_envs[0].env_config["deployment"]["type"] == "local"
    assert created_envs[0].env_config["post_setup_cmd"] == "cd /testbed && git checkout abc123"
