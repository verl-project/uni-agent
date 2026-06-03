import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
_INFER_SCRIPT = _REPO_ROOT / "examples" / "agent_interaction" / "parallel_infer.py"
_WORKSPACE_ROOT = _REPO_ROOT.parent

_PARALLEL_INFER_ENV_DEFAULTS = [
    "UNI_AGENT_DATA_PATH",
    "DATA_PATH",
    "UNI_AGENT_MODEL_PATH",
    "MODEL_PATH",
    "UNI_AGENT_AGENT_CONFIG_PATH",
    "AGENT_CONFIG_PATH",
    "UNI_AGENT_HF_CACHE_DIR",
    "HF_DATASETS_CACHE",
    "UNI_AGENT_RAY_TMP_DIR",
    "RAY_TMPDIR",
    "LOCAL_DEPLOYMENT_IMAGE_DIR",
    "UNI_AGENT_LOCAL_SIF_IMAGE_DIR",
    "UNI_AGENT_LOCAL_IMAGE_DIR",
]


def _clear_parallel_infer_env(monkeypatch):
    for name in _PARALLEL_INFER_ENV_DEFAULTS:
        monkeypatch.delenv(name, raising=False)


def _load_infer_module():
    module_name = "_parallel_infer_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, _INFER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sample(instance_id="astropy__astropy-12907"):
    return {
        "prompt": [{"role": "user", "content": "fix it"}],
        "agent_name": "swe_agent",
        "extra_info": {
            "tools_kwargs": {
                "env": {
                    "image": "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
                    "post_setup_cmd": "cd /testbed && git checkout abc123",
                },
                "reward": {
                    "name": "swe_bench",
                    "metadata": {"instance_id": instance_id},
                },
            }
        },
    }


def test_path_defaults_are_env_owned(monkeypatch):
    _clear_parallel_infer_env(monkeypatch)
    module = _load_infer_module()

    args = module.build_arg_parser().parse_args([])

    assert args.data_path is None
    assert args.model_path is None
    assert args.agent_config_path is None
    assert args.cache_dir is None
    assert args.ray_temp_dir is None
    assert args.local_image_dir is None


def test_runtime_env_defines_parallel_infer_path_defaults():
    runtime_env = yaml.safe_load((_WORKSPACE_ROOT / "scripts" / "runtime_env.yaml").read_text(encoding="utf-8"))
    env_vars = runtime_env["env_vars"]

    assert env_vars["UNI_AGENT_DATA_PATH"] == "data/swe_agent/swe_bench_verified_local.parquet"
    assert env_vars["UNI_AGENT_MODEL_PATH"] == "models/Qwen3.5-35B-A3B"
    assert env_vars["UNI_AGENT_AGENT_CONFIG_PATH"] == "uni-agent/examples/agent_interaction/agent_config_local.yaml"
    assert env_vars["UNI_AGENT_HF_CACHE_DIR"] == "tmp/inference/hf-cache"
    assert env_vars["UNI_AGENT_RAY_TMP_DIR"] == "tmp/ray"
    assert env_vars["UNI_AGENT_LOCAL_SIF_IMAGE_DIR"] == "data/swe_agent/arch_specific_images/images"


def test_arg_parser_defaults_are_environment_configurable(monkeypatch, tmp_path):
    module = _load_infer_module()
    env_defaults = {
        "UNI_AGENT_DATA_PATH": str(tmp_path / "data.parquet"),
        "UNI_AGENT_MODEL_PATH": "Qwen/Qwen3.5-35B-A3B",
        "UNI_AGENT_AGENT_CONFIG_PATH": str(tmp_path / "agent.yaml"),
        "UNI_AGENT_HF_CACHE_DIR": str(tmp_path / "hf-cache"),
        "UNI_AGENT_RAY_TMP_DIR": str(tmp_path / "ray"),
        "UNI_AGENT_DATASET_SPLIT": "validation",
        "DEPLOYMENT": "modal",
        "LOCAL_DEPLOYMENT_IMAGE_ARCH": "keep",
        "UNI_AGENT_LOCAL_SIF_IMAGE_DIR": str(tmp_path / "sifs"),
        "UNI_AGENT_LOCAL_IMAGE_NAMESPACE": "registry.example.com/team/images",
        "UNI_AGENT_MAX_TURNS": "12",
        "UNI_AGENT_PROMPT_LENGTH": "1024",
        "UNI_AGENT_RESPONSE_LENGTH": "2048",
        "UNI_AGENT_MAX_MODEL_LEN": "4096",
        "UNI_AGENT_TEMPERATURE": "0.25",
        "UNI_AGENT_TOP_P": "0.75",
        "UNI_AGENT_N": "3",
        "UNI_AGENT_MAX_SAMPLES": "5",
        "UNI_AGENT_ENGINE": "sglang",
        "UNI_AGENT_NUM_WORKERS": "6",
        "UNI_AGENT_NNODES": "2",
        "UNI_AGENT_N_GPUS_PER_NODE": "4",
        "UNI_AGENT_TENSOR_PARALLEL_SIZE": "4",
        "UNI_AGENT_GPU_MEMORY_UTILIZATION": "0.55",
        "UNI_AGENT_ALLOW_REMOTE_MODEL": "true",
    }
    for key, value in env_defaults.items():
        monkeypatch.setenv(key, value)

    args = module.build_arg_parser().parse_args([])

    assert args.data_path == env_defaults["UNI_AGENT_DATA_PATH"]
    assert args.model_path == env_defaults["UNI_AGENT_MODEL_PATH"]
    assert args.allow_remote_model is True
    assert args.agent_config_path == env_defaults["UNI_AGENT_AGENT_CONFIG_PATH"]
    assert args.cache_dir == env_defaults["UNI_AGENT_HF_CACHE_DIR"]
    assert args.ray_temp_dir == env_defaults["UNI_AGENT_RAY_TMP_DIR"]
    assert args.dataset_split == "validation"
    assert args.deployment == "modal"
    assert args.local_image_arch == "keep"
    assert args.local_image_dir == env_defaults["UNI_AGENT_LOCAL_SIF_IMAGE_DIR"]
    assert args.local_image_namespace == env_defaults["UNI_AGENT_LOCAL_IMAGE_NAMESPACE"]
    assert args.max_turns == 12
    assert args.prompt_length == 1024
    assert args.response_length == 2048
    assert args.max_model_len == 4096
    assert args.temperature == 0.25
    assert args.top_p == 0.75
    assert args.n == 3
    assert args.max_samples == 5
    assert args.engine == "sglang"
    assert args.num_workers == 6
    assert args.nnodes == 2
    assert args.n_gpus_per_node == 4
    assert args.tensor_parallel_size == 4
    assert args.gpu_memory_utilization == 0.55


def test_arg_parser_cli_overrides_environment_defaults(monkeypatch):
    module = _load_infer_module()
    monkeypatch.setenv("UNI_AGENT_MODEL_PATH", "Qwen/env-model")
    monkeypatch.setenv("UNI_AGENT_MAX_SAMPLES", "7")
    monkeypatch.setenv("UNI_AGENT_ALLOW_REMOTE_MODEL", "true")

    args = module.build_arg_parser().parse_args(
        [
            "--model-path",
            "models/cli-model",
            "--max-samples",
            "2",
            "--no-allow-remote-model",
        ]
    )

    assert args.model_path == "models/cli-model"
    assert args.max_samples == 2
    assert args.allow_remote_model is False


def test_invalid_environment_defaults_raise_clear_error(monkeypatch):
    module = _load_infer_module()
    monkeypatch.setenv("UNI_AGENT_MAX_TURNS", "not-an-int")

    try:
        module.build_arg_parser()
    except ValueError as exc:
        assert "UNI_AGENT_MAX_TURNS/MAX_TURNS must be an integer" in str(exc)
    else:
        raise AssertionError("invalid integer environment default should fail parser construction")


def test_resolve_optional_workspace_dir_handles_relative_and_disabled(tmp_path):
    module = _load_infer_module()

    relative = module.resolve_optional_workspace_dir("tmp/inference/test-ray")
    absolute = module.resolve_optional_workspace_dir(tmp_path / "absolute-ray")

    assert relative == str(module.WORKSPACE_ROOT / "tmp" / "inference" / "test-ray")
    assert Path(relative).is_dir()
    assert absolute == str(tmp_path / "absolute-ray")
    assert Path(absolute).is_dir()
    assert module.resolve_optional_workspace_dir("none") is None


def test_init_ray_disables_dashboard_and_uses_configured_temp_dir(tmp_path):
    module = _load_infer_module()
    calls = []
    fake_ray = SimpleNamespace(init=lambda **kwargs: calls.append(kwargs))

    module.init_ray(fake_ray, tmp_path / "ray")
    module.init_ray(fake_ray, "none")

    assert calls == [
        {"include_dashboard": False, "_temp_dir": str(tmp_path / "ray")},
        {"include_dashboard": False},
    ]


def test_local_image_prefers_existing_sif(monkeypatch, tmp_path):
    module = _load_infer_module()
    monkeypatch.setattr(module.platform, "machine", lambda: "aarch64")
    sif = tmp_path / "swebench_sweb.eval.arm64.astropy_1776_astropy-12907_latest.sif"
    sif.write_text("fake sif", encoding="utf-8")

    image = module.resolve_local_image(
        "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
        image_dir=tmp_path,
    )

    assert image == str(sif)


def test_prepare_samples_for_local_deployment_rewrites_copy(monkeypatch, tmp_path):
    module = _load_infer_module()
    monkeypatch.setattr(module.platform, "machine", lambda: "aarch64")
    sample = _sample()

    prepared = module.prepare_samples_for_deployment([sample], "local", local_image_dir=tmp_path)

    assert sample["extra_info"]["tools_kwargs"]["env"]["image"].startswith("swebench/sweb.eval.x86_64.")
    assert prepared[0]["extra_info"]["tools_kwargs"]["env"]["image"] == (
        "swebench/sweb.eval.arm64.astropy_1776_astropy-12907:latest"
    )


def test_load_samples_uses_cache_split_and_local_rewrite(monkeypatch, tmp_path):
    module = _load_infer_module()
    monkeypatch.setattr(module.platform, "machine", lambda: "aarch64")
    calls = {}

    class FakeDataset:
        def to_list(self):
            return [_sample("one"), _sample("two")]

    def fake_load_parquet_dataset(data_path, dataset_split, cache_dir):
        calls.update({"data_path": data_path, "dataset_split": dataset_split, "cache_dir": cache_dir})
        return FakeDataset()

    monkeypatch.setattr(module, "_load_parquet_dataset", fake_load_parquet_dataset)

    samples = module.load_samples(
        "data.parquet",
        cache_dir=str(tmp_path / "cache"),
        dataset_split="train",
        max_samples=1,
        deployment="local",
        local_image_arch="arm64",
        local_image_dir=tmp_path,
        local_image_namespace=None,
    )

    assert len(samples) == 1
    assert calls == {
        "data_path": "data.parquet",
        "dataset_split": "train",
        "cache_dir": str(tmp_path / "cache"),
    }
    assert samples[0]["extra_info"]["tools_kwargs"]["env"]["image"].startswith("swebench/sweb.eval.arm64.")


def test_local_image_namespace_is_argument_not_default(monkeypatch, tmp_path):
    module = _load_infer_module()
    monkeypatch.setattr(module.platform, "machine", lambda: "aarch64")

    image = module.resolve_local_image(
        "swebench/sweb.eval.x86_64.astropy_1776_astropy-13033:latest",
        image_dir=tmp_path,
        image_namespace="registry.example.com/team/swe-bench-images",
    )

    assert image == "registry.example.com/team/swe-bench-images/sweb.eval.arm64.astropy_1776_astropy-13033:latest"


def test_validate_model_path_requires_existing_local_path(tmp_path):
    module = _load_infer_module()
    missing = tmp_path / "Qwen3.5-35B-A3B"

    try:
        module.validate_model_path(str(missing), allow_remote_model=False)
    except FileNotFoundError as exc:
        assert "Qwen3.5-35B-A3B" in str(exc)
    else:
        raise AssertionError("validate_model_path should reject a missing local model path")

    assert module.validate_model_path("Qwen/Qwen3.5-35B-A3B", allow_remote_model=True) == "Qwen/Qwen3.5-35B-A3B"
