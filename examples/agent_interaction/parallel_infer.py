import argparse
import copy
import logging
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

UNI_AGENT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = UNI_AGENT_ROOT.parent
sys.path.insert(0, str(UNI_AGENT_ROOT))
sys.path.insert(0, str(UNI_AGENT_ROOT / "verl"))

# Setup basic logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=os.getenv("VERL_LOGGING_LEVEL", "INFO")
)
logger = logging.getLogger(__name__)

def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return None


def _env_str(*names: str, default: str | os.PathLike[str] | None = None) -> str | None:
    value = _first_env(*names)
    if value is not None:
        return value
    if default is None:
        return None
    return str(default)


def _env_int(*names: str, default: int) -> int:
    value = _first_env(*names)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        joined = "/".join(names)
        raise ValueError(f"{joined} must be an integer, got {value!r}") from exc


def _env_float(*names: str, default: float) -> float:
    value = _first_env(*names)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        joined = "/".join(names)
        raise ValueError(f"{joined} must be a float, got {value!r}") from exc


def _env_bool(*names: str, default: bool = False) -> bool:
    value = _first_env(*names)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    joined = "/".join(names)
    raise ValueError(f"{joined} must be a boolean, got {value!r}")


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
    image_namespace: str | None = None,
) -> str:
    image_arch = (
        image_arch or _env_str("LOCAL_DEPLOYMENT_IMAGE_ARCH", "UNI_AGENT_LOCAL_IMAGE_ARCH", default="auto")
    ).lower()
    if image_arch in {"keep", "none", "false"}:
        return image
    if image_arch == "auto":
        image_arch = _host_swe_image_arch()
    if image_arch not in {"x86_64", "arm64"}:
        raise ValueError("local image arch must be one of: auto, keep, x86_64, arm64")

    resolved_image = image.replace("sweb.eval.x86_64.", f"sweb.eval.{image_arch}.")
    if image_dir is None:
        image_dir = _env_str(
            "LOCAL_DEPLOYMENT_IMAGE_DIR",
            "UNI_AGENT_LOCAL_SIF_IMAGE_DIR",
            "UNI_AGENT_LOCAL_IMAGE_DIR",
        )
    local_sif = _resolve_local_sif(resolved_image, image_dir)
    if local_sif:
        return local_sif

    image_namespace = image_namespace or _env_str(
        "LOCAL_DEPLOYMENT_IMAGE_NAMESPACE",
        "UNI_AGENT_LOCAL_IMAGE_NAMESPACE",
    )
    if image_namespace:
        image_name = resolved_image.split("/", 1)[1] if "/" in resolved_image else resolved_image
        return f"{image_namespace.rstrip('/')}/{image_name}"
    return resolved_image


def prepare_samples_for_deployment(
    samples: list[dict[str, Any]],
    deployment: str,
    *,
    local_image_arch: str | None = None,
    local_image_dir: str | os.PathLike[str] | None = None,
    local_image_namespace: str | None = None,
) -> list[dict[str, Any]]:
    if deployment != "local":
        return samples

    prepared_samples = []
    for sample in samples:
        prepared = copy.deepcopy(sample)
        env = prepared.get("extra_info", {}).get("tools_kwargs", {}).get("env", {})
        if "image" in env:
            env["image"] = resolve_local_image(
                env["image"],
                image_arch=local_image_arch,
                image_dir=local_image_dir,
                image_namespace=local_image_namespace,
            )
        prepared_samples.append(prepared)
    return prepared_samples


def _load_parquet_dataset(data_path: str, dataset_split: str, cache_dir: str | None):
    from datasets import load_dataset

    return load_dataset("parquet", data_files={dataset_split: data_path}, split=dataset_split, cache_dir=cache_dir)


def load_samples(
    data_path: str | os.PathLike[str],
    *,
    cache_dir: str | None,
    dataset_split: str,
    max_samples: int,
    deployment: str,
    local_image_arch: str | None,
    local_image_dir: str | os.PathLike[str] | None,
    local_image_namespace: str | None,
) -> list[dict[str, Any]]:
    data_path = os.path.expanduser(data_path)
    cache_dir = os.path.expanduser(cache_dir) if cache_dir else None
    logger.info("Loading dataset from: %s", data_path)
    dataset = _load_parquet_dataset(data_path, dataset_split, cache_dir)
    samples = dataset.to_list()
    if max_samples > 0:
        samples = samples[: max_samples]
        logger.info("Using first %d samples (--max-samples=%d)", len(samples), max_samples)
    return prepare_samples_for_deployment(
        samples,
        deployment,
        local_image_arch=local_image_arch,
        local_image_dir=local_image_dir,
        local_image_namespace=local_image_namespace,
    )


def validate_model_path(model_path: str, *, allow_remote_model: bool) -> str:
    expanded = os.path.expanduser(model_path)
    if allow_remote_model:
        return expanded
    if not Path(expanded).exists():
        raise FileNotFoundError(
            f"Model path does not exist: {expanded}. "
            "Place Qwen3.5-35B-A3B under ./models/Qwen3.5-35B-A3B or pass --allow-remote-model."
        )
    return expanded


def _require_text(value: str | None, *, option: str, env_names: tuple[str, ...]) -> str:
    if value:
        return value
    env_list = ", ".join(env_names)
    raise ValueError(f"{option} is required. Pass {option} or set one of: {env_list}.")


def validate_required_args(args: argparse.Namespace) -> None:
    args.data_path = _require_text(
        args.data_path,
        option="--data-path",
        env_names=("UNI_AGENT_DATA_PATH", "DATA_PATH"),
    )
    args.model_path = _require_text(
        args.model_path,
        option="--model-path",
        env_names=("UNI_AGENT_MODEL_PATH", "MODEL_PATH"),
    )
    args.agent_config_path = _require_text(
        args.agent_config_path,
        option="--agent-config-path",
        env_names=("UNI_AGENT_AGENT_CONFIG_PATH", "AGENT_CONFIG_PATH"),
    )


def resolve_optional_workspace_dir(path: str | os.PathLike[str] | None) -> str | None:
    if path is None:
        return None
    path_text = str(path).strip()
    if path_text.lower() in {"", "none", "false", "off"}:
        return None
    resolved = Path(path_text).expanduser()
    if not resolved.is_absolute():
        resolved = WORKSPACE_ROOT / resolved
    resolved.mkdir(parents=True, exist_ok=True)
    return str(resolved)


def init_ray(ray_module, ray_temp_dir_arg: str | os.PathLike[str] | None) -> None:
    ray_init_kwargs: dict[str, Any] = {"include_dashboard": False}
    ray_temp_dir = resolve_optional_workspace_dir(ray_temp_dir_arg)
    if ray_temp_dir:
        ray_init_kwargs["_temp_dir"] = ray_temp_dir
    ray_module.init(**ray_init_kwargs)


def _default_visible_gpu_count() -> int:
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible_devices and visible_devices.lower() not in {"none", "void", "no_dev_files", "-1"}:
        return max(1, len([item for item in visible_devices.split(",") if item.strip()]))
    return 1


def init_config(args: argparse.Namespace):
    """Initialize the configuration from hydra and override with command-line arguments."""
    from hydra import compose, initialize_config_dir

    import verl

    # config_dir = os.path.abspath("verl/trainer/config")
    config_dir = str(Path(verl.__file__).resolve().parent / "trainer" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")

    # Override rollout configs
    config.actor_rollout_ref.rollout.agent.agent_loop_config_path = os.path.expanduser(args.agent_config_path)
    config.actor_rollout_ref.rollout.agent.num_workers = args.num_workers
    config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = args.max_turns
    config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls = 1

    # Validation / sampling kwargs
    config.actor_rollout_ref.rollout.temperature = args.temperature
    config.actor_rollout_ref.rollout.top_p = args.top_p
    config.actor_rollout_ref.rollout.val_kwargs.temperature = args.temperature
    config.actor_rollout_ref.rollout.val_kwargs.top_p = args.top_p
    config.actor_rollout_ref.rollout.calculate_log_probs = True

    # Hardware configs
    config.actor_rollout_ref.rollout.nnodes = args.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = args.n_gpus_per_node
    config.trainer.nnodes = args.nnodes
    config.trainer.n_gpus_per_node = args.n_gpus_per_node

    # Model and engine configs
    config.actor_rollout_ref.model.path = os.path.expanduser(args.model_path)
    config.actor_rollout_ref.rollout.name = args.engine
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.prompt_length = args.prompt_length
    config.actor_rollout_ref.rollout.response_length = args.response_length
    config.actor_rollout_ref.rollout.n = args.n
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = args.tensor_parallel_size
    config.actor_rollout_ref.rollout.gpu_memory_utilization = args.gpu_memory_utilization
    if args.max_model_len is not None:
        config.actor_rollout_ref.rollout.max_model_len = args.max_model_len

    # Data configs
    config.data.return_raw_chat = True
    config.data.max_prompt_length = args.prompt_length
    config.data.max_response_length = args.response_length

    return config


def run_inference(args: argparse.Namespace):
    """Run the inference pipeline using the provided arguments."""
    import numpy as np
    import ray

    from verl import DataProto
    from verl.experimental.agent_loop import AgentLoopManager
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

    validate_required_args(args)
    args.model_path = validate_model_path(args.model_path, allow_remote_model=args.allow_remote_model)

    # 1. Init Ray
    init_ray(ray, args.ray_temp_dir)

    # 2. Init rollout manager
    logger.info("Initializing configuration and AgentLoopManager...")
    config = init_config(args)
    agent_loop_manager = AgentLoopManager.create(config=config)

    # 3. Load dataset
    samples = load_samples(
        args.data_path,
        cache_dir=args.cache_dir,
        dataset_split=args.dataset_split,
        max_samples=args.max_samples,
        deployment=args.deployment,
        local_image_arch=args.local_image_arch,
        local_image_dir=args.local_image_dir,
        local_image_namespace=args.local_image_namespace,
    )

    # 4. Prepare batch data
    logger.info("Preparing data batch...")
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array([sample["prompt"] for sample in samples], dtype=object),
            "agent_name": np.array([sample["agent_name"] for sample in samples], dtype=object),
            "tools_kwargs": np.array([sample["extra_info"]["tools_kwargs"] for sample in samples], dtype=object),
        },
        meta_info={"validate": True},
    ).repeat(config.actor_rollout_ref.rollout.n)

    # 5. Generate sequences
    logger.info("Starting sequence generation...")
    size_divisor = config.actor_rollout_ref.rollout.agent.num_workers
    batch_padded, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
    output_padded = agent_loop_manager.generate_sequences(batch_padded)
    output = unpad_dataproto(output_padded, pad_size=pad_size)

    # 6. Process results
    rm_scores = output.batch["rm_scores"].sum(dim=-1).tolist()
    mean_score = np.mean(rm_scores)

    logger.info(f"Generation completed. Mean RM Score: {mean_score:.4f}")
    print(f"\n=> Mean RM Score: {mean_score:.4f}\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Uni-Agent Inference Runner")

    # Input / Output configs
    parser.add_argument(
        "--data-path",
        type=str,
        default=_env_str("UNI_AGENT_DATA_PATH", "DATA_PATH"),
        help="Path to the input dataset (Parquet format).",
    )
    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        default=_env_str("UNI_AGENT_MODEL_PATH", "MODEL_PATH"),
        help="Path to the local model checkpoint.",
    )
    parser.add_argument(
        "--allow-remote-model",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("UNI_AGENT_ALLOW_REMOTE_MODEL", "ALLOW_REMOTE_MODEL", default=False),
        help="Allow --model-path to be a remote model ID instead of an existing local path.",
    )
    parser.add_argument(
        "--agent-config-path",
        type=str,
        default=_env_str("UNI_AGENT_AGENT_CONFIG_PATH", "AGENT_CONFIG_PATH"),
        help="Path to the agent loop configuration YAML.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=_env_str("UNI_AGENT_HF_CACHE_DIR", "HF_DATASETS_CACHE"),
        help="Writable Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--ray-temp-dir",
        type=str,
        default=_env_str("UNI_AGENT_RAY_TMP_DIR", "RAY_TMPDIR"),
        help="Directory for local Ray session temp/log files; use 'none' to let Ray choose.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default=_env_str("UNI_AGENT_DATASET_SPLIT", "DATASET_SPLIT", default="train"),
        help="Dataset split name to load from parquet.",
    )
    parser.add_argument(
        "--deployment",
        type=str,
        default=_env_str("DEPLOYMENT", "UNI_AGENT_DEPLOYMENT", default="local").lower(),
        choices=["local", "modal", "vefaas"],
        help="Deployment mode used for sample preprocessing.",
    )
    parser.add_argument(
        "--local-image-arch",
        type=str,
        default=_env_str("LOCAL_DEPLOYMENT_IMAGE_ARCH", "UNI_AGENT_LOCAL_IMAGE_ARCH", default="auto"),
        choices=["auto", "keep", "x86_64", "arm64"],
        help="Architecture used when rewriting SWE-Bench images for local deployment.",
    )
    parser.add_argument(
        "--local-image-dir",
        type=str,
        default=_env_str(
            "LOCAL_DEPLOYMENT_IMAGE_DIR",
            "UNI_AGENT_LOCAL_SIF_IMAGE_DIR",
            "UNI_AGENT_LOCAL_IMAGE_DIR",
        ),
        help="Directory containing prebuilt local SIF images; use 'none' to disable SIF lookup.",
    )
    parser.add_argument(
        "--local-image-namespace",
        type=str,
        default=_env_str("LOCAL_DEPLOYMENT_IMAGE_NAMESPACE", "UNI_AGENT_LOCAL_IMAGE_NAMESPACE"),
        help="Optional image namespace used after public ARM64/SIF lookup fails.",
    )

    # Inference parameters
    parser.add_argument(
        "--max-turns",
        type=int,
        default=_env_int("UNI_AGENT_MAX_TURNS", "MAX_TURNS", default=300),
        help="Maximum number of interaction turns per episode.",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=_env_int("UNI_AGENT_PROMPT_LENGTH", "PROMPT_LENGTH", default=4096),
        help="Maximum prompt length (tokens).",
    )
    parser.add_argument(
        "--response-length",
        type=int,
        default=_env_int("UNI_AGENT_RESPONSE_LENGTH", "RESPONSE_LENGTH", default=131072),
        help="Maximum response length (tokens).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=_env_int("UNI_AGENT_MAX_MODEL_LEN", "MAX_MODEL_LEN", default=262144),
        help="Maximum model context length.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_env_float("UNI_AGENT_TEMPERATURE", "TEMPERATURE", default=0.8),
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=_env_float("UNI_AGENT_TOP_P", "TOP_P", default=0.9),
        help="Sampling top-p (nucleus sampling).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=_env_int("UNI_AGENT_N", "N", default=1),
        help="Number of rollouts per prompt (N).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=_env_int("UNI_AGENT_MAX_SAMPLES", "MAX_SAMPLES", default=-1),
        help="Max number of samples to run (default -1). Use -1 for no limit (full dataset).",
    )

    # Execution / Engine configs
    parser.add_argument(
        "--engine",
        type=str,
        default=_env_str("UNI_AGENT_ENGINE", "ENGINE", default="vllm"),
        choices=["vllm", "sglang"],
        help="Inference engine backend (e.g., vllm or sglang).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=_env_int("UNI_AGENT_NUM_WORKERS", "NUM_WORKERS", default=8),
        help="Number of agent rollout workers.",
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=_env_int("UNI_AGENT_NNODES", "NNODES", default=1),
        help="Number of nodes to run the job.",
    )
    parser.add_argument(
        "--n-gpus-per-node",
        type=int,
        default=_env_int("UNI_AGENT_N_GPUS_PER_NODE", "N_GPUS_PER_NODE", default=_default_visible_gpu_count()),
        help="Number of GPUs per node.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp",
        type=int,
        default=_env_int(
            "UNI_AGENT_TENSOR_PARALLEL_SIZE",
            "TENSOR_PARALLEL_SIZE",
            "TP",
            default=_default_visible_gpu_count(),
        ),
        help="Tensor parallel size for the model.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=_env_float("UNI_AGENT_GPU_MEMORY_UTILIZATION", "GPU_MEMORY_UTILIZATION", default=0.7),
        help="Rollout GPU memory utilization.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
