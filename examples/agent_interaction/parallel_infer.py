import argparse
import logging
import os
import shutil
import socket
import time
from pathlib import Path

import numpy as np
import ray
from datasets import load_dataset
from omegaconf import DictConfig
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

import verl
from verl import DataProto
from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

# Setup basic logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=os.getenv("VERL_LOGGING_LEVEL", "INFO")
)
logger = logging.getLogger(__name__)


def init_config(args: argparse.Namespace) -> DictConfig:
    """Initialize the configuration from hydra and override with command-line arguments."""
    from hydra import compose, initialize_config_dir

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
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.9

    # Data configs
    config.data.return_raw_chat = True
    config.data.max_prompt_length = args.prompt_length
    config.data.max_response_length = args.response_length

    return config


def _build_batch(samples: list, n_per_prompt: int) -> DataProto:
    """Pack a list of sample dicts into a DataProto and repeat each prompt N times (GRPO-style)."""
    return DataProto(
        non_tensor_batch={
            "raw_prompt": np.array([s["prompt"] for s in samples], dtype=object),
            "agent_name": np.array([s["agent_name"] for s in samples], dtype=object),
            "tools_kwargs": np.array([s["extra_info"]["tools_kwargs"] for s in samples], dtype=object),
        },
        meta_info={"validate": True},
    ).repeat(n_per_prompt)


def _select_step_samples(
    all_samples: list,
    step_idx: int,
    train_batch_size: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> list:
    """Pick `train_batch_size` prompts for this step.

    With shuffle on, samples are drawn without replacement from the full dataset
    via a freshly permuted index per step (mimics dataloader behavior).
    Without shuffle, take a sequential slice and wrap around if needed.
    """
    if shuffle:
        # Permute fresh per step so each batch has a distinct, reproducible draw
        idx = rng.permutation(len(all_samples))[:train_batch_size]
    else:
        start = (step_idx * train_batch_size) % len(all_samples)
        end = start + train_batch_size
        if end <= len(all_samples):
            idx = np.arange(start, end)
        else:
            idx = np.concatenate([np.arange(start, len(all_samples)), np.arange(0, end - len(all_samples))])
    return [all_samples[int(i)] for i in idx]


@ray.remote(num_cpus=1)
def _move_local_to_hdfs(local_root: str, hdfs_dst: str) -> tuple[str, int, float]:
    """Move all `<run_id>/` subdirs from `local_root` to `hdfs_dst` on the node this task lands on.

    Returns `(host, num_moved, elapsed_secs)`. Idempotent: targets that already exist are skipped.
    """
    host = socket.gethostname()
    t0 = time.time()
    src = Path(local_root)
    if not src.exists():
        return (host, 0, 0.0)
    dst = Path(hdfs_dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for run_dir in src.iterdir():
        if not run_dir.is_dir():
            continue
        target = dst / run_dir.name
        if target.exists():
            continue
        try:
            shutil.move(str(run_dir), str(target))
            n += 1
        except OSError:
            # Best-effort; don't fail the whole drain on one bad dir.
            continue
    return (host, n, time.time() - t0)


def _drain_workers_to_hdfs(local_root: str, hdfs_dst: str) -> None:
    """Pin one upload task per live Ray node; wait for all to finish."""
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    if not nodes:
        logger.warning("No live Ray nodes found; skipping drain to %s", hdfs_dst)
        return
    futs = [
        _move_local_to_hdfs.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=n["NodeID"], soft=False)
        ).remote(local_root, hdfs_dst)
        for n in nodes
    ]
    results = ray.get(futs)
    total = sum(c for _, c, _ in results)
    logger.info(
        "Drained %d trajectories to %s (from %d node(s))",
        total,
        hdfs_dst,
        len(results),
    )
    for host, c, elapsed in results:
        logger.info("  %s: %d moved in %.1fs", host, c, elapsed)


def run_inference(args: argparse.Namespace):
    """Run the inference pipeline using the provided arguments."""
    # 1. Init Ray
    ray.init()

    # 2. Init rollout manager
    logger.info("Initializing configuration and AgentLoopManager...")
    config = init_config(args)
    agent_loop_manager = AgentLoopManager.create(config=config)

    # 3. Load dataset
    data_path = os.path.expanduser(args.data_path)
    logger.info(f"Loading dataset from: {data_path}")
    all_samples = load_dataset("parquet", data_files=data_path, split="train").to_list()

    # Optional global cap; keeps the full pool when 0 / negative
    if args.max_samples > 0:
        all_samples = all_samples[: args.max_samples]
        logger.info("Using first %d samples (--max-samples=%d)", len(all_samples), args.max_samples)

    # Resolve effective per-step batch size
    if args.train_batch_size > 0:
        train_batch_size = args.train_batch_size
    else:
        # Fallback: whole-dataset single batch (legacy behavior)
        train_batch_size = len(all_samples)

    n_per_prompt = config.actor_rollout_ref.rollout.n
    size_divisor = config.actor_rollout_ref.rollout.agent.num_workers
    rng = np.random.default_rng(args.seed)
    # Advance rng past the steps already done so the (seed, start_step) pair
    # reproduces the same samples as a continuous run.
    if args.shuffle:
        for _ in range(args.start_step):
            rng.permutation(len(all_samples))

    # Per-step drain target: read worker-local log_dir from the agent config so we
    # can move <log_dir>/<run_id>/ to HDFS at each step boundary.
    local_log_dir: str | None = None
    if args.hdfs_root:
        import yaml

        with open(os.path.expanduser(args.agent_config_path)) as f:
            agents = yaml.safe_load(f)
        agent_cfg = agents[0] if isinstance(agents, list) else agents
        local_log_dir = agent_cfg.get("log_dir")
        if not local_log_dir:
            raise ValueError(
                f"--hdfs-root requires `log_dir` to be set in {args.agent_config_path}"
            )
        logger.info("Will drain '%s' -> '%s' after each step", local_log_dir, args.hdfs_root)

    logger.info(
        "Running %d step(s): train_batch_size=%d prompts x n=%d responses = %d trajectories/step "
        "(num_workers=%d)",
        args.num_steps,
        train_batch_size,
        n_per_prompt,
        train_batch_size * n_per_prompt,
        size_divisor,
    )

    all_step_scores: list[float] = []
    for offset in range(args.num_steps):
        step_idx = args.start_step + offset
        batch_tag = f"step_{step_idx:03d}"

        step_samples = _select_step_samples(
            all_samples=all_samples,
            step_idx=step_idx,
            train_batch_size=train_batch_size,
            rng=rng,
            shuffle=args.shuffle,
        )
        logger.info("[%s] Preparing batch from %d prompts...", batch_tag, len(step_samples))
        batch = _build_batch(step_samples, n_per_prompt=n_per_prompt)
        batch_padded, pad_size = pad_dataproto_to_divisor(batch, size_divisor)

        t0 = time.time()
        logger.info("[%s] Starting sequence generation...", batch_tag)
        output_padded = agent_loop_manager.generate_sequences(batch_padded)
        wall = time.time() - t0
        output = unpad_dataproto(output_padded, pad_size=pad_size)

        rm_scores = output.batch["rm_scores"].sum(dim=-1).tolist()
        mean_score = float(np.mean(rm_scores))
        all_step_scores.append(mean_score)
        logger.info("[%s] Done in %.1fs. Mean RM Score: %.4f", batch_tag, wall, mean_score)
        print(f"\n=> [{batch_tag}] wall={wall:.1f}s | Mean RM Score: {mean_score:.4f}\n")

        if args.hdfs_root and local_log_dir:
            hdfs_step_dir = f"{args.hdfs_root.rstrip('/')}/{batch_tag}"
            _drain_workers_to_hdfs(local_root=local_log_dir, hdfs_dst=hdfs_step_dir)

    overall_mean = float(np.mean(all_step_scores)) if all_step_scores else 0.0
    logger.info("All %d steps completed. Overall mean RM Score: %.4f", args.num_steps, overall_mean)
    print(f"\n=> Overall Mean RM Score across {args.num_steps} steps: {overall_mean:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Uni-Agent Inference Runner")

    # Input / Output configs
    parser.add_argument(
        "--data-path",
        type=str,
        default="~/data/swe_agent/swe_bench_verified.parquet",
        help="Path to the input dataset (Parquet format).",
    )
    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        default="~/models/Qwen3-Coder-30B-A3B-Instruct",
        help="Path to the local model checkpoint.",
    )
    parser.add_argument(
        "--agent-config-path",
        type=str,
        default="examples/agent_interaction/agent_config.yaml",
        help="Path to the agent loop configuration YAML.",
    )

    # Inference parameters
    parser.add_argument("--max-turns", type=int, default=100, help="Maximum number of interaction turns per episode.")
    parser.add_argument("--prompt-length", type=int, default=4096, help="Maximum prompt length (tokens).")
    parser.add_argument("--response-length", type=int, default=65536, help="Maximum response length (tokens).")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Sampling top-p (nucleus sampling).")
    parser.add_argument("--n", type=int, default=1, help="Number of rollouts per prompt (N), GRPO group size.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Cap the dataset pool size before stepping. Use -1 for no limit (full dataset).",
    )

    # Multi-step batch driver: emulate training-style batches so we can sample
    # many batches and analyze tail latency per batch.
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=-1,
        help=(
            "Prompts per step (matches training's train_prompt_bsz). "
            "Total trajectories per step = train_batch_size * n. "
            "Use -1 to fall back to legacy behavior (entire pool as one batch)."
        ),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1,
        help="Number of sequential batches (steps) to run. Each step is fully completed before the next starts.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Reshuffle the dataset per step. Without this flag, sequential slices (wrapping if needed) are used.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for --shuffle sampling.")
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help=(
            "Skip the first N step draws so a subsequent run with the same --seed "
            "produces steps [start, start+num_steps). On-disk batch tags are also "
            "shifted by start-step so HDFS dirs continue from step_<start>."
        ),
    )
    parser.add_argument(
        "--hdfs-root",
        type=str,
        default="",
        help=(
            "If set, after each step the driver pins a ray task on every live node "
            "that moves `<log_dir>/<run_id>/` (configured in agent_config.yaml) to "
            "`<hdfs-root>/step_<idx>/`. Local /tmp is emptied per step. "
            "Empty string disables drain."
        ),
    )

    # Execution / Engine configs
    parser.add_argument(
        "--engine",
        type=str,
        default="vllm",
        choices=["vllm", "sglang"],
        help="Inference engine backend (e.g., vllm or sglang).",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="Number of agent rollout workers.")
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes to run the job.")
    parser.add_argument("--n-gpus-per-node", type=int, default=8, help="Number of GPUs per node.")
    parser.add_argument(
        "--tensor-parallel-size", "--tp", type=int, default=4, help="Tensor parallel size for the model."
    )

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
