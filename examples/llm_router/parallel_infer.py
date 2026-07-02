import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import ray
from datasets import load_dataset
from omegaconf import DictConfig

import verl
from verl import DataProto
from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.workers.rollout.llm_server import LLMServerManager

# Setup basic logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=os.getenv("VERL_LOGGING_LEVEL", "INFO")
)
logger = logging.getLogger(__name__)

# Ray's default idle-worker reaper (~10 s) kills agent workers between
# dispatch gaps, ending the job prematurely.  Use a very large threshold
# so long-running agent loops are not interrupted.
_RAY_IDLE_WORKER_TIMEOUT_MS = int(os.getenv("RAY_IDLE_WORKER_TIMEOUT_MS", str(2**30 - 1)))


def init_config(args: argparse.Namespace) -> DictConfig:
    """Initialize the configuration from hydra and override with command-line arguments."""
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(verl.__file__).resolve().parent / "trainer" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")

    # Override rollout configs
    config.actor_rollout_ref.rollout.agent.agent_loop_config_path = os.path.expanduser(args.agent_config_path)
    config.actor_rollout_ref.rollout.agent.num_workers = args.num_workers
    config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = args.max_turns
    config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls = 1

    # Router config (optional): enable an external router via plugin_extension.
    # When set, VeRL loads the YAML (router_class) and instantiates the router
    # as the rollout load balancer.  Unset = VeRL built-in router.
    if args.router_config_path:
        config.actor_rollout_ref.rollout.router.router_strategy = "plugin_extension"
        config.actor_rollout_ref.rollout.router.router_config_path = args.router_config_path

    # Sampling kwargs
    config.actor_rollout_ref.rollout.temperature = args.temperature
    config.actor_rollout_ref.rollout.top_p = args.top_p

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
    config.actor_rollout_ref.rollout.max_num_seqs = args.max_num_seqs
    if args.max_model_len is not None:
        config.actor_rollout_ref.rollout.max_model_len = args.max_model_len
    config.actor_rollout_ref.rollout.disable_log_stats = False  # expose engine metrics on /metrics endpoint

    # Data configs
    config.data.return_raw_chat = True
    config.data.max_prompt_length = args.prompt_length
    config.data.max_response_length = args.response_length

    # Optionally attach MooncakeStoreConnector for cross-replica KV sharing.
    if args.enable_mooncake:
        config.actor_rollout_ref.rollout.engine_kwargs = {
            "vllm": {
                "kv_transfer_config": {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "mooncake_config_path": args.mooncake_config_path,
                    },
                }
            }
        }

    return config


def run_inference(args: argparse.Namespace):
    """Run the inference pipeline using the provided arguments."""
    # 1. Init Ray — disable idle-worker reaper so agent workers survive
    # dispatch gaps (default ~10 s threshold would kill them prematurely).
    ray.init(_system_config={"idle_worker_killing_time_threshold_ms": _RAY_IDLE_WORKER_TIMEOUT_MS})

    # 2. Init rollout manager
    logger.info("Initializing configuration and AgentLoopManager...")
    config = init_config(args)
    llm_server_manager = LLMServerManager.create(config=config)
    agent_loop_manager = AgentLoopManager.create(
        config=config,
        llm_client=llm_server_manager.get_client(),
    )

    # 3. Load dataset
    data_path = os.path.expanduser(args.data_path)
    logger.info(f"Loading dataset from: {data_path}")
    samples = load_dataset("parquet", data_files=data_path, split="train").to_list()

    # Limit number of samples (-1 = no limit)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
        logger.info("Using first %d samples (--max-samples=%d)", len(samples), args.max_samples)

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
    mean_score = float(np.mean(rm_scores)) if len(rm_scores) > 0 else 0.0

    logger.info(f"Generation completed. Mean RM Score: {mean_score:.4f}")
    print(f"\n=> Mean RM Score: {mean_score:.4f}\n")

    # 7. Optionally persist a machine-readable result file (used by eval_checkpoints.py).
    if args.result_path:
        result_path = os.path.expanduser(args.result_path)
        os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
        result = {
            "model_path": os.path.expanduser(args.model_path),
            "data_path": data_path,
            "agent_config_path": os.path.expanduser(args.agent_config_path),
            "n": config.actor_rollout_ref.rollout.n,
            "num_samples": len(rm_scores),
            "mean_rm_score": mean_score,
            "rm_scores": rm_scores,
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Wrote result file to: {result_path}")

    return mean_score


def main():
    parser = argparse.ArgumentParser(description="Uni-Agent Inference Runner")

    # Input / Output configs
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the input dataset (Parquet format).",
    )
    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        default=None,
        help="Path to the local model checkpoint.",
    )
    parser.add_argument(
        "--agent-config-path",
        type=str,
        default=None,
        help="Path to the agent loop configuration YAML.",
    )
    parser.add_argument(
        "--result-path",
        type=str,
        default=None,
        help="Optional path to write a JSON result file (mean reward and per-rollout scores).",
    )
    parser.add_argument(
        "--router-config-path",
        type=str,
        default=None,
        help=(
            "Optional router config YAML (e.g. pkg://...). When set, enables "
            "plugin_extension router strategy; the YAML's router_class is "
            "instantiated as the rollout load balancer. Default None = VeRL "
            "built-in router."
        ),
    )

    # Inference parameters
    parser.add_argument("--max-turns", type=int, default=100, help="Maximum number of interaction turns per episode.")
    parser.add_argument("--prompt-length", type=int, default=4096, help="Maximum prompt length (tokens).")
    parser.add_argument("--response-length", type=int, default=8192, help="Maximum response length (tokens).")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Sampling top-p (nucleus sampling).")
    parser.add_argument("--n", type=int, default=1, help="Number of rollouts per prompt (N).")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Max number of samples to run. Use -1 for no limit (full dataset).",
    )

    # Execution / Engine configs
    parser.add_argument(
        "--engine",
        type=str,
        default="vllm",
        choices=["vllm", "sglang"],
        help="Inference engine backend (e.g., vllm or sglang).",
    )
    parser.add_argument("--num-workers", type=int, default=1, help="Number of agent rollout workers.")
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes to run the job.")
    parser.add_argument("--n-gpus-per-node", type=int, default=2, help="Number of GPUs per node.")
    parser.add_argument(
        "--tensor-parallel-size", "--tp", type=int, default=1, help="Tensor parallel size for the model."
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length (tokens). If unset the engine default is used.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of concurrent sequences per engine.",
    )
    parser.add_argument(
        "--enable-mooncake",
        action="store_true",
        help="Attach MooncakeStoreConnector for cross-replica KV sharing (a mooncake master must run separately).",
    )
    parser.add_argument(
        "--mooncake-config-path",
        type=str,
        default="mooncake_config.json",
        help="Path to the mooncake config JSON (used with --enable-mooncake).",
    )

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
