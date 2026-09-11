"""Parallel agent inference over a verl-launched engine, through the training path.

Same job as ``parallel_infer_api.py`` (run each row's task, report a score), but verl
brings the engine up and rollouts flow through the *exact* training stack -- the agent
framework adapter + TransferQueue (TQ):

    verl LLMServerManager (vLLM / SGLang)
    ->  AgentFrameworkRolloutAdapter.generate_sequences   (fire-and-forget -> TQ)
          ->  Gateway sessions (per-session OpenAI-compatible endpoints)
          ->  uni_agent.framework.task_runner.run_task  ->  uni_agent task
    ->  per-trajectory records written to TransferQueue

The per-sample score is the trainer's own ``rm_scores`` read back from TQ: ``run_task``
returns a typed task result, and the framework writes its reward as ``reward_score`` --
no external reward model. Fan-out is ``rollout.n`` (``--n``),
with no resolved/wrong-answer/timeout bucketing (just mean ``rm_scores``).

Example (single node, 4-way tensor parallel)::

    python examples/inference/parallel_infer_verl.py \
        --data-path ~/data/swe_agent/swe_bench_verified.parquet \
        --model-path ~/models/Qwen3-Coder-30B-A3B-Instruct \
        --tool-parser qwen3_coder --tensor-parallel-size 4 \
        --task-config examples/quickstart/inference/task_config_react.yaml --limit 8

``--task-config`` is required (same YAML shape as ``parallel_infer_api.py``); the policy
endpoint is the gateway session, bound by the runner, not a flag.
"""

import argparse
import copy
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import numpy as np
import ray
from datasets import load_dataset
from omegaconf import OmegaConf

import verl

try:
    import transfer_queue as tq
except ImportError:  # fall back to verl's shim (mock raises a clear error if TQ is missing)
    from verl.utils.transferqueue_utils import tq

from uni_agent.framework.entry import AgentFrameworkRolloutAdapter
from uni_agent.tasks import TaskConfigResolver
from verl.utils import tensordict_utils as tu
from verl.workers.rollout.llm_server import LLMServerManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


GLOBAL_CONCURRENCY = int(os.getenv("GLOBAL_CONCURRENCY", 128))
PARTITION_ID = "val"

DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_RESPONSE_LENGTH = 65536
DEFAULT_PROMPT_LENGTH = 4096


def _rule(text: str = "", width: int = 50, ch: str = "-") -> str:
    """A centered-title horizontal rule."""
    if not text:
        return ch * width
    pad = max(0, width - len(text) - 2)
    return f"{ch * (pad // 2)} {text} {ch * (pad - pad // 2)}"


def init_config(args: argparse.Namespace, *, task_configs: list[dict], served_model_name: str):
    """Compose verl's ``ppo_trainer`` config and override the engine + framework knobs."""
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(verl.__file__).resolve().parent / "trainer" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")

    rollout = config.actor_rollout_ref.rollout

    model_cfgs = [entry.get("agent", {}).get("model", {}) for entry in task_configs]
    temperature = model_cfgs[0].get("temperature", DEFAULT_TEMPERATURE)
    top_p = model_cfgs[0].get("top_p", DEFAULT_TOP_P)
    top_k = model_cfgs[0].get("top_k", -1)
    rollout.temperature = temperature
    rollout.top_p = top_p
    rollout.top_k = top_k
    rollout.val_kwargs.temperature = temperature
    rollout.val_kwargs.top_p = top_p
    rollout.val_kwargs.top_k = top_k
    rollout.val_kwargs.do_sample = True

    # response_length = the agent's episode token budget (max_total_tokens: the full
    # prompt+gen context the loop may consume); DEFAULT_RESPONSE_LENGTH is the fallback.
    max_total_tokens = max(
        (m.get("max_total_tokens", DEFAULT_RESPONSE_LENGTH) for m in model_cfgs),
        default=DEFAULT_RESPONSE_LENGTH,
    )
    response_length = int(max_total_tokens)

    # Fan-out: the framework runs rollout.n gateway sessions per prompt.
    rollout.n = max(1, args.n)
    rollout.val_kwargs.n = rollout.n

    # Hardware.
    rollout.nnodes = args.nnodes
    rollout.n_gpus_per_node = args.n_gpus_per_node
    config.trainer.nnodes = args.nnodes
    config.trainer.n_gpus_per_node = args.n_gpus_per_node

    # Model + engine.
    config.actor_rollout_ref.model.path = os.path.expanduser(args.model_path)
    rollout.name = args.engine
    rollout.mode = "async"
    # Standalone inference has no trainer to broadcast weights.
    rollout.load_format = "auto"
    rollout.prompt_length = DEFAULT_PROMPT_LENGTH
    rollout.response_length = response_length
    rollout.tensor_model_parallel_size = args.tensor_parallel_size
    rollout.gpu_memory_utilization = args.gpu_memory_utilization
    if args.max_model_len is not None:
        rollout.max_model_len = args.max_model_len
    if args.max_num_seqs is not None:
        rollout.max_num_seqs = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        rollout.max_num_batched_tokens = args.max_num_batched_tokens
    if args.enforce_eager:
        rollout.enforce_eager = True
    if args.enable_chunked_prefill:
        rollout.enable_chunked_prefill = True
    rollout.calculate_log_probs = True
    rollout.enable_rollout_routing_replay = args.enable_rollout_routing_replay
    rollout.disable_log_stats = False
    rollout.free_cache_engine = args.free_cache_engine
    OmegaConf.update(config, "actor_rollout_ref.rollout.enable_sleep_mode", False, force_add=True)

    if args.language_model_only:
        OmegaConf.update(
            config,
            "actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only",
            True,
            force_add=True,
        )
    if args.cudagraph_mode is not None:
        OmegaConf.update(
            config,
            "actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode",
            args.cudagraph_mode,
            force_add=True,
        )
    if args.mamba_cache_mode is not None:
        OmegaConf.update(
            config,
            "actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_mode",
            args.mamba_cache_mode,
            force_add=True,
        )
    if args.enable_cpu_binding:
        OmegaConf.update(
            config,
            "actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding",
            True,
            force_add=True,
        )
    if args.async_scheduling:
        OmegaConf.update(
            config,
            "actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling",
            True,
            force_add=True,
        )
    if args.multi_turn:
        OmegaConf.update(config, "actor_rollout_ref.rollout.multi_turn.enable", True, force_add=True)
        if args.max_assistant_turns is not None:
            OmegaConf.update(
                config,
                "actor_rollout_ref.rollout.multi_turn.max_assistant_turns",
                args.max_assistant_turns,
                force_add=True,
            )
        if args.max_parallel_calls is not None:
            OmegaConf.update(
                config,
                "actor_rollout_ref.rollout.multi_turn.max_parallel_calls",
                args.max_parallel_calls,
                force_add=True,
            )

    # Gateway tool-call parser: the gateway decodes tool calls from raw tokens, so
    # this must match the model's chat template (the analog of vLLM's
    # --tool-call-parser, e.g. qwen3_coder for Qwen3-Coder, hermes for Qwen3).
    OmegaConf.update(config, "actor_rollout_ref.rollout.multi_turn.format", args.tool_parser, force_add=True)

    agent_framework_cfg = {
        "gateway_count": args.gateway_count,
        "agent_runners": {
            "task": {
                "runner_fqn": "uni_agent.framework.task_runner.run_task",
                "dispatch_mode": "ray_task",
                "max_concurrent_sessions": max(0, args.concurrency),
                "runner_kwargs": {
                    "task_config_path": args.task_config,
                    "model_name": served_model_name,
                },
            }
        },
    }
    agent_framework_cfg["log_dir"] = args.log_dir
    OmegaConf.update(config, "actor_rollout_ref.rollout.custom.agent_framework", agent_framework_cfg, force_add=True)

    # TransferQueue carries the rollout trajectories (and their rm_scores).
    OmegaConf.update(config, "transfer_queue.enable", True, force_add=True)

    # Data.
    config.data.return_raw_chat = True
    config.data.max_prompt_length = DEFAULT_PROMPT_LENGTH
    config.data.max_response_length = response_length

    return config


def _build_prompts(samples: list, uids: list, *, artifact_dir: str | None = None):
    """Assemble the TensorDict batch the framework's ``generate_sequences`` expects."""
    tools_kwargs = []
    for sample in samples:
        sample_tools_kwargs = copy.deepcopy(sample["extra_info"]["tools_kwargs"])
        task = sample_tools_kwargs.get("task")
        if artifact_dir and isinstance(task, dict):
            agent = task.get("agent")
            if isinstance(agent, dict) and agent.get("name") == "openclaw":
                agent["artifact_dir"] = artifact_dir
        tools_kwargs.append(sample_tools_kwargs)
    return tu.get_tensordict(
        tensor_dict={
            "raw_prompt": [sample.get("prompt") for sample in samples],
            "uid": list(uids),
            "tools_kwargs": tools_kwargs,
        },
        non_tensor_dict={"global_steps": None, "validate": True},
    )


def _read_rm_scores(uids: list, *, partition_id: str = PARTITION_ID) -> dict:
    """Read each session's final trajectory back from TQ and score it."""
    input_uids = set(uids)
    listing = tq.kv_list() or {}
    partition = listing.get(partition_id, {}) or {}

    # (uid, session) -> (max_index, key); also collect every key we touch for cleanup.
    final: dict[tuple[str, str], tuple[int, str]] = {}
    traj_keys: list[str] = []
    uid_status: dict[str, str] = {}
    for key, tag in partition.items():
        tag = tag or {}
        parts = key.rsplit("_", 2)
        if len(parts) != 3:
            # uid-level status marker (uid has no underscores: it is a uuid4 hex-with-dashes).
            if key in input_uids:
                uid_status[key] = tag.get("status")
            continue
        uid, session, index_str = parts
        if uid not in input_uids or tag.get("status") != "success":
            continue
        try:
            index = int(index_str)
        except ValueError:
            continue
        traj_keys.append(key)
        session_key = (uid, session)
        if session_key not in final or final[session_key][0] < index:
            final[session_key] = (index, key)

    # Deterministic order so scores align with the (uid, session) they came from.
    final_items = sorted(final.items())
    final_keys = [key for _, (_, key) in final_items]
    final_sessions = [session_key for session_key, _ in final_items]

    per_uid: dict[str, list[float]] = defaultdict(list)
    scores: list[float] = []
    if final_keys:
        data = tq.kv_batch_get(keys=final_keys, partition_id=partition_id, select_fields=["rm_scores"])
        scores = [float(s) for s in data["rm_scores"].sum(dim=-1).tolist()]
        for (uid, _session), score in zip(final_sessions, scores, strict=True):
            per_uid[uid].append(score)

    uid_keys = [uid for uid in input_uids if uid in uid_status]
    return {
        "scores": scores,
        "per_uid": dict(per_uid),
        "uid_status": uid_status,
        "final_keys": final_keys,
        "traj_keys": traj_keys,
        "uid_keys": uid_keys,
    }


def _report(
    read: dict, *, wall: float, num_prompts: int, n: int, args: argparse.Namespace, served_model_name: str
) -> None:
    """Print the mean-rm_scores summary and optionally persist a JSON result file."""
    scores = read["scores"]
    per_uid = read["per_uid"]
    uid_status = read["uid_status"]

    expected = num_prompts * n
    num_scored = len(scores)
    mean_score = float(np.mean(scores)) if scores else 0.0
    # Per-prompt score = mean over that prompt's sessions; then averaged over prompts.
    prompt_means = [float(np.mean(v)) for v in per_uid.values() if v]
    mean_over_prompts = float(np.mean(prompt_means)) if prompt_means else 0.0
    failed_uids = sum(1 for status in uid_status.values() if status != "finished")

    summary = "\n".join(
        [
            "",
            _rule("inference summary"),
            f"  mean rm_score      {mean_score:>8.4f}   (over {num_scored} sessions)",
            f"  mean over prompts  {mean_over_prompts:>8.4f}   (over {len(prompt_means)} prompts)",
            f"  scored sessions    {num_scored:>4} / {expected:<4} ({num_prompts} prompts x n={n})",
            f"  failed prompts     {failed_uids:>4}",
            _rule(f"wall {wall:.1f}s"),
            "",
        ]
    )
    print(summary)

    if args.result_path:
        result_path = os.path.expanduser(args.result_path)
        os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
        payload = {
            "model_path": os.path.expanduser(args.model_path),
            "served_model_name": served_model_name,
            "data_path": os.path.expanduser(args.data_path),
            "task_config": args.task_config,
            "n": n,
            "num_prompts": num_prompts,
            "num_scored_sessions": num_scored,
            "mean_rm_score": mean_score,
            "mean_rm_score_over_prompts": mean_over_prompts,
            "scores": scores,
            "scores_by_uid": per_uid,
        }
        with open(result_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"wrote result file to: {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel agent inference over a verl-launched engine (framework + TQ)."
    )

    # Input / output.
    parser.add_argument(
        "--data-path",
        default=os.getenv("DATA_PATH", os.path.expanduser("~/data/swe_agent/swe_bench_verified.parquet")),
        help="Path to the input dataset (Parquet format).",
    )
    parser.add_argument(
        "--model-path",
        "--model",
        dest="model_path",
        default=os.path.expanduser("~/models/Qwen3-Coder-30B-A3B-Instruct"),
        help="Local model checkpoint the engine loads.",
    )
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Model name sent on chat-completions requests (default: basename of --model-path).",
    )
    parser.add_argument(
        "--task-config",
        required=True,
        help="Path to a YAML task config: one ``- name: ...`` entry or a list of them (required). "
        "run_task routes each row to the entry whose 'name' matches the row's task; all agent/model "
        "knobs (sampling, max_total_tokens, max_steps, ...) come from it. The endpoint is bound to the "
        "gateway session.",
    )
    parser.add_argument(
        "--result-path",
        default=None,
        help="Optional path to write a JSON result file (mean rm_score and per-session scores).",
    )
    parser.add_argument(
        "--limit",
        "--max-samples",
        dest="limit",
        type=int,
        default=None,
        help="Only run the first N samples (smoke testing); omit for the full dataset.",
    )

    parser.add_argument(
        "--n", type=int, default=1, help="Rollout sessions per instance (rollout.n; scores average over all)."
    )

    # Engine / hardware.
    parser.add_argument(
        "--engine",
        default="vllm",
        choices=["vllm", "sglang"],
        help="Inference engine backend.",
    )
    parser.add_argument(
        "--enable-rollout-routing-replay",
        action="store_true",
        help="Enable R3 routed-expert capture in the rollout engine for routing-replay diagnostics.",
    )
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes to run the engine on.")
    parser.add_argument("--n-gpus-per-node", type=int, default=8, help="Number of GPUs per node.")
    parser.add_argument(
        "--tensor-parallel-size", "--tp", dest="tensor_parallel_size", type=int, default=4, help="Tensor parallel size."
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="Engine GPU memory fraction.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Optional vLLM max model length.")
    parser.add_argument("--max-num-seqs", type=int, default=None, help="Optional vLLM max concurrent sequences.")
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=None, help="Optional vLLM max batched tokens."
    )
    parser.add_argument(
        "--enforce-eager", action="store_true", help="Disable torch.compile and CUDA graphs in vLLM."
    )
    parser.add_argument(
        "--enable-chunked-prefill", action="store_true", help="Enable vLLM chunked prefill."
    )
    parser.add_argument(
        "--language-model-only", action="store_true", help="Enable text-only vLLM language-model mode."
    )
    parser.add_argument(
        "--free-cache-engine", action="store_true", help="Free the rollout cache engine between requests."
    )
    parser.add_argument("--cudagraph-mode", default=None, help="Optional vLLM cudagraph mode.")
    parser.add_argument("--mamba-cache-mode", default=None, help="Optional vLLM mamba cache mode.")
    parser.add_argument(
        "--enable-cpu-binding", action="store_true", help="Enable vLLM CPU binding in additional_config."
    )
    parser.add_argument(
        "--async-scheduling", action="store_true", help="Enable vLLM async scheduling in additional_config."
    )
    parser.add_argument("--multi-turn", action="store_true", help="Enable verl multi-turn tool rollout.")
    parser.add_argument(
        "--max-assistant-turns", type=int, default=None, help="Optional multi-turn assistant turn limit."
    )
    parser.add_argument(
        "--max-parallel-calls", type=int, default=None, help="Optional maximum parallel tool calls per turn."
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Optional host directory for exported OpenClaw trajectory.json files.",
    )
    parser.add_argument(
        "--gateway-count",
        type=int,
        default=4,
        help="Number of gateway actors fronting the engine (each serves many concurrent sessions).",
    )
    parser.add_argument(
        "--tool-parser",
        default=os.getenv("TOOL_PARSER", "qwen3_coder"),
        help="Gateway tool-call parser; MUST match the model's chat template (e.g. qwen3_coder, hermes).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=GLOBAL_CONCURRENCY,
        help="Max in-flight gateway sessions for the runner (runner.max_concurrent_sessions; env GLOBAL_CONCURRENCY).",
    )
    parser.add_argument(
        "--log-dir",
        default=os.getenv("UNI_AGENT_LOG_DIR", "/tmp/uni_agent_logs"),
        help="Root directory for per-session logs and trajectories; use an empty value to disable.",
    )

    args = parser.parse_args()

    ray.init()

    resolver = TaskConfigResolver.from_file(args.task_config)
    served_model_name = args.served_model_name or os.path.basename(os.path.expanduser(args.model_path).rstrip("/"))

    dataset = load_dataset("parquet", data_files=args.data_path, split="train")
    samples = dataset.to_list()
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        logger.warning("no samples selected; exiting")
        return
    n = max(1, args.n)

    task_configs = list(resolver.defaults_by_name.values())

    logger.info(f"loaded {len(samples)} prompts (x n={n} sessions each) from {args.data_path}")

    # 1. TransferQueue + verl inference engine (Ray auto-inits via the actors below).
    logger.info("initializing configuration, TransferQueue, and LLMServerManager...")
    config = init_config(args, task_configs=task_configs, served_model_name=served_model_name)
    tq.init(config.transfer_queue)
    llm_server_manager = LLMServerManager.create(config=config)

    # 2. Framework rollout adapter over the engine.
    adapter = AgentFrameworkRolloutAdapter.create(
        config=config,
        llm_client=llm_server_manager.get_client(),
    )

    # 3. Submit the batch and wait for every trajectory to land in TQ.
    uids = [str(uuid4()) for _ in samples]
    artifact_dir = os.path.expanduser(args.artifact_dir) if args.artifact_dir else None
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
    prompts = _build_prompts(samples, uids, artifact_dir=artifact_dir)
    logger.info("starting inference...")
    begin_time = time.time()
    adapter.generate_sequences_and_wait(prompts)
    wall = time.time() - begin_time

    # 4. Read rm_scores back from TQ and report.
    read = _read_rm_scores(uids, partition_id=PARTITION_ID)
    _report(read, wall=wall, num_prompts=len(samples), n=n, args=args, served_model_name=served_model_name)


if __name__ == "__main__":
    main()
