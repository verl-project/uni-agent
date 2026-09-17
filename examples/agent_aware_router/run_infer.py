"""Parallel agent inference over a verl-launched engine with the KV-cache-aware router.

vLLM replicas sit behind the kvcaware router (KV-cache hit rate + load-aware
dispatch), a gateway pool drives blackbox agent sessions in openyuanrong
sandboxes, and each task's reward is read back from the TransferQueue — no
trainer is started.

    verl LLMServerManager (vLLM, KVCAware router plugin)
    ->  AgentFrameworkRolloutAdapter.generate_sequences   (fire-and-forget -> TQ)
          ->  Gateway sessions (per-session OpenAI-compatible endpoints)
          ->  uni_agent.framework.task_runner.run_task  ->  uni_agent task
    ->  per-trajectory records written to TransferQueue

The per-sample score is the trainer's ``rm_scores`` read back from TQ:
``run_task`` (``report_reward=True``) posts the task reward to its session, and
the framework writes it as ``reward_score`` -- no external reward model. Fan-out
is ``rollout.n`` (``--n``).

The router is wired through verl's plugin mechanism (``rollout.router_config_path``),
which points at the packaged
``uni_agent/agent_aware_router/configs/agent_aware_router.yaml``. Runtime knob overrides
(``--load-threshold``) ride on ``rollout.custom.agent_framework.router`` — the balancer
reads that node from each server's rollout config at construction, so no router-YAML
rewrite is needed. ``--task-config`` selects the agent/sandbox per row (required; same YAML
shape as ``examples/inference/parallel_infer_verl.py``); the policy endpoint is
the gateway session, bound by the runner, not a flag.

KV-cache-aware knobs:

  --router-config-path  router YAML handed to verl via rollout.router_config_path
                        (default uni_agent/agent_aware_router/configs/agent_aware_router.yaml,
                        relative to the repo root)
  --kv-events           vLLM kv-events zmq publisher; the kvcaware collector's load signal
                        (retained-cache occupancy).
  --load-threshold      strategy[0] overrides; each falls back to the packaged YAML value.
  --max-num-seqs        engine max concurrent sequences.

Debug mode — env overrides for the internalized strategy knobs (no CLI flags;
see uni_agent/agent_aware_router/debug.py)::

    UNI_AGENT_ROUTER_DEBUG=1 \
    UNI_AGENT_ROUTER_SLOW_CUT=least-inflight \
    UNI_AGENT_ROUTER_OVERLOAD_MODE=None \
        python examples/agent_aware_router/run_infer.py ...   # pure sticky

Example (single node, 2-way tensor parallel, kvcaware router + kv-events)::

    python examples/agent_aware_router/run_infer.py \
        --data-path ~/data/swe_agent/swe_bench_verified_openyuanrong.parquet \
        --model-path ~/models/Qwen/Qwen3-8B \
        --task-config examples/agent_aware_router/task_config_mini_swe_agent.yaml \
        --tool-parser qwen3_coder --tensor-parallel-size 2 \
        --max-model-len 40960 --limit 1
"""

import argparse
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

from uni_agent.agent_aware_router import (
    server as _kv_events_server,  # noqa: F401  (import overrides the vllm rollout backend)
)
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

# Default router plugin YAML (FQN-injected), repo-root relative (pkg:// is no
# longer supported).
DEFAULT_ROUTER_CONFIG_PATH = "uni_agent/agent_aware_router/configs/agent_aware_router.yaml"

# Ray's default idle-worker reaper (~10 s) kills agent workers between dispatch
# gaps, ending the job prematurely.
_RAY_IDLE_WORKER_TIMEOUT_MS = int(os.getenv("RAY_IDLE_WORKER_TIMEOUT_MS", str(2**30 - 1)))


def _rule(text: str = "", width: int = 50, ch: str = "-") -> str:
    """A centered-title horizontal rule."""
    if not text:
        return ch * width
    pad = max(0, width - len(text) - 2)
    return f"{ch * (pad // 2)} {text} {ch * (pad - pad // 2)}"


# =====================================================================
# Router configuration overrides (plugin mechanism)
# =====================================================================


def _resolve_router_config_path(path: str) -> str:
    """Resolve a router config path to an absolute path.

    Plain filesystem paths only, repo-root relative OK — pkg:// URIs are no
    longer supported (upstream dropped them).
    """
    return os.path.abspath(path)


def init_config(args: argparse.Namespace, *, task_configs: list[dict], served_model_name: str):
    """Compose verl's ``ppo_trainer`` config with the KV-cache-aware router plugin and
    override the engine + framework knobs.

    This is the 144-verified ``run_infer.py``'s ``init_config`` with the
    router wired through verl's plugin mechanism (``rollout.router_config_path``) instead
    of the removed ``rollout/router@...=kvcaware`` Hydra group. The framework/TQ path
    (``rollout.custom.agent_framework``, ``transfer_queue``) is preserved untouched.
    """
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(verl.__file__).resolve().parent / "trainer" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")

    rollout = config.actor_rollout_ref.rollout

    model_cfgs = [entry.get("agent", {}).get("model", {}) for entry in task_configs]
    temperature = model_cfgs[0].get("temperature", DEFAULT_TEMPERATURE)
    top_p = model_cfgs[0].get("top_p", DEFAULT_TOP_P)
    rollout.temperature = temperature
    rollout.top_p = top_p
    rollout.val_kwargs.temperature = temperature
    rollout.val_kwargs.top_p = top_p

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
    rollout.agent.num_workers = args.num_workers
    rollout.tensor_model_parallel_size = args.tensor_parallel_size
    rollout.gpu_memory_utilization = args.gpu_memory_utilization
    # Cap the engine context. Must be >= the agent's per-episode token budget,
    # else vLLM rejects the request (HTTP 400, transcript overruns the limit).
    rollout.max_model_len = args.max_model_len
    # kvcaware additions: engine concurrency + expose metrics on /metrics for the
    # kvcaware collector (which polls them as one of its load signals).
    rollout.max_num_seqs = args.max_num_seqs
    rollout.disable_log_stats = False
    # Load real weights: verl's rollout.yaml defaults to ``load_format: dummy``
    # (random weights => garbage). Force ``auto`` (matches the 144-verified run).
    rollout.load_format = "auto"

    # Gateway tool-call parser: must match the model's chat template (qwen3_coder
    # for Qwen3-Coder, hermes for Qwen3).
    OmegaConf.update(config, "actor_rollout_ref.rollout.multi_turn.format", args.tool_parser, force_add=True)

    # vLLM engine kwargs: MFU metric (always on) + optional mooncake connector / kv-events.
    vllm_kwargs: dict = {"enable_mfu_metrics": True}
    if args.enable_mooncake:
        # Cross-replica KV sharing via mooncake (config via MOONCAKE_CONFIG_PATH env).
        # GPU build uses "MooncakeStoreConnector"; vllm-ascend uses "MooncakeConnectorStoreV1".
        mooncake_connector = "MooncakeConnectorStoreV1" if args.device == "ascend" else "MooncakeStoreConnector"
        vllm_kwargs["kv_transfer_config"] = {
            "kv_connector": mooncake_connector,
            "kv_role": "kv_both",
            "kv_connector_extra_config": {},
        }
    if args.kv_events:
        # vLLM kv-events (zmq publisher) — kvcaware load signal (retained-cache
        # occupancy). Ports are placeholders (the uni-agent kv-events server
        # assigns ephemeral per-replica blocks).
        vllm_kwargs["kv-events-config"] = {
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "topic": "kv-events",
            "endpoint": "tcp://*:0",
            "replay_endpoint": "tcp://*:0",
        }
    rollout.engine_kwargs = {"vllm": vllm_kwargs}

    # Point verl at the router plugin YAML directly — runtime knob overrides are
    # carried on rollout.custom.agent_framework.router (see below) and read by the
    # balancer at construction, so no temp-copy rewrite is required.
    rollout.router_config_path = _resolve_router_config_path(args.router_config_path)

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
                    "report_reward": True,
                },
            }
        },
    }
    if args.simulated_runner_fqn:
        # Swap the sandbox-backed runner for a test double (canned observations,
        # no container): the framework treats runners as interchangeable
        # AgentRunner-protocol callables.
        task_runner = agent_framework_cfg["agent_runners"]["task"]
        task_runner["runner_fqn"] = args.simulated_runner_fqn
        task_runner["runner_kwargs"] = {}
    # KV-cache-aware router knobs — the balancer overrides the packaged router
    # YAML with this node at construction (``custom.agent_framework.router``).
    agent_framework_cfg["router"] = {"load_threshold": args.load_threshold}
    agent_framework_cfg["log_dir"] = args.log_dir
    OmegaConf.update(config, "actor_rollout_ref.rollout.custom.agent_framework", agent_framework_cfg, force_add=True)

    # TransferQueue carries the rollout trajectories (and their rm_scores).
    OmegaConf.update(config, "transfer_queue.enable", True, force_add=True)

    # Data.
    config.data.return_raw_chat = True
    config.data.max_prompt_length = args.prompt_length
    config.data.max_response_length = args.response_length

    return config


def _build_prompts(samples: list, uids: list):
    """Assemble the TensorDict batch the framework's ``generate_sequences`` expects."""
    return tu.get_tensordict(
        tensor_dict={
            "raw_prompt": [sample.get("prompt") for sample in samples],
            "uid": list(uids),
            "tools_kwargs": [sample["extra_info"]["tools_kwargs"] for sample in samples],
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
        description=(
            "Parallel agent inference over a verl-launched engine with the KV-cache-aware router (framework + TQ path)."
        )
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

    # Sampling.
    parser.add_argument("--prompt-length", type=int, default=4096, help="Maximum prompt length (tokens).")
    parser.add_argument("--response-length", type=int, default=8192, help="Maximum response length (tokens).")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before slicing (--limit / --n). Aligns with fully_async data.shuffle.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --shuffle.")

    # Engine / hardware.
    parser.add_argument(
        "--engine",
        default="vllm",
        choices=["vllm", "sglang"],
        help="Inference engine backend.",
    )
    parser.add_argument("--num-workers", type=int, default=1, help="Number of agent rollout workers.")
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes to run the engine on.")
    parser.add_argument("--n-gpus-per-node", type=int, default=8, help="Number of GPUs per node.")
    parser.add_argument(
        "--tensor-parallel-size", "--tp", dest="tensor_parallel_size", type=int, default=4, help="Tensor parallel size."
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="Engine GPU memory fraction.")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.getenv("MAX_MODEL_LEN", "0")) or None,
        help=(
            "Maximum model context length (tokens) passed to the engine. Must be >= "
            "the task config's max_total_tokens; otherwise multi-step episodes crash "
            "with HTTP 400 once the transcript grows past the engine limit."
        ),
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

    # ---- KV-cache-aware router ----
    parser.add_argument(
        "--router-config-path",
        type=str,
        default=DEFAULT_ROUTER_CONFIG_PATH,
        help="Router YAML (repo-root relative or absolute) whose strategy[0] is overridden "
        "and passed to verl via rollout.router_config_path (no rewrite; runtime "
        "knob overrides ride on rollout.custom.agent_framework.router).",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of concurrent sequences per engine.",
    )
    parser.add_argument(
        "--kv-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable vLLM kv-events zmq publisher for retained-cache occupancy collection. "
        "Required for KVCAware router load signal and standalone collector metrics. "
        "Use --no-kv-events to disable.",
    )
    parser.add_argument(
        "--simulated-runner-fqn",
        type=str,
        default=None,
        help="Swap the sandbox-backed task runner for a test double (canned tool "
        "observations, no container); the e2e tests drive the loop through this.",
    )
    parser.add_argument(
        "--enable-mooncake",
        action="store_true",
        help="Attach MooncakeStoreConnector for cross-replica KV sharing.",
    )
    parser.add_argument(
        "--mooncake-config-path",
        type=str,
        default="mooncake_config.json",
        help="Path to the mooncake config JSON (used with --enable-mooncake).",
    )
    parser.add_argument(
        "--device",
        choices=["gpu", "ascend"],
        default="gpu",
        help="Device backend (selects the mooncake connector class).",
    )

    # KVCAware router knobs — carried on rollout.custom.agent_framework.router and
    # applied by the balancer at construction (override the packaged router YAML).
    parser.add_argument(
        "--load-threshold",
        type=float,
        default=0.9,
        help="KVCAware load_threshold (overload when load > threshold, (0,1)); "
        "read by the balancer from rollout.custom.agent_framework.router at construction.",
    )

    args = parser.parse_args()

    if args.enable_mooncake and args.mooncake_config_path:
        os.environ["MOONCAKE_CONFIG_PATH"] = os.path.expanduser(args.mooncake_config_path)

    if not ray.is_initialized():
        if os.environ.get("RAY_ADDRESS"):
            ray.init(address="auto")
        else:
            # Disable Ray's idle-worker reaper so agent workers survive dispatch gaps
            # (default ~10 s threshold would kill them prematurely).
            ray.init(_system_config={"idle_worker_killing_time_threshold_ms": _RAY_IDLE_WORKER_TIMEOUT_MS})

    resolver = TaskConfigResolver.from_file(args.task_config)
    served_model_name = os.path.basename(os.path.expanduser(args.model_path).rstrip("/"))

    dataset = load_dataset("parquet", data_files=args.data_path, split="train")
    if args.shuffle:
        logger.info("Shuffling dataset (seed=%d) before sampling", args.seed)
        dataset = dataset.shuffle(seed=args.seed)
    samples = dataset.to_list()
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        logger.warning("no samples selected; exiting")
        return
    n = max(1, args.n)

    task_configs = list(resolver.defaults_by_name.values())

    logger.info(f"loaded {len(samples)} prompts (x n={n} sessions each) from {args.data_path}")

    # 1. TransferQueue + verl inference engine with the KV-cache-aware router.
    logger.info("initializing configuration, TransferQueue, and LLMServerManager (kvcaware router)...")
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
    prompts = _build_prompts(samples, uids)
    logger.info("starting inference...")
    begin_time = time.time()
    adapter.generate_sequences_and_wait(prompts)
    wall = time.time() - begin_time

    # 4. Read rm_scores back from TQ and report.
    read = _read_rm_scores(uids, partition_id=PARTITION_ID)
    _report(read, wall=wall, num_prompts=len(samples), n=n, args=args, served_model_name=served_model_name)


if __name__ == "__main__":
    main()
