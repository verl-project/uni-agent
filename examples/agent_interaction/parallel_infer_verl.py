# ruff: noqa: E501
"""Parallel agent inference over a verl-launched engine, through the training path.

Same job as ``parallel_infer_api.py`` (run each dataset row's task and report a
score) but instead of talking to an OpenAI endpoint you started yourself, this
brings the engine up with verl and drives rollouts through the *exact* training
rollout stack -- the agent **framework** adapter + TransferQueue (TQ):

    verl LLMServerManager (vLLM / SGLang)
    ->  AgentFrameworkRolloutAdapter.generate_sequences   (fire-and-forget -> TQ)
          ->  Gateway sessions (per-session OpenAI-compatible endpoints)
          ->  uni_agent.framework.task_runner.run_task  ->  uni_agent task
    ->  per-trajectory records written to TransferQueue

The per-sample score is the trainer's own signal: read each session's final
trajectory back from TQ and take ``rm_scores.sum(dim=-1)`` (mirrors
``main_ppo_sync``'s validation read). ``rm_scores`` is only populated when a
reward worker scored the trajectory, so we attach a tiny worker
(:class:`_TaskRewardWorker`) that surfaces the task's own reward -- which
``run_task`` posts to the session reward-info endpoint (``report_reward=True``)
-- as ``reward_score``. No external reward model is needed.

Because rollouts flow through the framework, fan-out is ``rollout.n`` (``--n``
sessions per prompt), not a driver loop, and there is no resolved / wrong-answer
/ timeout bucketing -- just the mean ``rm_scores``.

Example (single node, 4-way tensor parallel)::

    python examples/agent_interaction/parallel_infer_verl.py \
        --data-path ~/data/swe_agent/swe_bench_verified.parquet \
        --model-path ~/models/Qwen3-Coder-30B-A3B-Instruct \
        --tool-parser qwen3_coder --tensor-parallel-size 4 \
        --task-config examples/agent_interaction/task_config.yaml --limit 8

As with ``parallel_infer_api.py``, the agent config is built from the per-flag
knobs unless ``--task-config`` is given (a YAML task config deep-merged onto each
sample's task dict). Either way the policy endpoint is the gateway session, bound
by the runner -- not a flag.
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
import yaml
from datasets import load_dataset
from omegaconf import OmegaConf

import verl

try:
    import transfer_queue as tq
except ImportError:  # fall back to verl's shim (mock raises a clear error if TQ is missing)
    from verl.utils.transferqueue_utils import tq

from uni_agent.agents import get_agent_cls
from uni_agent.framework.entry import AgentFrameworkRolloutAdapter
from verl.utils import tensordict_utils as tu
from verl.workers.rollout.llm_server import LLMServerManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


GLOBAL_CONCURRENCY = int(os.getenv("GLOBAL_CONCURRENCY", 128))
PARTITION_ID = "train"  # framework routes non-"validate" batches to the "train" TQ partition


def _rule(text: str = "", width: int = 50, ch: str = "-") -> str:
    """A centered-title horizontal rule."""
    if not text:
        return ch * width
    pad = max(0, width - len(text) - 2)
    return f"{ch * (pad // 2)} {text} {ch * (pad - pad // 2)}"


def _load_task_yaml(path: str) -> dict:
    """Load a YAML task-config file into a dict (a one-item ``- name: ...`` list is unwrapped)."""
    raw = yaml.safe_load(Path(path).expanduser().read_text())
    if isinstance(raw, list):
        if not raw or not isinstance(raw[0], dict):
            raise ValueError(f"--task-config {path!r}: a YAML list must start with a mapping (the task config).")
        raw = raw[0]
    if not isinstance(raw, dict):
        raise ValueError(f"--task-config {path!r} must be a YAML mapping (the task config), got {type(raw).__name__}")
    return raw


def build_task_overrides(args: argparse.Namespace) -> dict:
    """Build the task-config dict deep-merged onto every sample's task.

    Mirrors ``parallel_infer_api.py`` minus the endpoint: the policy server is the
    per-sample gateway session, so ``agent.model.base_url`` is filled in at run
    time by ``run_task`` rather than here.
    """
    if args.task_config:
        overrides = _load_task_yaml(args.task_config)
    else:
        model_cfg: dict = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        }
        if args.max_tokens is not None:
            model_cfg["max_tokens_per_turn"] = args.max_tokens
        if args.max_total_tokens is not None:
            model_cfg["max_total_tokens"] = args.max_total_tokens
        agent_cfg: dict = {"name": args.agent, "model": model_cfg}
        # max_steps is code_act's turn budget; include it only for agents that declare it.
        if "max_steps" in get_agent_cls(args.agent).config_model.model_fields:
            agent_cfg["max_steps"] = args.max_steps
        overrides = {"agent": agent_cfg}

    agent = overrides.setdefault("agent", {})
    agent.setdefault("name", args.agent)
    agent.setdefault("model", {})
    return overrides


def init_config(args: argparse.Namespace, *, task_overrides: dict, served_model_name: str):
    """Compose verl's ``ppo_trainer`` config and override the engine + framework knobs."""
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(verl.__file__).resolve().parent / "trainer" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")

    rollout = config.actor_rollout_ref.rollout

    # Sampling (per-request params still win; kept here for engine-side defaults).
    rollout.temperature = args.temperature
    rollout.top_p = args.top_p
    rollout.val_kwargs.temperature = args.temperature
    rollout.val_kwargs.top_p = args.top_p

    # Fan-out: the framework runs rollout.n gateway sessions per prompt.
    rollout.n = max(1, args.n)

    # Hardware.
    rollout.nnodes = args.nnodes
    rollout.n_gpus_per_node = args.n_gpus_per_node
    config.trainer.nnodes = args.nnodes
    config.trainer.n_gpus_per_node = args.n_gpus_per_node

    # Model + engine.
    config.actor_rollout_ref.model.path = os.path.expanduser(args.model_path)
    rollout.name = args.engine
    rollout.mode = "async"
    rollout.prompt_length = args.prompt_length
    rollout.response_length = args.response_length
    rollout.tensor_model_parallel_size = args.tensor_parallel_size
    rollout.gpu_memory_utilization = args.gpu_memory_utilization

    # Gateway tool-call parser: the gateway decodes tool calls from raw tokens, so
    # this must match the model's chat template (the analog of vLLM's
    # --tool-call-parser, e.g. qwen3_coder for Qwen3-Coder, hermes for Qwen3).
    OmegaConf.update(config, "actor_rollout_ref.rollout.multi_turn.format", args.tool_parser, force_add=True)

    # Framework wiring: gateway pool size + the single task runner that turns each
    # gateway session into a uni_agent task. runner_kwargs are forwarded to
    # run_task; report_reward makes it post the task reward to the session so the
    # reward worker (and thus rm_scores) can pick it up.
    agent_framework_cfg = {
        "gateway_count": args.gateway_count,
        "agent_runners": {
            "task": {
                "runner_fqn": "uni_agent.framework.task_runner.run_task",
                "dispatch_mode": "inline_async",
                "max_concurrent_sessions": max(0, args.concurrency),
                "runner_kwargs": {
                    "task_overrides": task_overrides,
                    "model_name": served_model_name,
                    "report_reward": True,
                },
            }
        },
    }
    OmegaConf.update(config, "actor_rollout_ref.rollout.custom.agent_framework", agent_framework_cfg, force_add=True)

    # TransferQueue carries the rollout trajectories (and their rm_scores).
    OmegaConf.update(config, "transfer_queue.enable", True, force_add=True)

    # Data.
    config.data.return_raw_chat = True
    config.data.max_prompt_length = args.prompt_length
    config.data.max_response_length = args.response_length

    return config


def _build_prompts(samples: list, uids: list):
    """Assemble the TensorDict batch the framework's ``generate_sequences`` expects.

    Only the fields the framework needs are included: ``raw_prompt`` (required by
    the session lifecycle, unused by run_task), ``uid`` (TQ key namespace), and
    ``tools_kwargs`` (carries the per-sample task). ``global_steps`` is a
    batch-level scalar the framework tags every record with.
    """
    return tu.get_tensordict(
        tensor_dict={
            "raw_prompt": [sample.get("prompt") for sample in samples],
            "uid": list(uids),
            "tools_kwargs": [sample["extra_info"]["tools_kwargs"] for sample in samples],
        },
        non_tensor_dict={"global_steps": 0},
    )


@ray.remote
class _TaskRewardWorker:
    """Reward worker that surfaces each task's own reward as ``reward_score``.

    ``run_task`` (report_reward=True) posts the task reward to the session
    reward-info endpoint; the framework merges that ``reward_info`` into the
    sample's ``extra_info`` before dispatch, so this reads ``reward`` back and
    returns it in the RewardLoopWorker contract shape
    (``{"reward_score", "reward_extra_info"}``) -- no external reward model needed.
    Without it the framework skips scoring and every ``rm_scores`` stays 0.
    """

    def compute_score(self, data) -> dict:
        extra_info = data.non_tensor_batch.get("extra_info")
        info = extra_info[0] if extra_info is not None and len(extra_info) else {}
        reward = float((info or {}).get("reward", 0.0))
        return {"reward_score": reward, "reward_extra_info": {}}


def _read_rm_scores(uids: list, *, partition_id: str = PARTITION_ID) -> dict:
    """Read each session's final trajectory back from TQ and score it.

    Trajectory records are keyed ``{uid}_{session}_{index}``; a session may span
    several records (multi-turn segments), so we keep the highest ``index`` per
    ``(uid, session)`` -- the final segment -- and take ``rm_scores.sum(dim=-1)``
    as that session's score (same reduction as ``main_ppo_sync._validate``).
    """
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
    parser.add_argument("--agent", default="code_act", help="Registered agent name to run.")
    parser.add_argument(
        "--task-config",
        help="Path to a YAML task config (name/sandbox/agent/...) deep-merged onto each sample's task dict. "
        "Its `agent` section supersedes the per-flag knobs; the endpoint is bound to the gateway session.",
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

    # Sampling / rollout knobs (keep temperature/top_p/top_k aligned with training).
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k sampling; -1 disables it.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Per-turn generation cap (max_tokens per model response); omit to fall back to --max-total-tokens.",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=65536,
        help="Whole-episode generation budget (sum of completion tokens across turns).",
    )
    parser.add_argument("--max-steps", type=int, default=100, help="Max tool-calling turns per episode.")
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
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes to run the engine on.")
    parser.add_argument("--n-gpus-per-node", type=int, default=8, help="Number of GPUs per node.")
    parser.add_argument(
        "--tensor-parallel-size", "--tp", dest="tensor_parallel_size", type=int, default=4, help="Tensor parallel size."
    )
    parser.add_argument("--prompt-length", type=int, default=4096, help="Max prompt length (tokens).")
    parser.add_argument("--response-length", type=int, default=65536, help="Max response length (tokens).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="Engine GPU memory fraction.")
    parser.add_argument(
        "--gateway-count",
        "--num-workers",
        dest="gateway_count",
        type=int,
        default=8,
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

    args = parser.parse_args()

    task_overrides = build_task_overrides(args)
    agent_cfg = task_overrides.get("agent", {})
    model_cfg = agent_cfg.get("model", {})
    served_model_name = args.served_model_name or os.path.basename(os.path.expanduser(args.model_path).rstrip("/"))

    dataset = load_dataset("parquet", data_files=args.data_path, split="train")
    samples = dataset.to_list()[30:31]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        logger.warning("no samples selected; exiting")
        return
    n = max(1, args.n)

    logger.info(f"loaded {len(samples)} prompts (x n={n} sessions each) from {args.data_path}")
    logger.info(
        f"task={task_overrides.get('name') or '<dataset>'} agent={agent_cfg.get('name', args.agent)} "
        f"provider={task_overrides.get('sandbox', {}).get('provider')} "
        f"engine={args.engine} model={served_model_name} tool_parser={args.tool_parser}"
    )
    logger.info(
        f"gateways={args.gateway_count} concurrency={args.concurrency} "
        f"nnodes={args.nnodes} gpus/node={args.n_gpus_per_node} tp={args.tensor_parallel_size} "
        f"prompt_len={args.prompt_length} response_len={args.response_length} "
        f"sampling=temp{model_cfg.get('temperature')}/top_p{model_cfg.get('top_p')}/top_k{model_cfg.get('top_k')} "
        f"config={('yaml:' + args.task_config) if args.task_config else 'flags'}"
    )

    # 1. Ray + TransferQueue + verl inference engine.
    # ray.init()
    logger.info("initializing configuration, TransferQueue, and LLMServerManager...")
    config = init_config(args, task_overrides=task_overrides, served_model_name=served_model_name)
    tq.init(config.transfer_queue)
    llm_server_manager = LLMServerManager.create(config=config)

    # 2. Framework rollout adapter over the engine, with a reward worker that
    #    surfaces each task's own reward (posted via report_reward) as rm_scores.
    reward_worker = _TaskRewardWorker.remote()
    adapter = AgentFrameworkRolloutAdapter.create(
        config=config,
        llm_client=llm_server_manager.get_client(),
        reward_loop_worker_handles=[reward_worker],
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
