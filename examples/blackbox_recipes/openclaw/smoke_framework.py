"""Real OpenClaw through Uni-Agent Gateway: verify token-level single trajectory."""
import argparse
import asyncio
import dataclasses
import json
from pathlib import Path
import uuid
import os
import ray

from transformers import AutoTokenizer
from examples.gateway.debug_launcher import OpenAICompletionsBackend, TemplateResultTokenIdsWrapper
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.gateway import _GatewayActor
from uni_agent.framework.framework import GatewayAgentFramework, _RunnerConfig
from dataset import task_payload


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = TemplateResultTokenIdsWrapper(AutoTokenizer.from_pretrained(args.model_path))
    backend = OpenAICompletionsBackend(backend_base_url=args.backend, backend_model="Qwen3.5-9B", timeout=240)
    actor = _GatewayActor(GatewayActorConfig(tokenizer=tokenizer, tool_parser_name="qwen3_coder",
        rollout_backend="vllm", apply_chat_template_kwargs={"enable_thinking": False},
        prompt_length=8192, response_length=24576, enable_last_assistant_rollback=False), backend)
    if args.dispatch == "ray_task":
        ray.init(address="local", num_cpus=2, num_gpus=0, include_dashboard=False,
                 runtime_env={"env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}})
    await actor.start()
    session_id = uuid.uuid4().hex
    try:

        sample = task_payload(case=args.case)
        sample["agent"]["artifact_dir"] = str(args.output / "openclaw-trajectories")
        sample["agent"]["cli_timeout_seconds"] = 450
        sample["agent"]["run_timeout"] = 480
        sample["metadata"]["agent_timeout"] = 550
        runner = _RunnerConfig(runner_fqn="uni_agent.framework.task_runner.run_task",
            runner_kwargs={"model_name": "Qwen3.5-9B"}, dispatch_mode=args.dispatch,
            max_concurrent_sessions=1, trajectory_selection="all")
        framework = GatewayAgentFramework(actor, runner_registry={"task": runner},
                                          log_dir=str(args.output / "framework"))
        trajectories, _ = await framework._run_agent_episode(
            sample_fields={"uid": session_id, "raw_prompt": sample["prompt"], "tools_kwargs": {"task": sample}},
            sample_index=0, session_index=0, global_steps=None, runner_name="task", runner_config=runner,
            sampling_params={"logprobs": True, "temperature": 0.0})
        records = [dataclasses.asdict(t) for t in trajectories]
        (args.output / "gateway-trajectories.json").write_text(json.dumps(records))
        aligned = len(records) == 1 and bool(records[0]["response_logprobs"]) and (
            len(records[0]["response_ids"]) == len(records[0]["response_mask"]) == len(records[0]["response_logprobs"]))
        passed = aligned and records[0]["reward_score"] == 1.0 and records[0]["finished"] is True
        report = {"passed": passed, "trajectory_count": len(records), "logprobs_aligned": aligned,
                  "finished": records[0]["finished"] if records else None,
                  "reward_score": records[0]["reward_score"] if records else None,
                  "scope": f"real framework episode {args.dispatch}, no optimizer or TransferQueue"}
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
        return passed
    finally:
        await actor.shutdown()
        if args.dispatch == "ray_task":
            ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", default="http://127.0.0.1:18090/v1")
    parser.add_argument("--model-path", default="/home/zxh/models/Qwen3.5-9B")
    parser.add_argument("--case", default="merge_intervals", choices=["merge_intervals", "unicode_counts", "topological_sort"])
    parser.add_argument("--dispatch", choices=["inline_async", "ray_task"], default="inline_async")
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(args)) else 1)
