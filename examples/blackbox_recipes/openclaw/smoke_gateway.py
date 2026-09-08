"""Real OpenClaw through Uni-Agent Gateway: verify token-level single trajectory."""
import argparse
import asyncio
import dataclasses
import json
from pathlib import Path
import uuid

from transformers import AutoTokenizer
from examples.gateway.debug_launcher import OpenAICompletionsBackend, TemplateResultTokenIdsWrapper
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.gateway import _GatewayActor
from uni_agent.framework.task_runner import run_task
from dataset import task_payload


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = TemplateResultTokenIdsWrapper(AutoTokenizer.from_pretrained(args.model_path))
    backend = OpenAICompletionsBackend(backend_base_url=args.backend, backend_model="Qwen3.5-9B", timeout=240)
    actor = _GatewayActor(GatewayActorConfig(tokenizer=tokenizer, tool_parser_name="qwen3_coder",
        rollout_backend="vllm", apply_chat_template_kwargs={"enable_thinking": False},
        prompt_length=8192, response_length=24576, enable_last_assistant_rollback=False), backend)
    await actor.start()
    session_id = uuid.uuid4().hex
    try:
        session = await actor.create_session(session_id, sampling_params={"logprobs": True, "temperature": 0.0})
        sample = task_payload()
        sample["agent"]["artifact_dir"] = str(args.output / "openclaw-trajectories")
        sample["agent"]["cli_timeout_seconds"] = 450
        sample["agent"]["run_timeout"] = 480
        sample["metadata"]["agent_timeout"] = 550
        result = await run_task(session=session, raw_prompt=sample["prompt"], sample_index=0,
                                tools_kwargs={"task": sample}, model_name="Qwen3.5-9B")
        trajectories = await actor.finalize_session(session_id)
        records = [dataclasses.asdict(t) for t in trajectories]
        (args.output / "gateway-trajectories.json").write_text(json.dumps(records))
        report = dataclasses.asdict(result)
        report["gateway_trajectory_count"] = len(records)
        aligned = len(records) == 1 and bool(records[0]["response_logprobs"]) and (
            len(records[0]["response_ids"]) == len(records[0]["response_mask"]) == len(records[0]["response_logprobs"]))
        report["logprobs_aligned"] = aligned
        report["task_completed"] = result.reward == 1 and result.finished is True and len(records) == 1
        report["token_evidence_complete"] = report["task_completed"] and aligned
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"task_completed": report["task_completed"], "reward": result.reward,
                          "finished": result.finished, "gateway_trajectory_count": len(records)}))
        return report["token_evidence_complete"]
    finally:
        await actor.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", default="http://127.0.0.1:18090/v1")
    parser.add_argument("--model-path", default="/home/zxh/models/Qwen3.5-9B")
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(args)) else 1)
