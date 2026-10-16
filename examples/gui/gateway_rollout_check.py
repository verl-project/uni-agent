"""Live gateway rollout check: `gui` task -> ReAct agent -> GatewayActor -> in-process vLLM engine.

Runs one episode through the real Gateway with a vision-language model, then audits the captured
trajectory: aligned `response_ids` / `response_mask` / `response_logprobs`, one 768-token block per
screenshot, image pad tokens masked out of the loss, and `multi_modal_data` listing every screenshot.

Needs a GPU, `vllm==0.23.0`, `ray`, the verl submodule on the path, Xvfb + xdotool + ImageMagick, and the Tk
app from this directory at /app/button_app.py. On a machine without `nvcc`, set VLLM_USE_FLASHINFER_SAMPLER=0.
The engine rejects pre-expanded image tokens on vllm 0.11, so use the version the repo pins.
Exit code 0 means every assertion held.
"""

import asyncio
import logging
import subprocess
import time

from transformers import AutoProcessor, AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.gateway import _GatewayActor
from uni_agent.tasks import get_task
from verl.workers.rollout.replica import TokenOutput

logging.basicConfig(level=logging.INFO, format="%(message)s")
for noisy in ("aiohttp", "asyncio", "httpx", "vllm", "uvicorn", "uvicorn.access"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

NAME = "Qwen/Qwen3-VL-4B-Instruct"
TOOLS = ["gui_screenshot", "gui_click", "gui_type", "gui_key", "gui_scroll"]
SYSTEM = (
    "You control a Linux desktop with tools. You cannot see the screen until you call the screenshot tool, so your "
    "FIRST action must always be screenshot. Find the target in the image, then click its center using coordinates "
    "read from that image. After each action look at the new screenshot to check that it worked, and only call "
    "finish once the screen shows the task is done (for example the label changes to Done!)."
)


class VllmBackend:
    """Gateway backend over an in-process vLLM engine: token ids + processed images in, tokens + logprobs out."""

    def __init__(self, engine):
        self.engine = engine
        self.calls = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None, **kwargs):
        sp = SamplingParams(
            temperature=sampling_params.get("temperature", 1.0),
            top_p=sampling_params.get("top_p", 1.0),
            top_k=sampling_params.get("top_k", -1),
            max_tokens=sampling_params.get("max_tokens", 512),
            logprobs=0,
        )
        prompt = {"prompt_token_ids": list(prompt_ids)}
        if image_data:
            prompt["multi_modal_data"] = {"image": list(image_data)}
        self.calls.append({"n_prompt": len(prompt_ids), "n_images": len(image_data or [])})
        final = None
        async for out in self.engine.generate(prompt, sp, request_id):
            final = out
        comp = final.outputs[0]
        token_ids = list(comp.token_ids)
        log_probs = [lp[t].logprob for t, lp in zip(token_ids, comp.logprobs, strict=True)]
        return TokenOutput(token_ids=token_ids, log_probs=log_probs, stop_reason="completed")


async def main():
    xvfb = subprocess.Popen(
        ["Xvfb", ":0", "-screen", "0", "1024x768x24"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    for _ in range(50):
        if subprocess.run(["xdpyinfo", "-display", ":0"], capture_output=True).returncode == 0:
            break
        time.sleep(0.2)

    tok = AutoTokenizer.from_pretrained(NAME)
    proc = AutoProcessor.from_pretrained(NAME)
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(model=NAME, max_model_len=16384, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 12})
    )
    backend = VllmBackend(engine)
    actor = _GatewayActor(
        GatewayActorConfig(
            tokenizer=tok,
            processor=proc,
            tool_parser_name="hermes",
            rollout_backend="vllm",
            hf_model_type="qwen3_vl",
            allowed_request_sampling_param_keys={"temperature", "top_p", "top_k"},
        ),
        backend,
    )
    await actor.start()
    handle = await actor.create_session("gui-rollout-0", sampling_params={"logprobs": True})

    task = get_task(
        {
            "name": "gui",
            "sandbox": {"provider": "local"},
            "agent": {
                "name": "react",
                "model": {"base_url": handle.base_url, "model_name": "policy"},
                "tools": [{"name": n, "display": ":0", "coordinate_space": "relative_1000"} for n in TOOLS]
                + [{"name": "finish"}],
                "max_steps": 8,
                "sampling_params_override": {"temperature": 0.0, "max_tokens_per_turn": 512},
            },
            "setup_command": (
                "rm -f /tmp/gui_button_clicked; "
                "nohup /usr/bin/python3 /app/button_app.py >/tmp/app.log 2>&1 </dev/null & sleep 3"
            ),
            "verify_command": "test -f /tmp/gui_button_clicked",
            "prompt": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Click the SUBMIT button on screen."},
            ],
        }
    )
    result = await task.run()
    trajectories = await actor.finalize_session("gui-rollout-0")
    await actor.shutdown()
    xvfb.terminate()

    print("RESULT", result)
    print("BACKEND CALLS", backend.calls)
    assert len(trajectories) == 1, f"expected one trajectory, got {len(trajectories)}"
    t = trajectories[0]
    pad = tok.convert_tokens_to_ids("<|image_pad|>")
    ids = list(t.prompt_ids) + list(t.response_ids)
    n_images = len((t.multi_modal_data or {}).get("images", []))
    pads = ids.count(pad)
    mask_on_pad = sum(m for i, m in zip(t.response_ids, t.response_mask, strict=True) if i == pad)
    print(
        f"prompt={len(t.prompt_ids)} response={len(t.response_ids)} mask_sum={sum(t.response_mask)} "
        f"logprobs={len(t.response_logprobs or [])} images={n_images} image_pad_tokens={pads} num_turns={t.num_turns}"
    )
    assert result.reward == 1.0, result
    assert len(t.response_ids) == len(t.response_mask) == len(t.response_logprobs), "response arrays misaligned"
    assert n_images >= 2, "expected the first screenshot and the post-click screenshot"
    assert pads == 768 * n_images, f"image pad tokens {pads} != 768 * {n_images}"
    assert mask_on_pad == 0, "image pad tokens must be masked out of the loss"
    assert 0 < sum(t.response_mask) < len(t.response_mask), "mask should mix model (1) and context (0) tokens"
    segs, cur = [], []
    for tid, m in zip(t.response_ids, t.response_mask, strict=True):
        if m:
            cur.append(tid)
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    for i, seg in enumerate(segs):
        print(f"MODEL TURN {i}: {tok.decode(seg)[:230]!r}")
    print("GATEWAY ROLLOUT PASS")
    engine.shutdown()


asyncio.run(main())
