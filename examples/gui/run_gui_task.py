"""Run the example `gui` task against any OpenAI-compatible vision-language endpoint.

Serve a model first, for example (the setup the PR validated on one A10G):

    vllm serve Qwen/Qwen3-VL-4B-Instruct --served-model-name policy --port 8000 \
        --max-model-len 16384 --enable-auto-tool-choice --tool-call-parser hermes \
        --limit-mm-per-prompt '{"image": 12}'

Then, with the image from this directory built as `uni-agent-gui`:

    python examples/gui/run_gui_task.py --provider docker --image uni-agent-gui

`--provider local` runs against the host desktop instead: start Xvfb on :0 first and make sure
`xdotool` and ImageMagick are installed; use it only for a throwaway machine.
Exit code is 0 when the task is solved (reward 1.0), 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from uni_agent.tasks import get_task

SYSTEM_PROMPT = (
    "You control a Linux desktop with tools. You cannot see the screen until you call the screenshot tool, so your "
    "first action must always be screenshot. Find the target in the image, then click its center using coordinates "
    "read from that image. After each action look at the new screenshot to check that it worked, and only call "
    "finish once the screen shows the task is done (for example the label changes to Done!)."
)
GUI_TOOLS = ("gui_screenshot", "gui_click", "gui_type", "gui_key", "gui_scroll")


def build_task_config(args: argparse.Namespace) -> dict:
    sandbox: dict = {"provider": args.provider}
    if args.provider != "local":
        sandbox["image"] = args.image
    if args.provider == "modal":
        sandbox["sandbox_kwargs"] = {"startup_command": ["/opt/gui/start_desktop.sh", "sleep", "infinity"]}
    button_app = "/usr/bin/python3 /app/button_app.py" if args.provider == "local" else "python3 /opt/gui/button_app.py"
    return {
        "name": "gui",
        "sandbox": sandbox,
        "agent": {
            "name": "react",
            "model": {"base_url": args.base_url, "model_name": args.model},
            "tools": [
                {"name": n, "display": args.display, "coordinate_space": args.coordinate_space} for n in GUI_TOOLS
            ]
            + [{"name": "finish"}],
            "max_steps": args.max_steps,
            "sampling_params_override": {"temperature": 0.0, "max_tokens_per_turn": 512},
        },
        "display": args.display,
        "setup_command": f"rm -f /tmp/gui_button_clicked; nohup {button_app} >/tmp/app.log 2>&1 </dev/null & sleep 3",
        "verify_command": "test -f /tmp/gui_button_clicked",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Click the SUBMIT button on screen."},
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="policy")
    parser.add_argument("--provider", default="docker", help="Sandbox provider: docker, modal, or local.")
    parser.add_argument("--image", default="uni-agent-gui")
    parser.add_argument("--display", default=":0")
    parser.add_argument("--coordinate-space", default="relative_1000", choices=["pixel", "relative_1000"])
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    result = asyncio.run(get_task(build_task_config(args)).run())
    print(result)
    return 0 if result.reward == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
