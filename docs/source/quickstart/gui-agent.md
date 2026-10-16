# Run a GUI Agent

A GUI Agent sees the screen as images and acts with clicks, typing, key presses, and scrolling. Uni-Agent supports this with the white-box ReAct Agent, the `gui_*` [Tools](../concepts/tool-and-toolbox.md#gui-tools), and the `gui` [Task](../concepts/task-and-reward.md#gui-task). The model needs vision support behind its OpenAI-compatible endpoint.

## Build the Desktop Image

[`examples/gui`](https://github.com/verl-project/uni-agent/tree/main/examples/gui) holds a minimal desktop: Xvfb (1024x768), `xdotool`, ImageMagick, and a Tk app whose button writes a marker file.

```bash
docker build -t uni-agent-gui examples/gui
```

The image entrypoint starts Xvfb on `$DISPLAY` and then runs the command you give it, so a Sandbox can keep it alive with `sleep infinity`.

## Run the Task

```python
import asyncio
from uni_agent.tasks import get_task

task = get_task({
    "name": "gui",
    "sandbox": {"provider": "docker", "image": "uni-agent-gui"},
    "agent": {
        "name": "react",
        "model": {"base_url": "http://127.0.0.1:8000/v1", "model_name": "policy"},
        "tools": [{"name": n, "coordinate_space": "relative_1000"}
                  for n in ("gui_screenshot", "gui_click", "gui_type", "gui_key", "gui_scroll")]
                 + [{"name": "finish"}],
        "max_steps": 8,
    },
    "setup_command": "nohup python3 /opt/gui/button_app.py >/tmp/app.log 2>&1 </dev/null & sleep 3",
    "verify_command": "test -f /tmp/gui_button_clicked",
    "prompt": [
        {"role": "system", "content": "You control a Linux desktop with tools. Your first action must be screenshot. "
                                      "Check each new screenshot before you call finish."},
        {"role": "user", "content": "Click the SUBMIT button on screen."},
    ],
})
print(asyncio.run(task.run()))
```

`reward` is `1.0` when `verify_command` exits `0`. The Task display and the tools' `display` must agree; the Task copies its own display into tools that omit one.

## Tips

- **Coordinates.** Qwen-VL models answer on a 0-1000 grid, so use `coordinate_space: relative_1000`. Other models that answer in screen pixels use the default `pixel`.
- **Prompt.** Small models may click without looking. Tell the model to take a screenshot first and to check the result before finishing.
- **Parallel episodes.** Each episode needs its own X server. Use a per-session Sandbox such as Docker; with the `local` provider every episode shares one display.
- **Training.** Screenshots that first appear mid-episode are handled by the [Gateway](../concepts/gateway-and-trajectories.md) when the model has a multimodal processor, so the same Task can run under RL rollouts with a vision-language policy.
