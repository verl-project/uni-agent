# GUI agent example

A minimal desktop for the `gui_*` tools (`screenshot`, `click`, `type_text`, `press_key`,
`scroll`) and the `gui` task. The image runs Xvfb with `xdotool` and ImageMagick; the
test app is a Tk window whose button writes a marker file.

```bash
docker build -t uni-agent-gui examples/gui
```

Task sample (`gui` task, ReAct agent, docker sandbox with the image above):

```python
{
    "name": "gui",
    "sandbox": {"provider": "docker", "image": "uni-agent-gui"},
    "agent": {"name": "react", "tools": [{"name": n, "coordinate_space": "relative_1000"} for n in
              ("gui_screenshot", "gui_click", "gui_type", "gui_key", "gui_scroll")] + [{"name": "finish"}]},
    "setup_command": "nohup python3 /opt/gui/button_app.py >/tmp/app.log 2>&1 </dev/null & sleep 3",
    "verify_command": "test -f /tmp/gui_button_clicked",
    "metadata": {"instruction": "Click the SUBMIT button."},
}
```

The agent needs a vision-language model behind its endpoint. Screenshots reach it as a
user message with `image_url` parts after each turn's tool results.

See the [GUI agent quickstart](../../docs/source/quickstart/gui-agent.md) for the full walkthrough.

## Run it against a model

`run_gui_task.py` runs the sample task above against any OpenAI-compatible vision-language endpoint
and exits `0` when the button was clicked. The setup validated for this change was one A10G serving
`Qwen/Qwen3-VL-4B-Instruct` with vLLM (command in the script's docstring), the example image in a
sandbox, and `coordinate_space: relative_1000`.

```bash
python examples/gui/run_gui_task.py --provider docker --image uni-agent-gui
```

## Check the gateway path

`gateway_rollout_check.py` runs one episode through the real Gateway with an in-process vLLM engine
(needs a GPU and `vllm==0.23.0`; the docstring lists the rest) and audits the captured trajectory:
aligned response arrays, one 768-token block per screenshot, image pad tokens masked out, and every
screenshot listed in `multi_modal_data`.

