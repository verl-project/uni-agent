# Aider SWE Task Recipe

Run the [Aider](https://aider.chat) CLI as a black-box agent on SWE-bench-style
tasks. The adapter launches one headless Aider run inside the task sandbox
against an OpenAI-compatible endpoint (a session-scoped Gateway or an external
provider), and the Task scores the uncommitted working-tree diff.

## Build the sidecar image

The tool image carries a self-contained Python venv with `aider-chat==0.86.2`
at `/opt/aider`; the task image only needs `git` and a shell.

```bash
# Native build for the current host architecture:
bash examples/blackbox_recipes/aider_swe_task/build_tool.sh

# Native ARM64 build on an ARM host (no emulation):
bash examples/blackbox_recipes/aider_swe_task/build_tool.sh --platform linux/arm64

# x86 image on an ARM host (requires amd64 emulation):
bash examples/blackbox_recipes/aider_swe_task/build_tool.sh --platform linux/amd64

# Optional: pip mirror and registry publish.
bash examples/blackbox_recipes/aider_swe_task/build_tool.sh \
  --pip-index https://pypi.tuna.tsinghua.edu.cn/simple/ \
  --registry registry.example.com/team
```

Check the resulting platform with
`docker image inspect aider-tool:0.86.2 --format '{{.Os}}/{{.Architecture}}'`.

### Docker version note

`image_mounts` requires **Docker >= 26** (image-type mounts) with the
containerd image store enabled. On older Docker versions (e.g. Docker 25
shipped by openEuler), build a combined task image instead:

    FROM python:3.12
    COPY --from=aider-tool:0.86.2 / /opt/aider

and reference it as the sandbox `image` directly, keeping the same
`executable_paths`. The end-to-end validation for this recipe was run on
Docker 25.0.3 using this combined-image fallback.

## Run

Bind the image in the task config and select the `aider` agent:

```yaml
sandbox:
  image_mounts:
    - image: aider-tool:0.86.2
      mount_path: /opt/aider
  executable_paths:
    aider: /opt/aider/bin/aider
agent:
  name: aider
  run_timeout: 4800
```

The runner injects `model.base_url`, `model.api_key`, and `model.model_name`.
The adapter launches:

```
aider --model openai/<model_name> --openai-api-base <base_url> \
  --message-file <prompt> --exit --yes-always --no-auto-commits \
  --no-analytics --no-check-update --no-restore-chat-history \
  --no-browser --no-suggest-shell-commands --no-show-model-warnings
```

The API key travels through the private exec environment (`OPENAI_API_KEY`),
never through argv. `--no-auto-commits` leaves the diff uncommitted for Task
verification. `edit_format` and `extra_args` are forwarded when configured.

## Smoke-verified

- Headless invocation against the DeepSeek OpenAI-compatible endpoint on
  aarch64 (openEuler): Aider edited a git-tracked file and exited normally.
- Native `linux/arm64` sidecar build (python-build-standalone aarch64).

## Non-goals

Multi-agent orchestration, architect/watch-files modes, and GPU RL training
recipes are out of scope for this recipe.
