# Build a Sidecar Image

A sidecar packages a tool and its runtime as files mounted into a task sandbox. Use Docker and Bash on Linux x86_64.

Create `examples/blackbox_recipes/my_tool_swe_task/` with two build files:

- `Dockerfile.my-tool`: installs the tool and packages its runtime.
- `build_tool.sh`: builds the image and optionally pushes it to a registry.

Start from Mini-SWE-Agent's [Dockerfile](https://github.com/verl-project/uni-agent/blob/main/examples/mini_swe_agent/Dockerfile.mini-swe-agent-tool) and [build script](https://github.com/verl-project/uni-agent/blob/main/examples/mini_swe_agent/build_tool.sh) for Python, or OpenClaw's [Dockerfile](https://github.com/verl-project/uni-agent/blob/main/examples/blackbox_recipes/openclaw_swe_task/Dockerfile.openclaw-tool) and [build script](https://github.com/verl-project/uni-agent/blob/main/examples/blackbox_recipes/openclaw_swe_task/build_tool.sh) for Node.js.

## Adapt the Dockerfile

Replace the tool name, installation commands, and versions. Bundle the runtime and required dependencies under `/opt/my-tool`, using binaries compatible with the target sandbox's architecture and libc.

Copy a launcher such as `run_agent.sh` into that directory **only if your tool needs one**; otherwise remove the copied recipe's launcher `COPY` instruction. See the [Jiuwenswarm example](https://github.com/Erpim/uni-agent/blob/jiuwen_1020/examples/jiuwenswarm_swe_task/Dockerfile.jiuwenswarm-tool#L43-L50).

The usual final stage exports just the tool directory:

```dockerfile
FROM scratch
COPY --from=builder /opt/my-tool/ /
```

`FROM scratch` is optional: it keeps only the copied files. With another final base, still put the tool directory's contents at the image root, so mounting it at `/opt/my-tool` produces `/opt/my-tool/bin/...`. Match `builder` to your installation stage's name.

## Adapt and run build_tool.sh

Update the Dockerfile name, default image name, and tool-specific build arguments. Keep the script's directory as the Docker build context. If copying the OpenClaw script, also match `--target` to your final stage name, or remove it.

Run from the repository root:

```bash
# Build locally.
TOOL_IMAGE=my-tool TOOL_TAG=v1 bash examples/blackbox_recipes/my_tool_swe_task/build_tool.sh

# Optional: build and push to a registry the sandbox service can pull from.
docker login registry.example.com
TOOL_IMAGE=my-tool TOOL_TAG=v1 bash examples/blackbox_recipes/my_tool_swe_task/build_tool.sh \
  --registry registry.example.com/team
```

Both reference scripts build and push when `--registry` is given. The published image is `registry.example.com/team/my-tool:v1`.
