# Build a Sidecar Image

A sidecar is a [runtime image](../concepts/sandbox.md#runtime-images) attached through `sandbox.image_mounts`: its root filesystem appears at `mount_path`, supplying tool files without changing the task image. Docker, Modal, and OpenYuanRong support image mounts; Local, veFaaS, and E2B do not.

Use Docker and Bash, targeting `linux/amd64`. ARM builders need amd64 emulation for build steps that execute x86_64 binaries.

## Recipe layout

Create `examples/my_tool_swe_task/` (or use `examples/blackbox_recipes/my_tool_swe_task/`, matching your reference) with two build files:

- `Dockerfile.<name>-tool` installs the tool and packages its runtime.
- `build_tool.sh` builds and optionally publishes it; its `docker build -f` path must match that Dockerfile name.

| Runtime | Reference | Optional launcher |
| --- | --- | --- |
| Python | [Mini-SWE-Agent Dockerfile](https://github.com/verl-project/uni-agent/blob/main/examples/mini_swe_agent/Dockerfile.mini-swe-agent-tool), [script](https://github.com/verl-project/uni-agent/blob/main/examples/mini_swe_agent/build_tool.sh) | `run_agent.py` |
| Python | [Hermes Dockerfile](https://github.com/verl-project/uni-agent/blob/main/examples/blackbox_recipes/hermes_swe_task/Dockerfile.hermes-tool) | `run_hermes.py` |
| Node.js | [OpenClaw Dockerfile](https://github.com/verl-project/uni-agent/blob/main/examples/blackbox_recipes/openclaw_swe_task/Dockerfile.openclaw-tool), [script](https://github.com/verl-project/uni-agent/blob/main/examples/blackbox_recipes/openclaw_swe_task/build_tool.sh) | None; builds with `--target tool` |
| Native binary | [Codex Dockerfile](https://github.com/verl-project/uni-agent/blob/main/examples/blackbox_recipes/codex_swe_task/Dockerfile.codex-tool) | None; Node.js is used only during the build |
| Node.js | [Claude Code Dockerfile](https://github.com/verl-project/uni-agent/blob/main/examples/claude_code_swe_task/Dockerfile.claude-code-tool) | None |

Copy a launcher into `/opt/my-tool/bin` only if needed; otherwise remove the copied recipe's launcher `COPY` instruction.

## Adapt the Dockerfile

Replace installation commands and pin versions. Package the runtime and dependencies under `/opt/my-tool`; document anything required from the task image. For example, OpenClaw requires task-image Python and glibc 2.35 or newer. The usual final stage exports just the tool directory:

```dockerfile
FROM scratch
COPY --from=builder /opt/my-tool/ /
```

`FROM scratch` is optional; another final base must preserve the same layout. Mounting the image should produce `/opt/my-tool/bin/...`, not `/opt/my-tool/opt/my-tool/bin/...`. Match `builder` to your installation stage. Mounts supply files, not the image's `ENV` or `ENTRYPOINT`.

## Adapt and run build_tool.sh

Update image defaults and build arguments, keep the script directory as build context, and add `--platform linux/amd64` to `docker build`. Match OpenClaw's `--target` to your final stage, or remove it. Both reference scripts support `TOOL_IMAGE`, `TOOL_TAG`, and `--registry`; Mini-SWE-Agent also has `--pip-index`, while OpenClaw has `--tag`, `--version`, and `--network`. Run from the repository root:

```bash
# Build locally.
TOOL_IMAGE=my-tool TOOL_TAG=v1 bash examples/my_tool_swe_task/build_tool.sh
# Optional: rebuild and push.
docker login registry.example.com
TOOL_IMAGE=my-tool TOOL_TAG=v1 bash examples/my_tool_swe_task/build_tool.sh \
  --registry registry.example.com/team
```

Both reference scripts build and push when `--registry` is given. The published image is `registry.example.com/team/my-tool:v1`.

## Verify the image

Inspect the filesystem without starting the container. The sample final stage has no shell or default command, so `/unused` is only a placeholder and is never executed.

```bash
check=$(docker create --platform linux/amd64 my-tool:v1 /unused)
docker export "$check" | tar -tf - | sed -n '1,20p'
docker rm "$check"
docker image inspect my-tool:v1 --format '{{.Os}}/{{.Architecture}}'
```

Check for root-level tool files (such as `bin/...` and any launcher), rather than a nested `opt/my-tool/`, and confirm `linux/amd64`. This checks layout and platform; use the selected recipe for a task run.

## Mount the image

Add to the existing task's `sandbox` block, retaining its task image. Use a reference the sandbox service can pull; `image_map` rewrites only the task image. `executable_paths` exposes selected executables by command name (also map interpreters such as `node` when needed):

```yaml
sandbox:
  provider: openyuanrong
  image_mounts:
    - image: registry.example.com/team/my-tool:v1
      mount_path: /opt/my-tool
  executable_paths:
    my-tool: /opt/my-tool/bin/my-tool
```

Next: [launch the sandbox](launch-sandbox.md), or follow the [Codex recipe](https://github.com/verl-project/uni-agent/blob/main/examples/blackbox_recipes/codex_swe_task/README.md#build-the-sidecar) for an end-to-end example. See [Runtime images](../concepts/sandbox.md#runtime-images) for mount configuration.

## Let an agent scaffold the recipe

Fill in the inputs and paste this prompt into a coding agent:

```text
Add a sidecar recipe to this uni-agent checkout.
Inputs: name=<NAME>, version=<VERSION>, runtime=<Python/Node.js/native>,
source=<package or URL and archive checksum>, launcher=<relative path or none>,
task image=<distro, libc version, available interpreters>, tag=<TAG>.

Read these repository examples and follow the matching runtime's structure:
- examples/mini_swe_agent/Dockerfile.mini-swe-agent-tool and build_tool.sh
- examples/blackbox_recipes/openclaw_swe_task/Dockerfile.openclaw-tool and build_tool.sh
- examples/blackbox_recipes/codex_swe_task/Dockerfile.codex-tool (native runtime).

Create examples/<NAME>_swe_task/Dockerfile.<NAME>-tool and build_tool.sh.
Pin versions, verify downloaded archive checksums, and target linux/amd64.
Install tool/runtime files under /opt/<NAME>; document remaining task-image dependencies.
Use FROM scratch and COPY --from=<builder-stage> /opt/<NAME>/ / by default;
if a different final base is needed, preserve this root layout and explain why.
Copy a launcher into bin/ only if needed. Mounts do not apply image ENV/ENTRYPOINT.
Use set -euo pipefail, the script directory as build context, TOOL_IMAGE/TOOL_TAG,
and --registry to build/tag/push. Pass --platform linux/amd64 to docker build.
Return a YAML snippet for the existing task's sandbox block: image_mounts with
image=registry.example.com/team/<NAME>:<TAG> and mount_path=/opt/<NAME>, plus
executable_paths mapping the command and required interpreters to their mounted paths.

Keep credentials out of files. Install dependencies during the image build,
not sandbox startup; runtime model/API calls may still require network access.
Do not modify existing examples, framework code, or tests; do not add an Agent.
Build locally, then inspect the image platform and size. Check its filesystem
with docker create IMAGE /unused (do not start it), docker export | tar -tf -, and
docker rm; clean up the inspection container even if inspection fails.
Report created files, mount YAML, exact validation commands/results, and assumptions.
```
