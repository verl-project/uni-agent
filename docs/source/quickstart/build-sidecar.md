# Build a Sidecar Image

A sidecar is a [runtime image](../concepts/sandbox.md#runtime-images) that contains a tool and its dependencies.
Uni-Agent mounts the image's files into a task sandbox through `sandbox.image_mounts`.
The files appear under the configured `mount_path`, without changing the task image.

Docker, Modal, and OpenYuanRong support image mounts. Local, veFaaS, and E2B do not.

Build with Docker and Bash for `linux/amd64`. On ARM machines, enable amd64 emulation in Docker before building.

## Recipe layout

The example tool is called `my-tool`. Directory names use `my_tool`.
The commands below use `examples/my_tool_swe_task/`.
You can also place the recipe under `examples/blackbox_recipes/`, alongside the example you copy.

Create two build files:

- `Dockerfile.<name>-tool`: defines the installation steps and runtime files.
- `build_tool.sh`: builds the image and optionally pushes it to a registry.

The Dockerfile name passed to `docker build -f` must match the file you create.
Choose an existing Dockerfile and build script as a starting point:

| Runtime | Example Dockerfile | Build script | Optional launcher |
| --- | --- | --- | --- |
| Python | [Mini-SWE-Agent](https://github.com/verl-project/uni-agent/blob/main/examples/mini_swe_agent/Dockerfile.mini-swe-agent-tool) | [build_tool.sh](https://github.com/verl-project/uni-agent/blob/main/examples/mini_swe_agent/build_tool.sh) | `run_agent.py` |
| Python | [Hermes](https://github.com/verl-project/uni-agent/blob/main/examples/blackbox_recipes/hermes_swe_task/Dockerfile.hermes-tool) | [build_tool.sh](https://github.com/verl-project/uni-agent/blob/main/examples/blackbox_recipes/hermes_swe_task/build_tool.sh) | `run_hermes.py` |
| Node.js | [OpenClaw](https://github.com/verl-project/uni-agent/blob/main/examples/blackbox_recipes/openclaw_swe_task/Dockerfile.openclaw-tool) | [build_tool.sh](https://github.com/verl-project/uni-agent/blob/main/examples/blackbox_recipes/openclaw_swe_task/build_tool.sh) | None |
| Native binary | [Codex](https://github.com/verl-project/uni-agent/blob/main/examples/blackbox_recipes/codex_swe_task/Dockerfile.codex-tool) | [build_tool.sh](https://github.com/verl-project/uni-agent/blob/main/examples/blackbox_recipes/codex_swe_task/build_tool.sh) | None |
| Node.js | [Claude Code](https://github.com/verl-project/uni-agent/blob/main/examples/claude_code_swe_task/Dockerfile.claude-code-tool) | [build_tool.sh](https://github.com/verl-project/uni-agent/blob/main/examples/claude_code_swe_task/build_tool.sh) | None |

Codex uses Node.js only during the build. Its sidecar contains native binaries.

If your tool needs a launcher script, copy it into `/opt/my-tool/bin`.
Otherwise, remove the launcher `COPY` line from the Dockerfile you copied.

## Adapt the Dockerfile

Change the installation commands for your tool and pin package versions.
Install its runtime and dependencies under `/opt/my-tool`.
In a Dockerfile comment, list any dependencies the task image must provide.
For example, OpenClaw needs Python and glibc 2.35 or newer in the task image.

A typical final stage is:

```dockerfile
FROM scratch
COPY --from=builder /opt/my-tool/ /
```

`FROM scratch` starts with an empty image, so this stage keeps only the copied files.
You may choose another final base, but keep the same layout.
The contents of `/opt/my-tool/` go at the image root.
After mounting, `bin/my-tool` becomes `/opt/my-tool/bin/my-tool`.
Rename `builder` in `COPY --from=builder` if your build stage has a different name.

The sandbox mounts only the files. It does not apply the image's `ENV` settings or run its `ENTRYPOINT`.

## Adapt and run build_tool.sh

Edit the copied build script:

- Update the default image name and the tool-specific build arguments.
- Use the script's own directory as the Docker build context.
- Add `--platform linux/amd64` to `docker build`.
- If you copied OpenClaw's script, replace `tool` in `--target tool` with your final stage's name, or remove the `--target` option.

Use the `TOOL_IMAGE` and `TOOL_TAG` environment variables to set the image name and tag.
All five scripts accept `--registry`.
Other flags depend on the script.
Mini-SWE-Agent has `--pip-index`. OpenClaw has `--tag`, `--version`, and `--network`.

Run from the repository root:

```bash
# Build locally.
TOOL_IMAGE=my-tool TOOL_TAG=v1 bash examples/my_tool_swe_task/build_tool.sh
# Optional: build and push.
docker login registry.example.com
TOOL_IMAGE=my-tool TOOL_TAG=v1 bash examples/my_tool_swe_task/build_tool.sh \
  --registry registry.example.com/team
```

Passing `--registry` builds the image and pushes it to that registry.
The example above publishes `registry.example.com/team/my-tool:v1`.

## Verify the image

Inspect the files without starting the container. No shell is needed inside the image.
The Dockerfile above has no default command, so pass `/unused` to `docker create` as a placeholder.
The container is never started, and `/unused` is never executed.

```bash
check=$(docker create --platform linux/amd64 my-tool:v1 /unused)
docker export "$check" | tar -tf - | sed -n '1,20p'
docker rm "$check"
docker image inspect my-tool:v1 --format '{{.Os}}/{{.Architecture}}'
```

For the layout shown above, expect `bin/...` and any launcher at the top level.
If paths start with `opt/my-tool/`, check the final `COPY` instruction.
The platform output should be `linux/amd64`.

These checks confirm only the layout and platform.
To check that the tool itself works, follow the README of the recipe you copied.

## Mount the image

Add these fields to the task config's existing `sandbox` block. Keep its task image setting.
Use the full address of the published sidecar image. The sandbox service must be able to pull it.
`image_map` rewrites only the task image, so enter the sidecar address directly in `image_mounts`.

To run a mounted executable by name, add it to `executable_paths`.
Include interpreters such as `node` if the tool needs them:

```yaml
sandbox:
  provider: openyuanrong
  image_mounts:
    - image: registry.example.com/team/my-tool:v1
      mount_path: /opt/my-tool
  executable_paths:
    my-tool: /opt/my-tool/bin/my-tool
```

See [Launch a Sandbox](launch-sandbox.md) for provider setup and [Runtime images](../concepts/sandbox.md#runtime-images) for mount configuration.
For a complete workflow, follow the [Codex recipe](https://github.com/verl-project/uni-agent/blob/main/examples/blackbox_recipes/codex_swe_task/README.md#build-the-sidecar).

## Create the recipe with a coding agent

Fill in the inputs and paste this prompt into a coding agent:

```text
Add a sidecar recipe to this uni-agent checkout.

Inputs:
- name: <NAME>
- version: <VERSION>
- runtime: <Python / Node.js / native binary>
- source: <package or URL, plus archive checksum>
- launcher: <relative path, or none>
- task image: <distro, libc version, available interpreters>
- tag: <TAG>

Read the repository example that matches your tool's runtime:
- examples/mini_swe_agent/Dockerfile.mini-swe-agent-tool and build_tool.sh (Python)
- examples/blackbox_recipes/openclaw_swe_task/Dockerfile.openclaw-tool and build_tool.sh (Node.js)
- examples/blackbox_recipes/codex_swe_task/Dockerfile.codex-tool (native binary)

Create examples/<NAME>_swe_task/ with Dockerfile.<NAME>-tool and build_tool.sh.
Pin versions, verify downloaded archive checksums, and target linux/amd64.
Install tool and runtime files under /opt/<NAME>.
List any dependencies the task image must provide in Dockerfile comments.
Use FROM scratch followed by COPY --from=<builder-stage> /opt/<NAME>/ /.
If another final base is needed, keep this root layout and explain the choice.
Copy a launcher into bin/ only if needed.
Mounting the image does not apply its ENV or run its ENTRYPOINT.

In build_tool.sh, use set -euo pipefail and the script directory as build context.
Read the image name and tag from the TOOL_IMAGE and TOOL_TAG environment variables.
Support --registry to build, tag, and push.
Pass --platform linux/amd64 to docker build.
Return a YAML snippet for the existing task's sandbox block:
- image_mounts: image=registry.example.com/team/<NAME>:<TAG>, mount_path=/opt/<NAME>
- executable_paths: map the command and any interpreters to their mounted paths.

Keep credentials out of files. Install tool dependencies during the image build.
Do not install packages when the sandbox starts. The tool may still call model APIs.
Create only the recipe files and mount YAML. Do not modify existing examples,
framework code, or tests. Do not add uni_agent agent adapters.

Build locally, then check the image without starting the inspection container:
docker image inspect "<NAME>:<TAG>" --format '{{.Os}}/{{.Architecture}}'
check=$(docker create --platform linux/amd64 "<NAME>:<TAG>" /unused)
docker export "$check" | tar -tf -
docker rm "$check"
Remove the inspection container even if a check fails.
Report the created files, mount YAML, image size, exact commands and results,
and any assumptions.
```
