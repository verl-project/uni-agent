# Build and Integrate a Third-Party Sidecar

This guide is for contributors adding a new tool to Uni-Agent, such as OpenCode. The deliverables are a tool image, validation cases, an Agent adapter, and task configuration. The examples use `my-tool` as the image recipe name and `my_tool` as the Agent registry name; replace both with `opencode` when adding OpenCode. The shared scripts load custom manifests, so adding a tool does not require changing their list of supported tool names.

## 1. Prepare the environment and directory

Run the commands from the root of a source checkout containing the [shared scripts](https://github.com/verl-project/uni-agent/blob/main/examples/sidecar/README.md). You need Linux x86_64, Python 3.11, an accessible Docker Engine, and network access to download base images and tool dependencies. Check these prerequisites first:

```bash
set -euo pipefail
test "$(uname -m)" = x86_64
python3 -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
docker info >/dev/null
test -f examples/sidecar/build.py
test -f examples/sidecar/validation.py
```

Keep the new tool's files in a dedicated directory:

```text
examples/my_tool/
  Dockerfile             # Install fixed versions of the tool and its runtime
  .dockerignore          # Allow only files needed by COPY
  recipes.json           # Build arguments, mount location, entrypoint, dependency probe
  run_agent.py           # Convert framework input into tool calls and normalize results
  checks.py              # Functional checks for a custom protocol, when needed
  fixtures/              # Check inputs and expected results
  task_config.yaml       # Mount and Agent configuration
  README.md              # Tool versions, extra dependencies, usage, validation results
```

The build host runs the shared scripts. The sidecar supplies the tool files. The task image supplies the shell, project directory, and system libraries required by the tool. Check scripts use `python3` from the task image; model tools may also bundle their own Python, Node, or other runtime. Building and deterministic checks require neither a GPU nor real model credentials.

## 2. Define the image and manifest

Create `recipes.json` in the new tool's directory. Copy the complete structure from the [minimal custom manifest](https://github.com/verl-project/uni-agent/blob/main/examples/sidecar/sample/recipes.json), then replace the installation, versions, paths, and entrypoints for your tool. The BusyBox example demonstrates the filesystem layout; use your new tool's own checks for acceptance.

| Field | Contract |
| --- | --- |
| `schema_version`, `recipes` | Use schema version `1`; `recipes` maps recipe names to configurations |
| `context`, `dockerfile` | Resolve paths relative to the manifest directory; keep the build context specific to this tool |
| `image`, `tag` | Image name without a registry prefix, and tag; the tag may reference a version in `build_args`, such as `{TOOL_VERSION}` |
| `platform` | The shared workflow currently supports `linux/amd64` |
| `mount_target` | Absolute sandbox directory, such as `/opt/opencode` |
| `entrypoint` | Argument array, such as `["/opt/opencode/bin/python", "/opt/opencode/bin/run_agent.py"]` |
| `probe_argv` | Argument array that checks executability, versions, and dependencies; return `0` on success and nonzero for missing dependencies or version mismatches |
| `version_arg` | Key for the primary tool version in `build_args`, such as `TOOL_VERSION` |
| `build_args` | String-to-string mapping containing at least `BASE_IMAGE` and the primary tool version |

Pin the base image, tool version, and dependency versions in the tool's Dockerfile, and run version or dependency checks during the build. Use `FROM scratch` for the final stage and copy the **contents** of the complete tool directory into the image root:

```dockerfile
# The builder has installed the tool and its complete runtime in /opt/my-tool.
FROM scratch
COPY --from=builder /opt/my-tool/ /
```

This is the final-stage fragment; provide the installation steps in the tool's builder stage. For example, placing `bin/` at the image root produces `/opt/my-tool/bin/` after mounting at `/opt/my-tool`. The image needs no startup command: the shared validation scripts mount and execute its files.

Bundle the interpreter, native binaries, shared libraries, data files, and caches needed for offline startup. Set executable permissions, shebangs, runtime prefixes, and library search paths correctly. Build a Python venv at its final mount prefix, or use a relocatable runtime. Host libraries do not automatically appear in the task sandbox: dynamically linked programs need compatible libc and loaders, while static binaries have no such dynamic-linking requirement. Allow only required sources and entrypoints through `.dockerignore`, and keep credential files outside the directory.

## 3. Build your image

The commands below target the directory you created. Use your tool's own manifest and recipe name after replacing the example implementation:

```bash
python3 examples/sidecar/build.py my-tool \
  --recipe-file examples/my_tool/recipes.json --describe
python3 examples/sidecar/build.py my-tool \
  --recipe-file examples/my_tool/recipes.json --no-cache --pull
RESOLVED_RECIPE=$(python3 examples/sidecar/build.py my-tool \
  --recipe-file examples/my_tool/recipes.json --describe)
LOCAL_SIDECAR_IMAGE=$(printf '%s' "$RESOLVED_RECIPE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["image_ref"])')
export LOCAL_SIDECAR_IMAGE
```

`--describe` prints the resolved configuration, and `--dry-run` prints commands without executing them. `--version` overrides the version selected by `version_arg`; `--tag` overrides the tag; `--base-image` overrides the base image. `--build-arg KEY=VALUE` accepts only keys declared in the manifest. Use `--pip-index` or `--npm-registry` for public package mirrors after declaring the corresponding argument in the recipe.

Run subsequent commands in order in the same shell. When overriding versions, pass the same arguments to the build and `--describe` so the resolved reference matches the image you built. Update the installation directory, mount directory, and entrypoint together. For reproducible releases, record the source SHA, base-image digest, tool versions, complete dependency lock, and final registry digest.

## 4. Check the tool entrypoint

### Tools with deterministic input and output

The generic check first executes `probe_argv`, then executes `entrypoint` with the input file's raw bytes on stdin. It compares stdout byte for byte, including the final newline:

```bash
python3 examples/sidecar/smoke.py my-tool \
  --recipe-file examples/my_tool/recipes.json \
  --image "$LOCAL_SIDECAR_IMAGE" \
  --stdin-file examples/my_tool/fixtures/input.txt \
  --expected-stdout-file examples/my_tool/fixtures/expected.txt
```

The final JSON line should include `"smoke": "passed"` and `"cleanup": "passed"`. The script exports the sidecar, mounts it read-only in a fresh container, disables external networking, and removes its temporary containers when finished. Incorrect output or a failed check returns nonzero. Use `--runtime-image "$TASK_IMAGE"` to check compatibility with the actual task image; that image must provide `python3`.

### Model tools such as OpenCode

First make the model tool's wrapper implement the stdio contract in section 7. Chat Completions tools can use the shared [model check](https://github.com/verl-project/uni-agent/blob/main/examples/sidecar/check_model.py). It starts a deterministic local model endpoint, supports streaming and non-streaming requests, and requires the real tool to execute a shell command and report completion. No real model credentials are needed:

```bash
python3 examples/sidecar/smoke.py my-tool \
  --recipe-file examples/my_tool/recipes.json \
  --image "$LOCAL_SIDECAR_IMAGE" \
  --check-script examples/sidecar/check_model.py
```

The shared model check supports `bash`, `exec_command`, and `shell_command` tool calls. For another protocol or tool definition, implement `checks.py` in the new tool's directory and pass its path to `--check-script`. The check must use only the Python standard library and accept `--case CASE_JSON`. The JSON provides `recipe`, `mount_target`, `entrypoint`, and `probe_argv`. When both fixture arguments are supplied, it also provides `stdin_base64` and `expected_stdout_base64`. The check runs inside the container and can access its localhost, task files, and read-only sidecar.

Start a deterministic local model endpoint in the check, make the real tool execute an assertable command such as writing `/testbed/probe.txt`, and assert successful completion, correct file contents, and the expected model requests. Only after every assertion passes should the check print a final line containing `{"result":"passed"}` and exit `0`. On failure, return nonzero; close the local service and reap child processes. Running only `--version` checks startup and dependencies, not model functionality. The shared scripts require both a successful exit status and a successful final JSON result.

OpenCode's non-interactive entrypoint is `opencode run --format json`, with `--model provider/model` selecting the model. The wrapper must parse events from the pinned version and distinguish success, errors, and unfinished runs. Read the provider's `baseURL` and `apiKey` from the current request, and preserve the supplied model name and complete session path. Select the provider package for the protocol: `@ai-sdk/openai-compatible` for Chat Completions, or `@ai-sdk/openai` for Responses. Pass runtime configuration through `OPENCODE_CONFIG_CONTENT`. See the [CLI](https://opencode.ai/docs/cli/), [provider](https://opencode.ai/docs/providers/#custom-provider), and [configuration](https://opencode.ai/docs/config/) references.

## 5. Publish the image and pin its digest

The sandbox service must pull the image from a registry. Authenticate using your organization's supported method, then publish a dedicated tag for the new tool:

```bash
docker login registry.example.com
export SIDECAR_REGISTRY=registry.example.com/team
SIDECAR_TAG=$(printf '%s' "$RESOLVED_RECIPE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag"])')
export SIDECAR_TAG
python3 examples/sidecar/build.py my-tool \
  --recipe-file examples/my_tool/recipes.json \
  --registry "$SIDECAR_REGISTRY" --tag "$SIDECAR_TAG" --push
SIDECAR_NAME=$(printf '%s' "$RESOLVED_RECIPE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["image"])')
export SIDECAR_REPOSITORY="$SIDECAR_REGISTRY/$SIDECAR_NAME"
SIDECAR_IMAGE=$(
  docker image inspect "$SIDECAR_REPOSITORY:$SIDECAR_TAG" --format '{{json .RepoDigests}}' |
  python3 -c 'import json,sys; print(next(x for x in json.load(sys.stdin) if x.startswith(sys.argv[1]+"@")))' \
    "$SIDECAR_REPOSITORY"
)
export SIDECAR_IMAGE
```

`--registry` selects the destination name; `--push` uploads it. Configure registry pull permissions on the service as well. A successful build, version probe, or local login does not establish that the service can pull the image. Record the complete `repository@sha256:...` reference and use that immutable reference for subsequent mounts and acceptance checks.

## 6. Validate the custom sidecar in a real sandbox

Prepare an available OpenYuanRong service, a frontend JWT, and a task image. The task image needs Linux amd64, Bash, `python3`, pip, a writable `/testbed`, and the ability to install `swe-rex`. Keep the complete checkout, tool directory, manifest, and fixtures on the Linux validation host, then configure the SDK:

```bash
SIDECAR_VENV="$HOME/.venvs/sidecar-validation"
if [ ! -e "$SIDECAR_VENV" ] && [ ! -L "$SIDECAR_VENV" ]; then
  python3 -m venv "$SIDECAR_VENV"
fi
. "$SIDECAR_VENV/bin/activate"
python -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
python -m pip install 'openyuanrong-sandbox==0.10.2rc9'
python -c 'import importlib.metadata, yr_sandbox; assert importlib.metadata.version("openyuanrong-sandbox") == "0.10.2rc9"; print(yr_sandbox.__file__)'
```

Check an existing environment before reusing it. Store service credentials in a file outside the repository, populated through an editor or secure channel. Keep credentials out of build arguments, images, command-line arguments, and version control:

```bash
install -d -m 700 "$HOME/.config/uni-agent"
touch "$HOME/.config/uni-agent/openyuanrong.env"
chmod 600 "$HOME/.config/uni-agent/openyuanrong.env"
${EDITOR:-vi} "$HOME/.config/uni-agent/openyuanrong.env"
```

Provide `AKERNEL_SERVER_ADDRESS='your-service-host:443'` and `AKERNEL_TOKEN='your-frontend-jwt'` in the file. The service address has no URL scheme, and a short ACL identifier cannot replace the JWT. For port 443, the script sets `gateway_use_tls=True`. Certificate verification is enabled by default; add private CAs to the validation host's trust configuration.

```bash
set +x
set -a
. "$HOME/.config/uni-agent/openyuanrong.env"
set +a
export TASK_IMAGE=registry.example.com/team/task-runtime:validated
python test_sandbox.py \
  --image "$TASK_IMAGE" \
  --sidecar-recipe my-tool --sidecar-image "$SIDECAR_IMAGE" \
  --recipe-file examples/my_tool/recipes.json \
  --stdin-file examples/my_tool/fixtures/input.txt \
  --expected-stdout-file examples/my_tool/fixtures/expected.txt \
  --swe-rex-spec 'swe-rex==1.4.0' \
  --report artifacts/my-tool-remote.json
```

For model tools, replace the two fixture arguments with `--check-script examples/sidecar/check_model.py`, or with the tool's own check. Local and remote validation use the same check file and manifest. The mount location comes from `mount_target` rather than a tool-name lookup.

Acceptance requires exit status `0`, `success=true` in the report, successful creation, command execution, tool entrypoint, SWE-ReX installation, internal and external port-8000 health checks, and final cleanup with `cleanup.confirmed=true`. The report records the installed SDK version, file path, SHA-256, and CPU/memory request. The default request is `2000m/4096MB`; override it explicitly with `--cpu` and `--memory`. If capacity is unavailable, retain the report and retry with a new report path once a node can satisfy the request.

## 7. Adapt the Agent and task configuration

After validating the sidecar mount, connect the tool to Uni-Agent's Agent interface. The [Agent template](https://github.com/verl-project/uni-agent/blob/main/examples/sidecar/agent_template.py) defines a reusable stdio contract:

```json
{
  "messages": [{"role": "user", "content": "Task description"}],
  "model": {"base_url": "Model endpoint for this run", "api_key": "Runtime API key", "model_name": "Model name for this run"},
  "workdir": "/testbed"
}
```

`run_agent.py` reads this JSON from stdin, invokes the tool's real CLI, and writes **one complete JSON object** to stdout. The wrapper parses raw CLI events or sends events, progress, and logs to stderr. The normalized stdout result is:

```json
{"output": {"content": "Final result"}, "finished": true}
```

`finished` must be a Boolean and may be true only when the task has actually completed. A successful process exit cannot replace a completion assertion. For CLI errors, timeouts, or missing completion events, return nonzero or `finished=false`. The OpenCode wrapper translates its events into this structure. Put task-specific patch, submission, or other fields in `output` as required by the task's scoring interface.

Copy the template and register the new Agent:

```bash
mkdir -p uni_agent/agents/my_tool
cp examples/sidecar/agent_template.py uni_agent/agents/my_tool/agent.py
touch uni_agent/agents/my_tool/__init__.py
```

Set the template's registration name and `ToolConfig.name` to the new tool's name, then add `"my_tool": "uni_agent.agents.my_tool.agent"` to the existing [`AGENT_MODULES`](https://github.com/verl-project/uni-agent/blob/main/uni_agent/agents/registry.py) dictionary. Use `opencode` in all three positions when adding OpenCode. The adapter uses the sandbox supplied by the framework; the task owns its lifecycle.

Merge the following configuration into the target task's `task_config.yaml`, using the verified image and entrypoint. The file is a list keyed by task name; this example targets the existing `swe_bench` task. Its data configuration still supplies the task image, project, and evaluation metadata:

```yaml
- name: swe_bench
  sandbox:
    provider: openyuanrong
    image_mounts:
      - mount_path: /opt/my-tool
        image: registry.example.com/team/my-tool-sidecar@sha256:REPLACE_WITH_VERIFIED_DIGEST
    sandbox_kwargs:
      proxy_port: 38197
  agent:
    name: my_tool
    command: [/opt/my-tool/bin/python, /opt/my-tool/bin/run_agent.py]
    run_timeout: 600
```

Use this configuration after replacing the example with a real model tool; the minimal BusyBox example does not implement the model protocol. The host framework instantiates the Agent and passes `agent.model` through to the wrapper. Set [`run_task`](https://github.com/verl-project/uni-agent/blob/main/uni_agent/framework/task_runner.py)'s `task_config_path` to this file. That entrypoint also injects the tunnel upstream for the current Gateway session while preserving the complete `/sessions/<session_id>/v1` model URL path. Set `OPENYUANRONG_SERVER_ADDRESS` and `OPENYUANRONG_TOKEN` to the verified service address and frontend JWT before running a task with this provider.

Before launching a task, check lazy registration and configuration parsing in a fresh process. Confirm that the tool name, entrypoint, and mount directory agree:

```bash
python - <<'PY'
from uni_agent.agents import get_agent_cls
from uni_agent.sandbox import SandboxConfig
from uni_agent.tasks import TaskConfigResolver
task = TaskConfigResolver.from_file("examples/my_tool/task_config.yaml").resolve({"name": "swe_bench"})
agent_cls = get_agent_cls(task["agent"]["name"])
config = agent_cls.config_model(**task["agent"])
sandbox_config = SandboxConfig.model_validate(task["sandbox"])
target = sandbox_config.image_mounts[0].mount_path
assert config.command[0].startswith(target + "/")
print("AGENT_REGISTRATION_AND_CONFIG_OK")
PY
export OPENYUANRONG_SERVER_ADDRESS="$AKERNEL_SERVER_ADDRESS"
export OPENYUANRONG_TOKEN="$AKERNEL_TOKEN"
export OPENYUANRONG_TLS=1
export OPENYUANRONG_GATEWAY_TLS=1
export OPENYUANRONG_TLS_VERIFY=1
export OPENYUANRONG_TUNNEL_SSL_VERIFY=1
```

These connection settings assume a TLS frontend and TLS gateway, with certificates validated against the host's trusted CAs; adjust them for other service deployments. This check only loads configuration and does not run a model task. Install the framework dependencies described in the [installation guide](installation.md); an SDK-only environment is insufficient for a full task. Run one task from the target dataset first. Check real CLI execution, task changes, agreement between `AgentResult.finished` and tool completion, completed reward/evaluation, and sandbox cleanup before increasing concurrency or starting training.

## 8. Before submitting the new tool

- Include a fixed-version build, manifest, wrapper, functional checks, fixtures, task configuration, and tool README in the new tool's directory.
- Build and validate the new tool. Use the same source and published digest for local checks, real sandbox mounts, and target-task reports.
- Map the model protocol, runtime dependencies, and task-scoring fields explicitly. Errors and unfinished runs must not report success.
- Keep credentials out of images, source, and logs. Retain reports showing successful stages, failed stages, and cleanup results.

For `Exec format error`, check the architecture. If an existing file reports that it cannot be found, check its shebang, dynamic loader, and shared libraries. For `MANIFEST_UNKNOWN`, check publishing and the digest. For model HTTP 404 errors, check the protocol and session path. For `ERR_RESOURCE_NOT_ENOUGH`, check that a single node can satisfy both the CPU and memory request.
