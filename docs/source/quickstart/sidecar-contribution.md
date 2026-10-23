# Build and Integrate a Sidecar

Use this guide to add a tool such as OpenCode to Uni-Agent. The workflow consists of a build manifest, a tool wrapper, and one smoke test. Examples use `my-tool` as the recipe name and `my_tool` as the Agent name; replace both with your tool's name.

## 1. Package the tool

Work from the repository root on Linux x86_64 with Python 3.11 and Docker. Put the tool's Dockerfile, `.dockerignore`, `recipes.json`, and wrapper in `examples/my_tool/`.

Start with the [minimal manifest](https://github.com/verl-project/uni-agent/blob/main/examples/sidecar/sample/recipes.json) and [Dockerfile](https://github.com/verl-project/uni-agent/blob/main/examples/sidecar/sample/Dockerfile), then replace the sample installation and paths with your tool's implementation.

| Manifest field | Meaning |
| --- | --- |
| `schema_version`, `recipes` | Version `1` and a mapping from recipe names to configurations |
| `context`, `dockerfile` | Build paths relative to the manifest directory |
| `image`, `tag` | Image name without a registry prefix and its tag; the tag may reference `{TOOL_VERSION}` |
| `platform` | `linux/amd64` |
| `mount_target` | Absolute sandbox directory, such as `/opt/my-tool` |
| `entrypoint` | Tool wrapper argv, using paths under the mount directory |
| `probe_argv` | A short, non-interactive command that verifies the installed tool and required dependencies; return nonzero on failure |
| `version_arg`, `build_args` | Primary version key and string-valued build arguments, including `BASE_IMAGE` |

Pin the tool and runtime dependencies in the builder stage. Package the complete runtime, libraries, and data needed by the task image, and set executable permissions and interpreter paths correctly. Restrict `.dockerignore` to the files needed by the build.

The final image is a filesystem payload. Copy the tool directory's **contents** to the image root so mounting it at `/opt/my-tool` produces `/opt/my-tool/bin/...`:

```dockerfile
FROM scratch
COPY --from=builder /opt/my-tool/ /
```

Use `probe_argv` to check the actual installed executable, for example `/opt/opencode/bin/opencode --version`. Include dependency or wrapper startup assertions when required by your tool. The single smoke test below runs this probe from the mounted image.

## 2. Build and publish

The [build helper](https://github.com/verl-project/uni-agent/blob/main/examples/sidecar/build.py) resolves the manifest and constructs the Docker command:

```bash
python3 examples/sidecar/build.py my-tool \
  --recipe-file examples/my_tool/recipes.json --describe
python3 examples/sidecar/build.py my-tool \
  --recipe-file examples/my_tool/recipes.json --no-cache --pull
```

Use `--version`, `--base-image`, or a declared `--build-arg KEY=VALUE` to override build settings. `--dry-run` prints the command. Keep credentials out of build arguments and the image.

Publish a release tag to a registry the sandbox service can read:

```bash
docker login registry.example.com
python3 examples/sidecar/build.py my-tool \
  --recipe-file examples/my_tool/recipes.json \
  --registry registry.example.com/team --tag release-1 --push
docker image inspect registry.example.com/team/my-tool-sidecar:release-1 \
  --format '{{json .RepoDigests}}'
```

`--registry` selects the destination; `--push` uploads it. Configure pull access on the sandbox service too. Copy the matching `repository@sha256:...` reference from the output into `SIDECAR_IMAGE` for testing and task configuration.

## 3. Run the one smoke test

[test_sandbox.py](https://github.com/verl-project/uni-agent/blob/main/test_sandbox.py) is the only sidecar test entrypoint. It creates a sandbox, mounts the published sidecar, runs `probe_argv`, checks SWE-ReX and port forwarding, and confirms cleanup. It requests 2 CPUs and 4096 MB of memory.

In a Python 3.11 environment, install the validated SDK:

```bash
python -m pip install 'openyuanrong-sandbox==0.10.2rc9'
```

Prepare a task image with Bash, `python3`, pip, and a writable `/testbed`. It must be able to install `swe-rex==1.4.0`. Store these settings in a file outside the repository, with mode `0600`:

```text
AKERNEL_SERVER_ADDRESS='your-service-host:443'
AKERNEL_TOKEN='your-frontend-jwt'
```

Use the service's frontend JWT and a `host:port` address without a URL scheme. Trust the service CA on the validation host; the test enables certificate verification and uses TLS for a gateway on port 443.

```bash
set +x
set -a
. "$HOME/.config/uni-agent/openyuanrong.env"
set +a
export TASK_IMAGE=registry.example.com/team/task-runtime:validated
export SIDECAR_IMAGE=registry.example.com/team/my-tool-sidecar@sha256:REPLACE_WITH_PUBLISHED_DIGEST
python test_sandbox.py \
  --image "$TASK_IMAGE" --sidecar-image "$SIDECAR_IMAGE" \
  --sidecar-recipe my-tool --recipe-file examples/my_tool/recipes.json \
  --report sidecar-validation.json
```

Success requires exit status `0`, report `success=true`, and `stages.cleanup.confirmed=true`. The report records the installed SDK version and file hash, the sandbox ID, probe output, health checks, and cleanup. A failed probe or failed cleanup makes the test fail. If capacity is unavailable, retain the report and retry with a new report path when capacity is available.

## 4. Connect the Agent

The [Agent template](https://github.com/verl-project/uni-agent/blob/main/examples/sidecar/agent_template.py) launches a wrapper with this JSON on stdin:

```json
{
  "messages": [{"role": "user", "content": "Task description"}],
  "model": {"base_url": "Runtime endpoint", "api_key": "Runtime key", "model_name": "Runtime model"},
  "workdir": "/testbed"
}
```

The wrapper invokes the real tool and returns one JSON object on stdout:

```json
{"output": {"content": "Final result"}, "finished": true}
```

Send logs and raw CLI events to stderr. Set `finished=true` only after actual task completion, and return nonzero or `finished=false` for errors and unfinished runs. Preserve the complete model URL, including any `/sessions/<session_id>/v1` path. Add task-specific fields such as a submission to `output` when the scoring interface requires them.

For OpenCode, the wrapper can call `opencode run --format json`, configure its provider from the runtime model settings, and translate completion/error events into this result. See its [CLI](https://opencode.ai/docs/cli/) and [provider configuration](https://opencode.ai/docs/providers/#custom-provider) references.

Copy and register the adapter:

```bash
mkdir -p uni_agent/agents/my_tool
cp examples/sidecar/agent_template.py uni_agent/agents/my_tool/agent.py
touch uni_agent/agents/my_tool/__init__.py
```

Update the template's registration name and `ToolConfig.name`, and add `"my_tool": "uni_agent.agents.my_tool.agent"` to [`AGENT_MODULES`](https://github.com/verl-project/uni-agent/blob/main/uni_agent/agents/registry.py). The task owns sandbox creation and cleanup; the Agent uses the sandbox it receives.

Add the mount and Agent settings to the target task configuration. This example uses `swe_bench`; the task's data configuration supplies its project image and evaluation metadata:

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

Pass this file as [`run_task`](https://github.com/verl-project/uni-agent/blob/main/uni_agent/framework/task_runner.py)'s `task_config_path`. The framework supplies `agent.model` and configures the Gateway tunnel. For a TLS frontend and gateway, use:

```bash
export OPENYUANRONG_SERVER_ADDRESS="$AKERNEL_SERVER_ADDRESS"
export OPENYUANRONG_TOKEN="$AKERNEL_TOKEN"
export OPENYUANRONG_TLS=1
export OPENYUANRONG_GATEWAY_TLS=1
export OPENYUANRONG_TLS_VERIFY=1
export OPENYUANRONG_TUNNEL_SSL_VERIFY=1
```

Install framework dependencies following the [installation guide](installation.md). Run one normal task with the new Agent and confirm its completion status and task evaluation before scaling up. Submit the tool's Dockerfile, manifest, wrapper, task configuration, and smoke-test result together.
