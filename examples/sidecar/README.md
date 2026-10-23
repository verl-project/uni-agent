# Shared Sidecar Scripts

Build, functionally check, and mount a new third-party tool in a real sandbox. Follow the [sidecar contribution guide](../../docs/source/quickstart/sidecar-contribution.md) for the complete workflow. Run commands from the source checkout root on Linux x86_64 with Python 3.11 and Docker.

| File | Purpose |
| --- | --- |
| [build.py](build.py) | Build from a custom manifest, override versions, base images and public package mirrors, and publish explicitly |
| [smoke.py](smoke.py) | Mount the tool read-only in a fresh container, disable external networking, run functional checks, and clean up resources |
| [check_entrypoint.py](check_entrypoint.py) | Execute the manifest's dependency probe and entrypoint, then verify the stdin/stdout byte contract |
| [check_model.py](check_model.py) | Use deterministic local Chat Completions responses to check real tool calls and stdio-wrapper completion |
| [validation.py](validation.py) | Prepare manifests, fixtures, and custom checks shared by local and remote validation |
| [test_sandbox.py](../../test_sandbox.py) | Reuse the tool check for real sandbox creation, mounting, entrypoint execution, SWE-ReX, port health, and cleanup |
| [sample/](sample/recipes.json) | Copy a minimal custom recipe illustrating versions, paths, the manifest, and the mount contract |
| [agent_template.py](agent_template.py) | Copy a template that connects a custom stdio tool to the framework Agent interface |

## Build and check your tool

Prepare `examples/my_tool/recipes.json` and the tool's required files, then run:

```bash
python3 examples/sidecar/build.py my-tool \
  --recipe-file examples/my_tool/recipes.json --describe
python3 examples/sidecar/build.py my-tool \
  --recipe-file examples/my_tool/recipes.json --no-cache --pull
python3 examples/sidecar/smoke.py my-tool \
  --recipe-file examples/my_tool/recipes.json --image "$LOCAL_SIDECAR_IMAGE" \
  --stdin-file examples/my_tool/fixtures/input.txt \
  --expected-stdout-file examples/my_tool/fixtures/expected.txt
```

Set `LOCAL_SIDECAR_IMAGE` to the reference printed by the completed build. The generic check executes `probe_argv` and `entrypoint`, requiring successful exit statuses, expected output, and successful cleanup. For model tools, replace the two fixture arguments with `--check-script examples/sidecar/check_model.py`; supply your own check script for other protocols. Checks accept `--case CASE_JSON` and print `{"result":"passed"}` only after every assertion passes. The guide defines the complete contract.

Override build configuration with `--version`, `--tag`, `--base-image`, and declared `--build-arg KEY=VALUE` arguments. `--dry-run` only prints commands. Publish with `--registry HOST/NAMESPACE --push`, then record the registry digest.

Real sandbox validation also accepts `--recipe-file`, fixtures, or `--check-script`. `--sidecar-recipe` names the custom recipe in that manifest, and `mount_target` selects its mount location. See the guide for SDK installation, credential preparation, and complete commands.
