# OpenCode 2.2.5 adapter

Select `name: opencode2` for the separately provisioned `@opencode/cli@2.2.5`.
Other versions fail before the prompt is sent. The existing `opencode` entry
continues to accept stable 1.18.x / 1.19.x. No executable is installed at runtime.

```yaml
agent:
  name: opencode2
  opencode_bin: /opt/opencode/bin/opencode
  run_timeout: 1800
  model:
    base_url: http://gateway:8000/sessions/SESSION_ID/v1
    model_name: policy
    api_key: EMPTY
```

The runner supplies the sandbox-reachable, session-scoped Chat Completions URL.
OpenCode owns the tool loop; the Task owns provisioning, scoring and teardown.
Only one user text message is supported. Completion is not a successful reward.

Each invocation uses `run --standalone --format json`, a private server, UUID
HOME/XDG/database directories, and explicit PWD. It does not connect to or stop
the user's background service. Native v2 `providers`, `permissions`, `policies`
and `agents` configuration uses the bundled OpenAI-compatible provider; no npm
provider download is required. Project configuration discovery and Claude skills
are disabled, and only the supplied provider is allowed. Title, summary,
compaction, sharing and warming are disabled. API keys are passed through an
environment reference and redacted from returned diagnostics.

Host-managed configuration remains authoritative in 2.2.5: HOME/XDG isolation
does **not** suppress `/etc/opencode`, macOS system configuration or MDM. Run in
a clean task container for reproducible policy/configuration. LocalSandbox is
best-effort process cleanup, not a containment boundary. Owners must cancel/join
exec callers, then await `stop()` or leave the Sandbox async context; see
[cleanup boundaries](PROCESS_CLEANUP.md).

## Provision the Linux amd64 baseline-musl sidecar

The existing sidecar launcher also supports the scoped v2 platform package.
Keep the package, version and SHA-512 together:

```sh
docker build -t ua-opencode2 \
  --build-arg OPENCODE_VERSION=2.2.5 \
  --build-arg OPENCODE_PACKAGE=@opencode/cli-linux-x64-baseline-musl \
  --build-arg OPENCODE_SHA512=bae1fce61b19330d03ff788ede0205d34c1197ef8e7d24673090055eae5c324927d32f744aed25c73ac6f73820bc9aab326fca8123136b884cccec9b687b22a4 \
  examples/sidecar/opencode
```

Mount the payload read-only at `/opt/opencode`, as for the 1.x recipe. The musl
loader's library path applies to the CLI only, preserving the task tool's own
`LD_LIBRARY_PATH`, including custom glibc libraries.

For **v2 only**, the task image must also supply `/usr/bin/python3` (3.10+).
The explicit musl loader becomes Bun's `process.execPath`, so upstream's built-in
standalone self-spawn would execute the loader without its binary. The sidecar
supervisor instead starts `serve --stdio --port 0`, keeps its stdin ownership
lease, and connects the CLI with `--server` and a generated password. Both
processes use the same bundled loader and inherit the original library
environment. Closing the lease stops the server, including when the supervisor
is killed. Native macOS executables use upstream's built-in `--standalone`.

## Verification

```sh
OPENCODE2_BIN=/absolute/path/to/opencode PYTHONPATH=. python -m pytest \
  tests/uni_agent/agents/test_opencode2_agent.py \
  tests/uni_agent/agents/test_opencode2_cli.py -q -s
```

Set `OPENCODE2_SIDECAR_BIN=/opt/opencode/bin/opencode` to run the four Linux
library-environment cases. They are skipped on a native macOS run. Tests use
scripted loopback SSE and dummy credentials: text, tool edits, upstream errors,
wrong versions, timeout, cancellation, repeated cancellation, concurrent state
isolation, conflicting project configuration, unrelated-server survival, and
independent Task patch verification. Process tests observe both private server
and tool descendants **before** any failure-only rescue. The workflow
`opencode2-integration.yml` executes the mounted Linux payload on Python 3.10/3.12.

This is pinned 2.2.5 support, not a claim about all 2.x releases, real model
quality, E2B, SWE-rebench, Ascend or training correctness. Those require separate
integration runs with their owners.

Implementation references: [v2.2.5 source](https://github.com/anomalyco/opencode/tree/v2.2.5),
[CLI](https://opencode.ai/v2/docs/cli/),
[configuration](https://opencode.ai/v2/docs/config/).
