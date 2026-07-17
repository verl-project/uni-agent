# Gateway Debug Launcher

`debug_launcher.py` starts the existing gateway without training, creates one
external session, prints connection URLs, and writes finalized trajectories when
the session is released. It is meant for local gateway debugging and interaction
collection. It also writes a debug snapshot with the normalized message history
and pre-finalize session state so token-level trajectories can be interpreted
after the run.

It does not add a gateway HTTP control plane, does not create a training
framework mode, and does not add a sessionless `/v1/messages` facade.

## Local Backend

For real local inference, start an OpenAI-compatible completions backend first.
For example:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /data1/models/Qwen/Qwen3.5-4B \
  --host 127.0.0.1 \
  --port 38315 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 4096 \
  --trust-remote-code
```

The launcher calls `{backend_base_url}/completions` with token-id prompts and
`return_token_ids=true`. The response must include `choices[0].token_ids`.
Missing token ids are a hard failure because trajectory response tokens must
come from backend `TokenOutput`, not from text re-tokenization.

Before starting the launcher, verify that the backend is ready:

```bash
python - <<'PY'
import httpx

r = httpx.get("http://127.0.0.1:38315/v1/models", timeout=5)
print(r.status_code, r.text[:300])
PY
```

The launcher should point at the same host and port:

```text
--backend-base-url http://127.0.0.1:38315/v1
```

## Manual Claude Code Session

Start the launcher without `--run-claude`:

```bash
python examples/gateway/debug_launcher.py \
  --backend openai-completions \
  --backend-base-url http://127.0.0.1:38315/v1 \
  --backend-model /data1/models/Qwen/Qwen3.5-4B \
  --tokenizer /data1/models/Qwen/Qwen3.5-4B \
  --session-id manual-cc-001 \
  --output-dir /tmp/uni-agent-gateway-debug \
  --response-length 64
```

The launcher prints per-session URLs:

```text
Claude Code:
  export ANTHROPIC_BASE_URL=http://host:port/sessions/manual-cc-001
  export ANTHROPIC_API_KEY=dummy-local-key
OpenAI-compatible:
  base_url=http://host:port/sessions/manual-cc-001/v1
```

In another terminal, use the printed values:

```bash
export ANTHROPIC_BASE_URL=http://host:port/sessions/manual-cc-001
export ANTHROPIC_API_KEY=dummy-local-key
unset ANTHROPIC_AUTH_TOKEN
unset CLAUDE_CODE_API_BASE_URL
unset CLAUDE_CODE_API_KEY
export NO_PROXY=host,127.0.0.1,localhost

claude --bare --model claude-sonnet-4-5
```

Do not use `-p` / `--print` for a manual interactive session. `-p` sends one
prompt, prints the response, and exits.

When finished, return to the launcher terminal and press `Enter` or `Ctrl-C`.
This is the normal finalize path: the launcher finalizes the session and writes
trajectories, session metadata, and the debug snapshot.

## One-Shot Claude Code Smoke

Use `--run-claude` when you want the launcher to run one Claude Code prompt and
then finalize automatically:

```bash
python examples/gateway/debug_launcher.py \
  --backend openai-completions \
  --backend-base-url http://127.0.0.1:38315/v1 \
  --backend-model /data1/models/Qwen/Qwen3.5-4B \
  --tokenizer /data1/models/Qwen/Qwen3.5-4B \
  --session-id claude-smoke-001 \
  --output-dir /tmp/uni-agent-gateway-debug \
  --run-claude \
  --claude-prompt "Reply with OK only." \
  --claude-model claude-sonnet-4-5 \
  --claude-timeout 180 \
  --response-length 64
```

The equivalent manual one-shot command is:

```bash
claude --bare --no-session-persistence \
  -p "Reply with OK only." \
  --output-format text \
  --model claude-sonnet-4-5
```

`--response-length` is important for small local backends. Claude Code can ask
for a large `max_tokens` value such as `32000`; the session response budget
clamps that request before it reaches the backend.

## Direct API Smoke

You can also call the Anthropic Messages endpoint directly:

```bash
curl "$ANTHROPIC_BASE_URL/v1/messages" \
  -H "x-api-key: dummy-local-key" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5",
    "max_tokens": 32,
    "messages": [{"role": "user", "content": "Reply with OK only."}]
  }'
```

The OpenAI-compatible URL printed by the launcher keeps the `/v1` suffix:

```text
http://host:port/sessions/manual-cc-001/v1
```

The Claude Code base URL must not include `/v1`:

```text
http://host:port/sessions/manual-cc-001
```

Do not use `CLAUDE_CODE_API_BASE_URL` for this Anthropic Messages proxy path.

## Scope and Known Gaps

The launcher is a local debug entrypoint, not a general standalone gateway
service. It creates one session up front and writes artifacts only when that
session is finalized.

The documented smoke paths cover connection setup, text generation, trajectory
writing, and Claude Code CLI availability. Gateway-level automated tests cover
OpenAI tool calls and Anthropic `tool_use` round trips, but this launcher does
not currently provide a full Claude Code tool-workflow recipe. Use a dedicated
Claude Code SWE recipe or manual smoke when validating complex tool behavior.

## Model Arguments

`--model claude-sonnet-4-5` in a manual Claude Code command, and
`--claude-model claude-sonnet-4-5` in one-shot mode, are Claude Code frontend
model strings. They are not the local inference backend model.

For local collection:

- Claude Code uses that value when constructing the Anthropic Messages request.
- The gateway forwards the request through the local backend adapter.
- The actual local inference model is selected by `--backend-model`.

`claude-sonnet-4-5` was accepted by the locally tested Claude Code 2.1.177
client. If a different Claude Code version rejects that name before sending the
request, use a model name accepted by that CLI version. The local backend will
still be controlled by `--backend-model`.

## Output Files

For `--output-dir /tmp/uni-agent-gateway-debug` and
`--session-id manual-cc-001`, outputs are written under:

```text
/tmp/uni-agent-gateway-debug/manual-cc-001/
```

Files:

- `trajectories.jsonl`: one JSON record per finalized trajectory.
- `session_metadata.json`: launcher metadata, provider URLs, Claude return code,
  artifact paths, and trajectory count. This exists even when zero trajectories
  were recorded.
- `debug_snapshot.json`: provider URLs, normalized message history,
  active tool schemas, and session state captured immediately before
  finalization. This is a diagnostic artifact, not a training trajectory schema.
- `claude.stdout`: captured Claude Code stdout in one-shot mode.
- `claude.stderr`: captured Claude Code stderr in one-shot mode.
- `claude-debug.log`: Claude Code debug log in one-shot mode.

Each trajectory record has `schema_version: 1` and stores the finalized
`Trajectory` fields:

- `prompt_ids`
- `response_ids`
- `response_mask`
- `response_logprobs`
- `reward_info`
- `reward_score`
- `num_turns`
- `multi_modal_data`
- `extra_fields`

The trajectory records intentionally store token-level training data. Use
`debug_snapshot.json` when you need the normalized conversation history that
produced those tokens.

## Fake Backend

For a no-GPU smoke:

```bash
python examples/gateway/debug_launcher.py \
  --backend fake \
  --session-id fake-debug-001 \
  --output-dir /tmp/uni-agent-gateway-debug
```

The fake backend always returns token ids for `OK`. It is useful for checking
Claude Code connectivity and trajectory writing without spending Anthropic API
quota or starting a local inference server.

## Common Failures

If Claude Code does not hit the gateway:

- Use `ANTHROPIC_BASE_URL=http://host:port/sessions/{session_id}`.
- Do not append `/v1` to `ANTHROPIC_BASE_URL`.
- Unset `CLAUDE_CODE_API_BASE_URL`, `CLAUDE_CODE_API_KEY`, and
  `ANTHROPIC_AUTH_TOKEN`.
- Set `NO_PROXY` for the printed host, `127.0.0.1`, and `localhost`.

If vLLM rejects `max_tokens`:

- Add `--response-length 64` or another local budget appropriate for the model.

If Claude Code prints `API Error: 500 ConnectError: All connection attempts failed`:

- Claude Code reached the gateway.
- The gateway could not connect to `--backend-base-url`.
- Start vLLM, or fix the host/port, then verify `/v1/models` returns 200.

If `trajectories.jsonl` is empty:

- Check `session_metadata.json` first.
- Inspect `debug_snapshot.json` to confirm which messages reached the gateway.
- Then inspect `claude.stdout`, `claude.stderr`, and `claude-debug.log` if
  one-shot mode was used.

Never print or commit real API keys. Use `dummy-local-key` for local gateway
debugging.
