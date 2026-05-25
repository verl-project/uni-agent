# `lark_chat` — long-running Lark chat agent

A long-running process that listens for IM messages on Lark / Feishu,
dispatches each user message to a multi-step agent loop running on a
shared sandbox env, and replies back through `lark-cli`. Each chat is a
real **ongoing conversation**: the OpenAI-shaped message log is
persisted per `chat_id` and trimmed on a sliding window so you can
talk to the same bot across many turns and across process restarts.

The agent also keeps a **persistent user profile / preferences memory**
under a `memory_dir` (default `/workspace/memory/` for `local_attach`,
`~/.uni-agent/app/lark_chat/memory/` for `local_native`), written by
the model itself, so it accumulates a real picture of the user across
conversations (not just within a single chat thread).

Two deployments are supported, each with its own config file:

- **`local_native`** — agent runs shell commands directly against the
  host. **Unsafe on a personal machine** — only use this when the
  environment you're running in is already a sandbox / VM / container.
  Config: [`config.local_native.yaml`](./config.local_native.yaml) —
  picked up when `--config` is omitted because it requires zero
  bootstrap.
- **`local_attach`** — agent runs inside a user-managed Docker container;
  only the directories you bind-mount are reachable. Safer choice on a
  personal machine, **recommended**. Config:
  [`config.local_attach.yaml`](./config.local_attach.yaml) (pass via
  `--config`).

Sits on top of the same `AgentInteraction` loop as `examples/lark/demo.py`,
but evolves it from "one user request → one run → exit" to
"listener → many turns → many runs". The framework loop itself was
extended to support **multiple tool calls per assistant response** and
to recognize **`finish` (preferred) or a tool-call-less assistant
response (fallback)** as end-of-turn — see
`uni_agent/interaction/interaction.py`.

## Architecture

```
                ┌─────────────────────────────────────────────────────────┐
                │  host process: app.lark_chat.main                       │
                │                                                         │
   Lark IM ───► │  LarkEventListener                                      │
                │      └─ [docker exec -i <container>]? lark-cli event    │
                │            consume im.message.receive_v1                │
                │      │   NDJSON over stdout                             │
                │      ▼                                                  │
                │  async for event:                                       │
                │     handle_one_message(event)                           │
                │       1. TranscriptStore.load(chat_id)                  │
                │       2. compact_history(...) + append user msg         │
                │       3. AgentInteraction.run()  ──────────────┐        │
                │       4. TranscriptStore.save(messages)        │        │
                │                                                ▼        │
                │  shared AgentEnv                                        │
                │      ├─ local_attach:  swerex.server in <container>     │
                │      └─ local_native:  pexpect bash on the host         │
                │            execute_bash / lark-cli / str_replace_editor │
                │            / finish                                     │
                └─────────────────────────────────────────────────────────┘
                                          ▲
                                          │ HTTPS
                                          ▼
                                   Lark Open API
```

- **One runtime, one bash session, one model client** for the lifetime of the process.
- Inbound messages are handled **serially** (the bash session is single-threaded — running two agent turns in parallel through it is pointless). If the user sends two messages back-to-back, the second is queued.
- **Single lark identity, single auth.** Both the listener AND the agent's replies route through the *same* `lark-cli`:
  - `local_attach` → the container's `lark-cli` (host side uses `docker exec -i <container>`).
  - `local_native` → the host's `lark-cli` directly.
  Either way, you auth `lark-cli` **once** on the side that owns it — no host/container identity drift.
- Two distinct persistence stores, with different lifecycles and owners:
  - **Per-chat transcripts** — one JSON file per `chat_id` under `~/.uni-agent/app/lark_chat/transcripts/`, written by Python. Holds the **OpenAI-shaped** message log (`tool_calls` on assistants, `tool_call_id` on `role=tool` entries) so re-feeding preserves the assistant↔tool linkage and the model "remembers" the last N turns of THIS chat.
  - **Long-term memory** — files under the configured `memory_dir` (`/workspace/memory/` in the container for `local_attach`, `~/.uni-agent/app/lark_chat/memory/` on the host for `local_native`), written by **the model itself** via `str_replace_editor`. Holds the user's profile, preferences, and digested per-topic notes — survives runtime / process restarts, and is shared across all chats with the same bot.

## Setup

### 1. An OpenAI-compatible chat-completions endpoint

For example vLLM serving a tool-calling model:

```bash
vllm serve /path/to/Qwen3.6-35B-A3B \
  --served-model-name Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 4 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --port 8000
```

### 2. Lark / Feishu Developer Console

The bot must be subscribed to `im.message.receive_v1` (application-identity event) in the [Lark / Feishu Developer Console](https://open.feishu.cn), with event delivery mode set to **WebSocket / long-link** (so `lark-cli event consume` can attach as a client).

### 3. Install `lark-cli` and authorize a bot

```bash
npm install -g @larksuite/cli
lark-cli --version           # sanity check
lark-cli config init --new   # creates a new app on the Lark Open Platform
lark-cli auth login          # OAuth device flow, binds the app to your Feishu account
lark-cli auth status
```

The same authenticated `lark-cli` is reused by the event listener AND by the agent's replies — no host/container identity drift. Edit [`config.local_native.yaml`](./config.local_native.yaml) to point `model.base_url` / `model.name` at your endpoint.

### 4. (Optional, recommended on a personal machine) Deploy in a sandbox

The agent runs LLM-generated shell every turn. Running that directly against your host is unsafe on a personal machine. For isolation, spin up a Docker container first and run the same setup as Step 3 inside it (plus `swerex.server` so the host process can attach over HTTP):

```bash
docker rm -f lark-chat-sandbox 2>/dev/null
docker run -d --name lark-chat-sandbox -p 18000:18000 \
  -v ~/.uni-agent/app/lark_chat/workspace:/workspace \
  nikolaik/python-nodejs:python3.12-nodejs22-bookworm tail -f /dev/null

docker exec -it lark-chat-sandbox bash -lc '
  set -e
  npm install -g @larksuite/cli
  pip install swe-rex
  lark-cli config init --new
  lark-cli auth login
  lark-cli auth status'

docker exec -d lark-chat-sandbox bash -lc '
  python3 -m swerex.server --host 0.0.0.0 --port 18000 --auth-token CHANGEME'
```

The container only sees what you bind-mount (`-v ...:/workspace` above). On the host you only need Python + `docker`; `lark-cli` lives in the container.

Then edit [`config.local_attach.yaml`](./config.local_attach.yaml): `deployment.container` must match the container name above, and `deployment.swerex.auth_token` must match the `--auth-token` passed to `swerex.server`.

### 5. (Optional) Skills

Drop SKILL.md packs under `~/.agents/skills/`, or point `skills_dir` in the YAML at a different directory. The skill manifest is injected into the system prompt on the first turn of each chat, so the model knows it can `cat <path>/SKILL.md` for packs like `lark-im`, `lark-calendar`, `lark-doc`, etc.

### 6. Run

```bash
python -m app.lark_chat.main
```

Picks up `config.local_native.yaml` automatically. For `local_attach`:

```bash
LOCAL_ATTACH_AUTH_TOKEN=CHANGEME \
  python -m app.lark_chat.main --config app/lark_chat/config.local_attach.yaml
```

(`LOCAL_ATTACH_AUTH_TOKEN` is only needed when `deployment.swerex.auth_token` is left `null` in YAML.)

Send a message to the bot in Lark; the trace prints per-turn step / tool / status info. Ctrl+C to shut down (the listener stops cleanly via stdin EOF, the env is closed).

## Configuration

Two reference configs ship with the app — pick one and pass it via `--config <file>`:

- [`config.local_attach.yaml`](./config.local_attach.yaml) — attach to a user-managed Docker container (recommended on a personal machine).
- [`config.local_native.yaml`](./config.local_native.yaml) — host pexpect, no container (picked up when `--config` is omitted).

Top-level keys:

| Key | Default | Purpose |
|---|---|---|
| `deployment.type` | `local_native` (in the default config) | Selects deployment. Either `local_attach` (container — recommended on a personal machine) or `local_native` (host pexpect — only safe inside a sandbox / VM / container). |
| `deployment.container` | — (required for `local_attach`) | Container name the listener `docker exec`s into and `swerex.server` runs in. Owns the lark-cli auth. |
| `deployment.swerex.host` / `.port` | `http://127.0.0.1` / `18000` (`local_attach` only) | swerex.server endpoint |
| `deployment.swerex.auth_token` | `null` → `$LOCAL_ATTACH_AUTH_TOKEN` (`local_attach` only) | swerex.server `--auth-token`. Secret; leave `null` in YAML and set the env var, or inline for local dev. |
| `deployment.post_setup_cmd` | `cd /workspace` (`local_attach`) | Command run once after the bash session starts. |
| `deployment.startup_timeout` | `30.0` / `60.0` | Initial handshake timeout. |
| `deployment.tool_install_dir` | `~/.uni-agent/bin` (`local_native` only) | Where local-only tool scripts are written. Must be writable by the agent process. |
| `memory_dir` | `/workspace/memory` (`local_attach`) / `~/.uni-agent/app/lark_chat/memory` (`local_native`) | Path INSIDE the runtime bash session where the model writes its persistent profile / preferences / notes. Mirrored into the system prompt. |
| `model.base_url` / `.name` | `http://localhost:8000/v1` / `Qwen/Qwen3.6-35B-A3B` | OpenAI-compatible endpoint + model name |
| `model.api_key` | `null` → `$API_KEY` → `EMPTY` | Secret; same rule as `deployment.swerex.auth_token`. |
| `model.sampling_params` | sensible defaults | Forwarded to the chat-completion call (`temperature`, `top_p`, `top_k`, `presence_penalty`, `repetition_penalty`, ...) |
| `tools` | `[execute_bash, lark-cli, str_replace_editor, finish]` | Tools registered on `ToolsManager`. Each entry becomes `ToolConfig(name=...)`. |
| `skills_dir` | `~/.agents/skills` | Skill packs directory (`~` is expanded) |
| `transcripts_dir` | `~/.uni-agent/app/lark_chat/transcripts` | Where per-chat message-log JSON files live on the host (one file per `chat_id`). Distinct from `memory_dir` (model-curated user profile / preferences / notes). |
| `agent.action_timeout` | `60` | Seconds per single tool call |
| `agent.max_steps_per_turn` | `20` | Agent steps before forcing turn end (hard cap; `finish` should land it well under this) |
| `agent.history_max_tokens` | `128000` | Compaction trigger. While the persisted history fits under this, it is forwarded to the model **unchanged** so the server's prefix KV cache stays warm across turns. Crossing this triggers exactly one compaction down to `history_target_tokens`. |
| `agent.history_target_tokens` | `32000` | Post-compaction history size. Must be `<= history_max_tokens`; the gap is the cache-hit headroom that amortizes the compaction-turn cache miss across many subsequent appending turns. Trimming respects user-anchored chunks (`role=tool` is never separated from its parent `role=assistant`) and always keeps the most-recent chunk. Token counts use `tiktoken.cl100k_base` when available, else `len(text) // 4`. |

**Env var overrides** are intentionally limited to secrets:

| Env var | Falls back to | When YAML field is `null` |
|---|---|---|
| `LOCAL_ATTACH_AUTH_TOKEN` | `deployment.swerex.auth_token` | required for `local_attach` (raise if both missing); ignored for `local_native` |
| `API_KEY` | `model.api_key` | defaults to `"EMPTY"` |

## What happens per user message

1. **Filter at the source.** `lark-cli event consume` is launched with a `--jq` filter that drops events from the bot itself (`sender_id == <bot_open_id>`) and any non-text/post message types, so they never reach the Python loop.
2. **Load + (lazy) compact transcript.** `TranscriptStore.load(chat_id)` returns the persisted message list; `compact_history` forwards it unchanged while the total fits under `agent.history_max_tokens` (so the model server's prefix KV cache hits across turns), and only compacts down to `agent.history_target_tokens` when the threshold is crossed. Compaction respects user-anchored chunks (`role=tool` is never separated from its parent `role=assistant`) and always keeps at least the last chunk so the message being replied to is never dropped.
3. **Append the new user message.** Includes a structured Lark metadata block (`chat_id`, `message_id`, `sender_open_id`, `chat_type`, `message_type`, `create_time`) above the content so the agent can call `lark-cli im +messages-reply --message-id <om_...>` directly without parsing IDs out of prose.
4. **Run one turn.** `AgentInteraction.run()` loops: model call → parse 0..N tool calls → execute each sequentially → repeat. The system prompt requires the agent to `ls <memory_dir>/` + `cat <memory_dir>/profile.md <memory_dir>/preferences.md` at the start of every turn, then do the work, then write any newly-learned durable facts back into memory BEFORE replying. The turn ends when the model calls `finish` (preferred end-of-turn signal) or returns plain text with no tool call (fallback). `agent.max_steps_per_turn` is a hard safety cap.
5. **Persist transcript.** `result["messages"]` (the OpenAI-shape message log) is saved atomically back to the chat's JSON file. Memory files are *not* touched by Python — the model wrote them in step 4.

## Long-term memory contract (`memory_dir`)

`memory_dir` is the directory the agent reads/writes long-term memory from, *inside the runtime bash session*:

- For **`local_attach`**: default is `/workspace/memory/` (a container path). Bind-mount the container's `/workspace` to a host directory (per the bootstrap above) so memory survives container restarts.
- For **`local_native`**: default is `~/.uni-agent/app/lark_chat/memory/` — already on the host, so survival across process restarts is automatic.

Either way, `memory_dir` is shared across every chat the bot handles, since there is only one runtime.

The system prompt is rendered against `memory_dir` and establishes a fixed file layout the model is required to follow:

| Path (relative to `memory_dir`) | Purpose |
|---|---|
| `profile.md` | WHO the user is — name, role/team, timezone, language, contact, projects |
| `preferences.md` | HOW the user wants you to behave — reply language, formatting style, default identity, recurring constraints |
| `notes/<slug>.md` | Durable per-topic state — pending tasks, decisions, ongoing efforts |

Read/write protocol enforced by the prompt:

- **Read every turn**: `ls <memory_dir>/` + `cat <memory_dir>/profile.md <memory_dir>/preferences.md` in a single batched assistant response, unless the in-context history of THIS conversation already shows the load happened.
- **Write same turn** *before* replying: whenever the user reveals a fact (name, role, timezone, language, project) or a preference, update the relevant file via `str_replace_editor`.
- **Never mirror the transcript** — memory is digested bullets, not raw chat dumps.

`mkdir -p <memory_dir>/notes` is run at startup by `main.py` so the directory always exists; the model is responsible for actually populating it.

## Files

```
app/lark_chat/
├── __init__.py
├── README.md                       ← this file
├── config.local_attach.yaml        ← reference config for local_attach (recommended)
├── config.local_native.yaml        ← reference config for local_native (default --config)
├── main.py                         ← entrypoint: bootstrap + listener loop + LarkChatConfig
├── prompts.py                      ← build_system_prompt(memory_dir) + format_user_message
├── transcript.py                   ← TranscriptStore (JSON message log per chat_id)
└── listener.py                     ← LarkEventListener + fetch_bot_open_id
```
