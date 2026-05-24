# `lark_chat` — long-running Lark chat agent

A long-running process that listens for IM messages on Lark / Feishu,
dispatches each user message to a multi-step agent loop running on a
shared sandbox env, and replies back through `lark-cli`. Each chat is a
real **ongoing conversation**: the OpenAI-shaped message log is
persisted per `chat_id` and trimmed on a sliding window so you can
talk to the same bot across many turns and across process restarts.

The agent also keeps a **persistent user profile / preferences memory**
under the container's `/workspace/memory/`, written by the model
itself, so it accumulates a real picture of the user across
conversations (not just within a single chat thread).

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
                │      └─ docker exec -i <container> lark-cli event       │
                │            consume im.message.receive_v1                │
                │      │   NDJSON over stdout                             │
                │      ▼                                                  │
                │  async for event:                                       │
                │     handle_one_message(event)                           │
                │       1. TranscriptStore.load(chat_id)                  │
                │       2. trim_history(...) + append user msg            │
                │       3. AgentInteraction.run()  ──────────────┐        │
                │       4. TranscriptStore.save(messages)        │        │
                │                                                ▼        │
                │  shared AgentEnv (local_attach)                         │
                │      └─ swerex.server in <container>                    │
                │            execute_bash / lark-cli / str_replace_editor │
                │            / finish                                     │
                └─────────────────────────────────────────────────────────┘
                                          ▲
                                          │ HTTPS
                                          ▼
                                   Lark Open API
```

- **One container, one bash session, one model client** for the lifetime of the process.
- Inbound messages are handled **serially** (the bash session is single-threaded — running two agent turns in parallel through it is pointless). If the user sends two messages back-to-back, the second is queued.
- **Single lark identity, single auth.** Both the listener AND the agent's replies route through the container's `lark-cli` (via `docker exec -i <container>`). You auth `lark-cli` **once**, inside the container — no host/container identity drift.
- Two distinct persistence stores, with different lifecycles and owners:
  - **Per-chat transcripts** — one JSON file per `chat_id` under `~/.uni-agent/app/lark_chat/transcripts/`, written by Python. Holds the **OpenAI-shaped** message log (`tool_calls` on assistants, `tool_call_id` on `role=tool` entries) so re-feeding preserves the assistant↔tool linkage and the model "remembers" the last N turns of THIS chat.
  - **Long-term memory** — files under `/workspace/memory/` inside the container (bind-mounted to a host dir), written by **the model itself** via `str_replace_editor`. Holds the user's profile, preferences, and digested per-topic notes — survives chat-history trimming AND process restarts, and is shared across all chats with the same bot.

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

### 3. Sandbox container (one-time bootstrap)

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
  lark-cli auth login        # complete OAuth inside the container
  lark-cli auth status'

docker exec -d lark-chat-sandbox bash -lc '
  python3 -m swerex.server --host 0.0.0.0 --port 18000 --auth-token CHANGEME'
```

On the host you only need Python + `docker`. `lark-cli` lives in the container.

### 4. (Optional) Skills

Drop SKILL.md packs under `~/.uni-agent/skills/` (override with `LARK_SKILLS_DIR=...`). The skill manifest is injected into the system prompt on the first turn of each chat, so the model knows it can `cat <path>/SKILL.md` for things like `lark-im`, `lark-calendar`, `lark-doc`, etc.

### 5. Run

```bash
LOCAL_ATTACH_AUTH_TOKEN=CHANGEME python -m app.lark_chat.main
```

Send a message to the bot in Lark; the trace prints per-turn step / tool / status info. Ctrl+C to shut down (the listener stops cleanly via stdin EOF, the env is closed).

To use a custom config:

```bash
LOCAL_ATTACH_AUTH_TOKEN=CHANGEME python -m app.lark_chat.main --config /path/to/my.yaml
```

## Configuration

All non-secret settings live in [`config.yaml`](./config.yaml). Override the path with `--config <file>`. Top-level keys:

| Key | Default | Purpose |
|---|---|---|
| `container` | `lark-chat-sandbox` | Container name the listener `docker exec`s into and `swerex.server` runs in. Owns the lark-cli auth. |
| `swerex.host` / `.port` | `http://127.0.0.1` / `18000` | swerex.server endpoint |
| `swerex.auth_token` | `null` → `$LOCAL_ATTACH_AUTH_TOKEN` | swerex.server `--auth-token`. Secret; leave `null` in YAML and set the env var, or inline for local dev. |
| `model.base_url` / `.name` | `http://localhost:8000/v1` / `Qwen/Qwen3.6-35B-A3B` | OpenAI-compatible endpoint + model name |
| `model.api_key` | `null` → `$API_KEY` → `EMPTY` | Secret; same rule as `swerex.auth_token`. |
| `model.sampling_params` | sensible defaults | Forwarded to the chat-completion call (`temperature`, `top_p`, `top_k`, `presence_penalty`, `repetition_penalty`, ...) |
| `tools` | `[execute_bash, lark-cli, str_replace_editor, finish]` | Tools registered on `ToolsManager`. Each entry becomes `ToolConfig(name=...)`. |
| `skills_dir` | `~/.uni-agent/skills` | Skill packs directory (`~` is expanded) |
| `transcripts_dir` | `~/.uni-agent/app/lark_chat/transcripts` | Where per-chat message-log JSON files live on the host (one file per `chat_id`). Distinct from the container's `/workspace/memory/` (model-curated user profile / preferences / notes). |
| `agent.action_timeout` | `60` | Seconds per single tool call |
| `agent.max_steps_per_turn` | `20` | Agent steps before forcing turn end (hard cap; `finish` should land it well under this) |
| `agent.max_history_turns` | `30` | Trim history to last N user-anchored turns |

**Env var overrides** are intentionally limited to secrets:

| Env var | Falls back to | When YAML field is `null` |
|---|---|---|
| `LOCAL_ATTACH_AUTH_TOKEN` | `swerex.auth_token` | required (raise if both missing) |
| `API_KEY` | `model.api_key` | defaults to `"EMPTY"` |

## What happens per user message

1. **Filter at the source.** `lark-cli event consume` is launched with a `--jq` filter that drops events from the bot itself (`sender_id == <bot_open_id>`) and any non-text/post message types, so they never reach the Python loop.
2. **Load + trim transcript.** `TranscriptStore.load(chat_id)` returns the persisted message list; `trim_history` keeps the system message + the last `agent.max_history_turns` user-anchored chunks intact (never strips a `role=tool` away from its parent `role=assistant`).
3. **Append the new user message.** Includes a structured Lark metadata block (`chat_id`, `message_id`, `sender_open_id`, `chat_type`, `message_type`, `create_time`) above the content so the agent can call `lark-cli im +messages-reply --message-id <om_...>` directly without parsing IDs out of prose.
4. **Run one turn.** `AgentInteraction.run()` loops: model call → parse 0..N tool calls → execute each sequentially → repeat. The system prompt requires the agent to `ls /workspace/memory/` + `cat profile.md preferences.md` at the start of every turn, then do the work, then write any newly-learned durable facts back into memory BEFORE replying. The turn ends when the model calls `finish` (preferred end-of-turn signal) or returns plain text with no tool call (fallback). `agent.max_steps_per_turn` is a hard safety cap.
5. **Persist transcript.** `result["messages"]` (the OpenAI-shape message log) is saved atomically back to the chat's JSON file. Memory files are *not* touched by Python — the model wrote them in step 4.

## Long-term memory contract (`/workspace/memory/`)

The container's `/workspace` is bind-mounted to `~/.uni-agent/app/lark_chat/workspace` on the host (per the bootstrap above). `/workspace/memory/` therefore survives container / process restarts, and is shared across every chat the bot handles.

The system prompt establishes a fixed file layout the model is required to follow:

| Path | Purpose |
|---|---|
| `/workspace/memory/profile.md` | WHO the user is — name, role/team, timezone, language, contact, projects |
| `/workspace/memory/preferences.md` | HOW the user wants you to behave — reply language, formatting style, default identity, recurring constraints |
| `/workspace/memory/notes/<slug>.md` | Durable per-topic state — pending tasks, decisions, ongoing efforts |

Read/write protocol enforced by the prompt:

- **Read every turn**: `ls /workspace/memory/` + `cat profile.md preferences.md` in a single batched assistant response, unless the in-context history of THIS conversation already shows the load happened.
- **Write same turn** *before* replying: whenever the user reveals a fact (name, role, timezone, language, project) or a preference, update the relevant file via `str_replace_editor`.
- **Never mirror the transcript** — memory is digested bullets, not raw chat dumps.

`mkdir -p /workspace/memory/notes` is run at startup by `main.py` so the directory always exists; the model is responsible for actually populating it.

## Files

```
app/lark_chat/
├── __init__.py
├── README.md              ← this file
├── config.yaml            ← default config (override with --config <path>)
├── main.py                ← entrypoint: bootstrap + listener loop + LarkChatConfig
├── prompts.py             ← SYSTEM_PROMPT + format_user_message
├── transcript.py          ← TranscriptStore (JSON message log per chat_id)
└── listener.py            ← LarkEventListener + fetch_bot_open_id
```
