# `lark_chat` — long-running Lark chat agent

A long-running process that listens for IM messages on Lark / Feishu,
dispatches each user message to a multi-step agent loop running on a
shared sandbox env, and replies back through `lark-cli`. Each chat is a
real **ongoing conversation**: history is persisted per `chat_id` and
trimmed on a sliding window so you can talk to the same bot across many
turns and across process restarts.

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
                │       1. ConversationStore.load(chat_id)                │
                │       2. trim_history(...) + append user msg            │
                │       3. AgentInteraction.run()  ──────────────┐        │
                │       4. ConversationStore.save(messages)      │        │
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
- Per-chat history is one JSON file per `chat_id` under `~/.uni-agent/app/lark_chat/conversations/`. We persist the **OpenAI-shaped** history (`tool_calls` on assistant messages, `tool_call_id` on tool responses) so re-feeding it preserves the assistant↔tool linkage.
- The agent's **long-term notes** live at `/workspace/history/` inside the container, which is bind-mounted to a host directory — so notes survive container / process restarts. The system prompt nudges the model to read this on demand and write *digested* context there (decisions, preferences, pending tasks), not raw transcripts.

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
LOCAL_ATTACH_CONTAINER=lark-chat-sandbox \
LOCAL_ATTACH_PORT=18000 \
LOCAL_ATTACH_AUTH_TOKEN=CHANGEME \
BASE_URL=http://localhost:8000/v1 \
MODEL_NAME=Qwen/Qwen3.6-35B-A3B \
python -m app.lark_chat.main
```

Send a message to the bot in Lark; the trace prints per-turn step / tool / status info. Ctrl+C to shut down (the listener stops cleanly via stdin EOF, the env is closed).

## Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `LOCAL_ATTACH_CONTAINER` | `lark-chat-sandbox` | Container name the listener `docker exec`s into and `swerex.server` runs in |
| `LOCAL_ATTACH_HOST` | `http://127.0.0.1` | swerex.server host |
| `LOCAL_ATTACH_PORT` | `18000` | swerex.server port |
| `LOCAL_ATTACH_AUTH_TOKEN` | *(required)* | swerex.server `--auth-token` |
| `BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `API_KEY` | `EMPTY` | API key for the endpoint |
| `MODEL_NAME` | `Qwen/Qwen3.6-35B-A3B` | Model name sent to endpoint |
| `MODEL_TEMPERATURE` / `MODEL_TOP_P` / `MODEL_TOP_K` / `MODEL_PRESENCE_PENALTY` / `MODEL_REPETITION_PENALTY` | sensible defaults | Sampling overrides |
| `LARK_SKILLS_DIR` | `~/.uni-agent/skills` | Skill packs directory |
| `LARK_CHAT_HISTORY_DIR` | `~/.uni-agent/app/lark_chat` | Persisted conversations root |
| `LARK_CHAT_ACTION_TIMEOUT` | `60` | Seconds per single tool call |
| `LARK_CHAT_MAX_STEPS_PER_TURN` | `20` | Agent steps before forcing turn end (hard cap; `finish` should land it well under this) |
| `LARK_CHAT_MAX_HISTORY_TURNS` | `30` | Trim history to last N user-anchored turns |

## What happens per user message

1. **Filter at the source.** `lark-cli event consume` is launched with a `--jq` filter that drops events from the bot itself (`sender_id == <bot_open_id>`) and any non-text/post message types, so they never reach the Python loop.
2. **Load + trim history.** `ConversationStore.load(chat_id)` returns the persisted message list; `trim_history` keeps the system message + the last `LARK_CHAT_MAX_HISTORY_TURNS` user-anchored chunks intact (never strips a `role=tool` away from its parent `role=assistant`).
3. **Append the new user message.** Includes a structured Lark metadata block (`chat_id`, `message_id`, `sender_open_id`, `chat_type`, `message_type`, `create_time`) above the content so the agent can call `lark-cli im +messages-reply --message-id <om_...>` directly without parsing IDs out of prose.
4. **Run one turn.** `AgentInteraction.run()` loops: model call → parse 0..N tool calls → execute each sequentially → repeat. The turn ends when the model calls `finish` (preferred end-of-turn signal) or returns plain text with no tool call (fallback). `max_steps_per_turn` is a hard safety cap.
5. **Persist.** `result["messages"]` (the OpenAI-shape conversation) is saved atomically back to the chat's JSON file.

## Long-term memory contract

The container's `/workspace` is bind-mounted to `~/.uni-agent/app/lark_chat/workspace` on the host (per the bootstrap above). The agent's notes at `/workspace/history/` therefore survive container restarts.

The system prompt instructs the model to:

- `ls /workspace/history/` at the start of a turn IF the request hints at long-term context (skip for greetings / one-shot questions).
- Write **digested** context (decisions, preferences, pending work) — not chat transcripts — using `str_replace_editor` or `execute_bash`.
- Re-read relevant files before deciding how to act.

## Files

```
app/lark_chat/
├── __init__.py
├── README.md              ← this file
├── main.py                ← entrypoint: bootstrap + listener loop
├── prompts.py             ← SYSTEM_PROMPT + format_user_message
├── conversation.py        ← ConversationStore (JSON per chat_id)
└── listener.py            ← LarkEventListener + fetch_bot_open_id
```
