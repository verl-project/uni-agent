# Build Your Personal Agent with Lark

*2026-05-24 · [Yuyang Ding](https://yyding1.github.io/)*

*First post in an ongoing series on **Project Milo**, the personal chat agent. See the [Vision](./vision.md) for the bigger picture.*

<img src="../lark-summarize-group-chat.jpeg" alt="Lark chat" style="width: 100%; max-width: 900px; height: auto; display: block; margin: 0 auto;" />

<!-- *Real example: the user asks for a digest of unresolved issues in a Lark group chat. The agent pulls recent messages through `lark-cli`, identifies the open threads, cross-references the people involved, embeds linked GitHub issues, and writes back as native Lark Markdown.* -->

---

## What It Can Do

### Use Lark in plain language

Through `lark-cli`, the agent can operate the Lark surface area your account can access, including:

- **Calendar**: "book 30 min with Miko next Tuesday afternoon"
- **Docs and Wiki**: "summarize this doc and send it to me"
- **Instant Messaging**: "what did Miko say about the launch last week?"
- **Mail**: "draft a polite decline to this Friday invite"
- **Meetings and Minutes**: "what did I commit to in yesterday's standup?"
- **Bitable, Sheets, Tasks, Contacts, Drive, Approval, Whiteboard, OKR, and more**: plug-and-play skill packs

<!-- IMAGE: lark-actions.png: chat screenshot showing the agent find today's meetings and summarize one, with a terminal trace of lark-cli + docs fetch + reply -->

### Remember the person, not just the chat

Two layers of persistence:

- **Per-chat transcript**: a JSON message log per `chat_id`, preserving OpenAI-shaped assistant/tool linkage so multi-turn conversations continue cleanly.
- **Long-term memory**: model-written files under `/workspace/memory/`, such as `profile.md`, `preferences.md`, and topic notes.

The transcript captures what happened recently in this chat. Memory captures what should remain true tomorrow: your name, timezone, team, projects, language preference, recurring constraints, and the people you often mention.

If history is trimmed, the process restarts, or the container is recreated, memory still lives on the host bind mount. The next Lark message picks up from there.

### Bring your own model

The app talks to any OpenAI-compatible chat-completions endpoint. Self-hosted serving stacks like vLLM or SGLang, public APIs, or an internal model gateway all work the same way. The model is just config:

```yaml
model:
  base_url: http://localhost:8000/v1
  name: Qwen/Qwen3.6-35B-A3B
  api_key: EMPTY
```

Big GPU box? Run the bigger model. Laptop demo? Point it at a smaller endpoint. The Lark integration stays the same.

---

## Quickstart

### Step 0: Prerequisites and dependencies

- macOS or Linux with Docker (Recommended)
- An OpenAI-compatible chat-completions endpoint
- A Lark/Feishu app in the [Lark Open Platform](https://open.feishu.cn)

Host-side dependencies are intentionally minimal. Clone the repo, create a virtualenv, install:

```bash
git clone https://github.com/yyDing1/uni-agent.git
cd uni-agent

python3 -m venv .venv
source .venv/bin/activate

pip install swe-rex pydantic loguru orjson aiohttp openai
```


### Step 1: Start the sandbox and authenticate `lark-cli`

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

The container owns the shell runtime, `lark-cli` authentication, skills, and `swerex.server`. The host process talks to it through the local attach endpoint. `CHANGEME` can be any token as long as it matches the config below.

### Step 2: Configure the bot

Edit `app/lark_chat/config.yaml` (the default config the app reads on startup). A working example:

```yaml
container: lark-chat-sandbox

swerex:
  host: http://127.0.0.1
  port: 18000
  auth_token: CHANGEME

model:
  base_url: http://localhost:8000/v1
  name: Qwen/Qwen3.6-35B-A3B
  api_key: EMPTY
  sampling_params:
    temperature: 1.0
    top_p: 0.95
    presence_penalty: 1.5
    top_k: 20
    repetition_penalty: 1.0

tools:
  - execute_bash
  - lark-cli
  - str_replace_editor
  - finish

skills_dir: ~/.uni-agent/skills
transcripts_dir: ~/.uni-agent/app/lark_chat/transcripts

agent:
  action_timeout: 60
  max_steps_per_turn: 20
  max_history_turns: 30
```

`container` and `swerex.auth_token` must match what Step 1 used. Everything else has sensible defaults; tweak `model.base_url` / `model.name` to point at your endpoint.

### Step 3: Run the bot

```bash
python -m app.lark_chat.main
```

(Pass `--config <path>` if you keep your config somewhere other than `app/lark_chat/config.yaml`.)

Startup resolves the bot `open_id`, starts the sandbox env, installs tools and skills, wires the model client, starts the Lark event listener, and enters the chat loop. When you see this line, it is live:

```text
Entering chat loop. Send a Lark message to the bot.
```
