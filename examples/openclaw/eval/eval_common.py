"""Shared helpers for the OpenClaw GSM8K evaluation harness.

Configuration (environment variables):

  OPENCLAW_GATEWAY_URL    base URL of the agent under test
                          (default: http://localhost:18789)
  OPENCLAW_GATEWAY_TOKEN  bearer token for the agent under test (required)
  OPENCLAW_ENDPOINT_MODE  "gateway" (default) or "openai"
  OPENCLAW_AGENT_MODEL    model name sent to the agent endpoint (default: default)
  OPENCLAW_WORKSPACE      workspace path (default: ~/.openclaw/workspace)

  Sampling:
  OPENCLAW_DRIVER_TEMPERATURE         (default: 0.7)
  OPENCLAW_DRIVER_TOP_P               (default: 0.8)
  OPENCLAW_DRIVER_MAX_TOKENS          (default: 2048)
  OPENCLAW_DRIVER_REPETITION_PENALTY  (default: 1.1)
  OPENCLAW_AGENT_TEMPERATURE          (default: 0.7)
  OPENCLAW_AGENT_TOP_P                (default: 0.8)
  OPENCLAW_AGENT_MAX_TOKENS           (default: 4096)
  OPENCLAW_AGENT_REPETITION_PENALTY   (default: 1.1)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time

import requests

DEFAULT_DRIVER_HISTORY_MAX_CHARS = 20000
DEFAULT_AGENT_REPLY_MAX_CHARS = 12000

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_DRIVER_MAX_TOKENS = 2048
DEFAULT_AGENT_MAX_TOKENS = 4096
DEFAULT_REPETITION_PENALTY = 1.1


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from model output."""
    return re.sub(r"<think>[\s\S]*?</think>\s*", "", text or "").strip()


def get_env_or_exit(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        print(f"Error: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def get_int_env(name: str, default: int) -> int:
    val = os.environ.get(name, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        print(f"Warning: ignoring invalid integer env {name}={val!r}; using {default}")
        return default


def get_float_env(name: str, default: float) -> float:
    val = os.environ.get(name, "").strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        print(f"Warning: ignoring invalid float env {name}={val!r}; using {default}")
        return default


def get_driver_sampling() -> dict:
    """Sampling params for the role-playing driver LLM (env-overridable)."""
    return {
        "temperature": get_float_env("OPENCLAW_DRIVER_TEMPERATURE", DEFAULT_TEMPERATURE),
        "top_p": get_float_env("OPENCLAW_DRIVER_TOP_P", DEFAULT_TOP_P),
        "max_tokens": get_int_env("OPENCLAW_DRIVER_MAX_TOKENS", DEFAULT_DRIVER_MAX_TOKENS),
        "repetition_penalty": get_float_env("OPENCLAW_DRIVER_REPETITION_PENALTY", DEFAULT_REPETITION_PENALTY),
    }


def get_agent_sampling() -> dict:
    """Sampling params for the agent under test (env-overridable)."""
    return {
        "temperature": get_float_env("OPENCLAW_AGENT_TEMPERATURE", DEFAULT_TEMPERATURE),
        "top_p": get_float_env("OPENCLAW_AGENT_TOP_P", DEFAULT_TOP_P),
        "max_tokens": get_int_env("OPENCLAW_AGENT_MAX_TOKENS", DEFAULT_AGENT_MAX_TOKENS),
        "repetition_penalty": get_float_env("OPENCLAW_AGENT_REPETITION_PENALTY", DEFAULT_REPETITION_PENALTY),
    }


def truncate_text_middle(text: str, max_chars: int) -> str:
    """Keep both ends of long text while making prompt size predictable."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    marker = "\n\n[... truncated for driver context ...]\n\n"
    keep = max(0, max_chars - len(marker))
    head = keep // 3
    tail = keep - head
    return text[:head] + marker + text[-tail:]


def build_driver_messages(
    system_prompt: str,
    conversation_history: list[dict],
    max_chars: int,
) -> list[dict]:
    """Build a bounded prompt for the role-playing driver LLM.

    The external driver only needs enough recent context to decide the next
    instruction. Agent replies can be very long, so keep the most recent turns
    and cap the total character count before sending them to vLLM.
    """
    messages = [{"role": "system", "content": system_prompt}]
    budget = max_chars - len(system_prompt)
    if budget <= 0:
        return messages

    selected: list[dict] = []
    used = 0
    for message in reversed(conversation_history):
        content = str(message.get("content", ""))
        cost = len(content)
        if selected and used + cost > budget:
            break
        if cost > budget:
            content = truncate_text_middle(content, budget)
            cost = len(content)
        selected.append({"role": message.get("role", "user"), "content": content})
        used += cost

    messages.extend(reversed(selected))
    return messages


def load_dataset(path: str) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: dataset file not found: {path}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, list):
        print("Error: dataset must be a JSON array.", file=sys.stderr)
        sys.exit(1)
    return data


def get_workspace() -> str:
    return os.environ.get(
        "OPENCLAW_WORKSPACE",
        os.path.expanduser("~/.openclaw/workspace"),
    )


def ensure_homework_dir(workspace_dir: str, source_name: str, target_name: str) -> None:
    """If ``target_name`` doesn't exist in the workspace, copy from ``source_name``."""
    target = os.path.join(workspace_dir, target_name)
    if os.path.isdir(target):
        print(f"Homework dir already exists: {target}")
        return
    source = os.path.join(workspace_dir, source_name)
    if not os.path.isdir(source):
        print(f"Error: source homework dir not found: {source}", file=sys.stderr)
        sys.exit(1)
    shutil.copytree(source, target)
    print(f"Copied {source} -> {target}")


class AgentEndpoint:
    """Configurable connection to the agent under test.

    In ``gateway`` mode each turn sends a single user message and relies on the
    server to maintain session state (keyed by the OpenAI ``user`` field). In
    ``openai`` mode the conversation history is kept client-side and replayed on
    every request, so any stateless OpenAI-compatible service can be evaluated.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        mode: str = "gateway",
        model: str = "default",
        timeout: float = 180.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.mode = mode.strip().lower()
        self.model = model
        self.timeout = timeout
        self._histories: dict[str, list[dict]] = {}

    @classmethod
    def from_env(cls) -> AgentEndpoint:
        token = get_env_or_exit("OPENCLAW_GATEWAY_TOKEN")
        base_url = os.environ.get("OPENCLAW_GATEWAY_URL", "http://localhost:18789")
        mode = os.environ.get("OPENCLAW_ENDPOINT_MODE", "gateway")
        model = os.environ.get("OPENCLAW_AGENT_MODEL", "default")
        return cls(base_url, token, mode=mode, model=model)

    def reset_session(self, session_id: str) -> None:
        self._histories.pop(session_id, None)

    def send(self, session_id: str, user_message: str, max_retries: int = 3) -> str:
        """Send one user turn to the agent and return the assistant reply text."""
        if self.mode == "openai":
            return self._send_openai(session_id, user_message, max_retries)
        return self._send_gateway(session_id, user_message, max_retries)

    def _post(self, payload: dict, max_retries: int) -> dict:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt < max_retries:
                    wait = 2**attempt
                    print(f"  [retry] agent request failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"  [retry] retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    def _send_gateway(self, session_id: str, user_message: str, max_retries: int) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "user": session_id,
            "messages": [{"role": "user", "content": user_message}],
            **get_agent_sampling(),
        }
        out = self._post(payload, max_retries)
        return out["choices"][0]["message"]["content"]

    def _send_openai(self, session_id: str, user_message: str, max_retries: int) -> str:
        history = self._histories.setdefault(session_id, [])
        history.append({"role": "user", "content": user_message})
        payload = {
            "model": self.model,
            "stream": False,
            "messages": list(history),
            **get_agent_sampling(),
        }
        out = self._post(payload, max_retries)
        reply = out["choices"][0]["message"]["content"]
        history.append({"role": "assistant", "content": reply or ""})
        return reply


# ---------------------------------------------------------------------------
# External (driver) LLM — the role-playing student / TA / teacher
# ---------------------------------------------------------------------------


def make_external_client():
    """Build the OpenAI client for the role-playing driver LLM (from env)."""
    from openai import OpenAI

    api_key = get_env_or_exit("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return OpenAI(api_key=api_key, base_url=base_url)


def get_external_model() -> str:
    return os.environ.get("EXTERNAL_MODEL", "gpt-4o").strip()


def generate_driver_message(
    client,
    model: str,
    system_prompt: str,
    conversation_history: list[dict],
    max_retries: int = 3,
) -> str:
    """Have the role-playing driver LLM decide what to say next."""
    max_chars = get_int_env("OPENCLAW_DRIVER_HISTORY_MAX_CHARS", DEFAULT_DRIVER_HISTORY_MAX_CHARS)
    messages = build_driver_messages(system_prompt, conversation_history, max_chars)
    sampling = get_driver_sampling()
    # repetition_penalty is a vLLM extension, not a standard OpenAI param.
    repetition_penalty = sampling.pop("repetition_penalty")
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body={"repetition_penalty": repetition_penalty},
                **sampling,
            )
            return strip_thinking(resp.choices[0].message.content)
        except Exception as e:
            if attempt < max_retries:
                wait = 2**attempt
                print(f"  [retry] generate_driver_message failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                print(f"  [retry] retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def run_role_session(
    *,
    session_id: str,
    endpoint: AgentEndpoint,
    external_client,
    model: str,
    system_prompt: str,
    first_message: str,
    done_sentinel: str,
    role_label: str,
    max_turns: int,
    max_retries: int = 3,
    output_file: str = "",
) -> bool:
    """Drive a single multi-turn role session against the agent under test."""
    endpoint.reset_session(session_id)
    conversation_history: list[dict] = []
    agent_reply_max_chars = get_int_env("OPENCLAW_AGENT_REPLY_MAX_CHARS", DEFAULT_AGENT_REPLY_MAX_CHARS)

    print(f"\n{'#' * 60}")
    print(f"# {role_label} (session: {session_id})")
    print(f"{'#' * 60}")

    for turn in range(max_turns):
        if turn == 0:
            driver_msg = first_message
        else:
            driver_msg = generate_driver_message(
                external_client,
                model,
                system_prompt,
                conversation_history,
                max_retries=max_retries,
            )

        if done_sentinel in driver_msg:
            print(f"\n  Turn {turn + 1}: driver confirmed session {session_id} is done!")
            return True

        print(f"\n  {'=' * 56}")
        print(f"  Turn {turn + 1}/{max_turns}")
        print(f"  {'=' * 56}")
        print(f"  >> driver -> agent:\n  {driver_msg}\n")

        time.sleep(1)
        conversation_history.append({"role": "assistant", "content": driver_msg})

        agent_reply = endpoint.send(session_id, driver_msg, max_retries=max_retries)
        print(f"  << agent -> driver:\n  {agent_reply}\n")

        if output_file and turn == 0:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"[session: {session_id}]\n")
                f.write(f"{agent_reply}\n\n")

        driver_visible_reply = truncate_text_middle(agent_reply, agent_reply_max_chars)
        if driver_visible_reply != agent_reply:
            print(
                "  [context] truncated long agent reply for the external driver "
                f"({len(agent_reply)} -> {len(driver_visible_reply)} chars)"
            )

        conversation_history.append(
            {
                "role": "user",
                "content": f"The AI assistant replied:\n\n{driver_visible_reply}",
            }
        )

    print(f"\n  Reached max turns ({max_turns}) for session {session_id}.")
    return False
