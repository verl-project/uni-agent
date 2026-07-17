"""Standalone gateway debug launcher for Claude Code and OpenAI-compatible clients."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.gateway import _GatewayActor
from uni_agent.gateway.session import SessionHandle, Trajectory
from verl.workers.rollout.replica import TokenOutput

DEFAULT_CLAUDE_PROMPT = "Reply with OK only."
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5"
DEFAULT_ANTHROPIC_API_KEY = "dummy-local-key"


class DebugFakeTokenizer:
    """Small tokenizer that keeps this launcher independent of test helpers."""

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[int] | str:
        parts = []
        for message in messages:
            parts.append(f"{message['role']}:{self._normalize_content(message.get('content', ''))}\n")
        if tools:
            parts.append(f"tools:{json.dumps(tools, sort_keys=True)}\n")
        if add_generation_prompt:
            parts.append("assistant:")
        text = "".join(parts)
        if tokenize:
            return self.encode(text, add_special_tokens=False)
        return text

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return "".join(chr(int(token_id.item() if hasattr(token_id, "item") else token_id)) for token_id in token_ids)

    def _normalize_content(self, content: Any) -> str:
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text", "")))
                else:
                    parts.append(str(part))
            return "".join(parts)
        if content is None:
            return ""
        return str(content)


class DebugFakeBackend:
    """Debug backend that always returns token ids for the text OK."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
    ) -> TokenOutput:
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": list(prompt_ids),
                "sampling_params": dict(sampling_params),
                "image_data": image_data,
                "video_data": video_data,
            }
        )
        return TokenOutput(
            token_ids=[ord("O"), ord("K")],
            log_probs=[0.0, 0.0],
            stop_reason="completed",
        )


def _token_ids_from_template_result(result: Any) -> Any:
    if hasattr(result, "ids"):
        return list(result.ids)
    if isinstance(result, list):
        token_ids: list[int] = []
        saw_encoding = False
        for item in result:
            if hasattr(item, "ids"):
                token_ids.extend(list(item.ids))
                saw_encoding = True
            else:
                token_ids.append(item)
        if saw_encoding:
            return token_ids
    return result


class TemplateResultTokenIdsWrapper:
    """Tokenizer proxy that flattens tokenizers Encoding template results."""

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        result = self._tokenizer.apply_chat_template(*args, **kwargs)
        if kwargs.get("tokenize", True):
            return _token_ids_from_template_result(result)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)


class OpenAICompletionsBackend:
    """Backend adapter for vLLM-style OpenAI completions with token ids enabled."""

    def __init__(self, *, backend_base_url: str, backend_model: str, timeout: float = 60.0):
        self._completions_url = f"{backend_base_url.rstrip('/')}/completions"
        self._backend_model = backend_model
        self._timeout = timeout

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
    ) -> TokenOutput:
        payload: dict[str, Any] = {
            "model": self._backend_model,
            "prompt": list(prompt_ids),
            "max_tokens": sampling_params.get("max_tokens", 16),
            "request_id": request_id,
            "return_token_ids": True,
            "logprobs": 1,
            "add_special_tokens": False,
        }
        for key in ("temperature", "top_p", "top_k", "stop"):
            if key in sampling_params:
                payload[key] = sampling_params[key]

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(self._completions_url, json=payload)
            if getattr(response, "is_error", False):
                status_code = getattr(response, "status_code", "unknown")
                body = getattr(response, "text", "")
                raise RuntimeError(
                    f"OpenAI completions backend request {request_id} failed with HTTP {status_code}: {body}"
                )
            response.raise_for_status()
        body = response.json()
        choices = body.get("choices") if isinstance(body, dict) else None
        if not choices or not isinstance(choices, list):
            raise RuntimeError("OpenAI completions backend returned no choices")
        choice = choices[0]
        token_ids = choice.get("token_ids") if isinstance(choice, dict) else None
        if token_ids is None:
            raise RuntimeError("OpenAI completions backend must support vLLM return_token_ids=true")

        log_probs = None
        logprobs = choice.get("logprobs")
        if isinstance(logprobs, dict):
            token_logprobs = logprobs.get("token_logprobs")
            if isinstance(token_logprobs, list) and len(token_logprobs) == len(token_ids):
                log_probs = list(token_logprobs)
        stop_reason = "length" if choice.get("finish_reason") == "length" else "completed"
        return TokenOutput(token_ids=list(token_ids), log_probs=log_probs, stop_reason=stop_reason)


@dataclass(frozen=True)
class DebugSessionResult:
    session_id: str
    output_path: Path
    metadata_path: Path
    debug_snapshot_path: Path
    trajectories_count: int
    provider_urls: dict[str, str]
    claude_returncode: int | None = None


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def trajectory_to_record(
    *,
    session_id: str,
    trajectory_index: int,
    trajectory: Trajectory,
    metadata: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "session_id": session_id,
        "trajectory_index": trajectory_index,
        "created_at": created_at or utc_now(),
        "metadata": dict(metadata or {}),
        "trajectory": {
            "prompt_ids": list(trajectory.prompt_ids),
            "response_ids": list(trajectory.response_ids),
            "response_mask": list(trajectory.response_mask),
            "response_logprobs": (
                list(trajectory.response_logprobs) if trajectory.response_logprobs is not None else None
            ),
            "reward_info": dict(trajectory.reward_info),
            "reward_score": trajectory.reward_score,
            "num_turns": trajectory.num_turns,
            "multi_modal_data": trajectory.multi_modal_data,
            "extra_fields": dict(trajectory.extra_fields),
        },
    }


def write_trajectories_jsonl(
    *,
    output_dir: Path,
    session_id: str,
    trajectories: list[Trajectory],
    metadata: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> Path:
    session_dir = output_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "trajectories.jsonl"
    timestamp = created_at or utc_now()
    with path.open("w", encoding="utf-8") as f:
        for index, trajectory in enumerate(trajectories):
            record = trajectory_to_record(
                session_id=session_id,
                trajectory_index=index,
                trajectory=trajectory,
                metadata=metadata,
                created_at=timestamp,
            )
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")
    return path


def write_session_metadata_json(
    *,
    output_dir: Path,
    session_id: str,
    metadata: dict[str, Any],
    provider_urls: dict[str, str],
    trajectories_count: int,
    trajectories_path: Path,
    debug_snapshot_path: Path,
    created_at: str | None = None,
) -> Path:
    session_dir = output_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "session_metadata.json"
    record = {
        "schema_version": 1,
        "session_id": session_id,
        "created_at": created_at or utc_now(),
        "metadata": dict(metadata),
        "provider_urls": dict(provider_urls),
        "trajectories_count": trajectories_count,
        "trajectories_path": str(trajectories_path),
        "debug_snapshot_path": str(debug_snapshot_path),
    }
    path.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_debug_snapshot_json(
    *,
    output_dir: Path,
    session_id: str,
    snapshot: dict[str, Any],
    created_at: str | None = None,
) -> Path:
    session_dir = output_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "debug_snapshot.json"
    record = {
        **snapshot,
        "schema_version": 1,
        "session_id": session_id,
        "created_at": created_at or utc_now(),
    }
    path.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def provider_urls(handle: SessionHandle) -> dict[str, str]:
    if not handle.base_url:
        raise RuntimeError("Session handle does not include a base_url")
    base_url = handle.base_url.rstrip("/")
    anthropic_base_url = base_url.removesuffix("/v1")
    return {
        "anthropic_base_url": anthropic_base_url,
        "openai_base_url": base_url,
        "reward_info_url": handle.reward_info_url,
    }


def build_claude_env(
    base_env: dict[str, str],
    *,
    anthropic_base_url: str,
    anthropic_api_key: str,
) -> dict[str, str]:
    env = {}
    for key, value in base_env.items():
        lower_key = key.lower()
        if key in {"ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_API_BASE_URL", "CLAUDE_CODE_API_KEY"}:
            continue
        if "proxy" in lower_key:
            continue
        env[key] = value

    host = urlparse(anthropic_base_url).hostname or "localhost"
    env.update(
        {
            "ANTHROPIC_BASE_URL": anthropic_base_url,
            "ANTHROPIC_API_KEY": anthropic_api_key,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "NO_PROXY": f"{host},127.0.0.1,localhost",
        }
    )
    return env


def build_claude_command(*, prompt: str, model: str, debug_file: Path) -> list[str]:
    return [
        "claude",
        "--bare",
        "--no-session-persistence",
        "--debug-file",
        str(debug_file),
        "-p",
        prompt,
        "--output-format",
        "text",
        "--model",
        model,
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("fake", "openai-completions"), default="fake")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-claude", action="store_true")
    parser.add_argument("--claude-prompt", default=DEFAULT_CLAUDE_PROMPT)
    parser.add_argument("--claude-model", default=DEFAULT_CLAUDE_MODEL)
    parser.add_argument("--claude-timeout", default=300.0, type=float)
    parser.add_argument("--anthropic-api-key", default=DEFAULT_ANTHROPIC_API_KEY)
    parser.add_argument("--backend-base-url")
    parser.add_argument("--backend-model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--prompt-length", type=int)
    parser.add_argument("--response-length", type=int)
    args = parser.parse_args(argv)

    if args.backend == "openai-completions":
        missing = [
            flag
            for flag, value in (
                ("--backend-base-url", args.backend_base_url),
                ("--backend-model", args.backend_model),
                ("--tokenizer", args.tokenizer),
            )
            if not value
        ]
        if missing:
            parser.error(f"--backend openai-completions requires {', '.join(missing)}")
    return args


def build_tokenizer(args: argparse.Namespace):
    if args.backend == "fake":
        return DebugFakeTokenizer()

    from transformers import AutoTokenizer

    return TemplateResultTokenIdsWrapper(AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True))


def build_backend(args: argparse.Namespace):
    if args.backend == "fake":
        return DebugFakeBackend()
    return OpenAICompletionsBackend(
        backend_base_url=args.backend_base_url,
        backend_model=args.backend_model,
    )


async def wait_for_manual_finalize(stdin: Any = sys.stdin) -> str:
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    trigger = "signal"

    def _stop(source: str) -> None:
        nonlocal trigger
        if not stop_event.is_set():
            trigger = source
            stop_event.set()

    registered_signals = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop, "signal")
            registered_signals.append(sig)
        except NotImplementedError:
            pass
    stdin_task: asyncio.Task | None = None
    stdin_fd = None
    try:
        if stdin is not None and stdin.isatty():
            print("Press Enter or Ctrl-C to finalize the session and write trajectories.")
            try:
                stdin_fd = stdin.fileno()
                loop.add_reader(stdin_fd, _stop, "stdin")
            except (AttributeError, OSError, NotImplementedError):
                stdin_task = asyncio.create_task(asyncio.to_thread(stdin.readline))
                stdin_task.add_done_callback(lambda _: _stop("stdin"))
        await stop_event.wait()
        return trigger
    finally:
        if stdin_fd is not None:
            try:
                loop.remove_reader(stdin_fd)
            except (OSError, NotImplementedError):
                pass
        if stdin_task is not None:
            stdin_task.cancel()
        for sig in registered_signals:
            loop.remove_signal_handler(sig)


def capture_debug_snapshot(actor: _GatewayActor, session_id: str, urls: dict[str, str]) -> dict[str, Any]:
    session = actor._sessions.get(session_id)
    if session is None:
        return {
            "session_state": None,
            "message_history": [],
            "active_tool_schemas": None,
            "active_chains": [],
            "provider_urls": dict(urls),
            "note": "session was not available when debug snapshot was captured",
        }
    active_chains = [
        {
            "chain_id": chain.chain_id,
            "message_history": list(chain.message_history),
            "active_tool_schemas": chain.active_tool_schemas,
        }
        for chain in session.active_chains
    ]
    snapshot = {
        "session_state": session.snapshot_state(),
        # Keep the original single-chain fields for existing debug consumers.
        "message_history": active_chains[0]["message_history"] if len(active_chains) == 1 else [],
        "active_tool_schemas": active_chains[0]["active_tool_schemas"] if len(active_chains) == 1 else None,
        "active_chains": active_chains,
        "provider_urls": dict(urls),
    }
    if len(active_chains) > 1:
        snapshot["note"] = "message_history and active_tool_schemas are omitted when multiple chains are active"
    return snapshot


async def run_claude_once(
    *,
    urls: dict[str, str],
    output_dir: Path,
    session_id: str,
    claude_prompt: str,
    claude_model: str,
    anthropic_api_key: str,
    claude_timeout: float,
) -> dict[str, Any]:
    session_dir = output_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    debug_file = session_dir / "claude-debug.log"
    command = build_claude_command(prompt=claude_prompt, model=claude_model, debug_file=debug_file)
    env = build_claude_env(
        dict(os.environ),
        anthropic_base_url=urls["anthropic_base_url"],
        anthropic_api_key=anthropic_api_key,
    )
    stdout_path = session_dir / "claude.stdout"
    stderr_path = session_dir / "claude.stderr"
    timed_out = False
    try:
        completed = await asyncio.to_thread(
            subprocess.run,
            command,
            env=env,
            check=False,
            text=True,
            capture_output=True,
            timeout=claude_timeout,
        )
        returncode = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = exc.stdout if exc.stdout is not None else exc.output
        stderr = exc.stderr
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        elif stdout is None:
            stdout = ""
        else:
            stdout = str(stdout)
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        elif stderr is None:
            stderr = ""
        else:
            stderr = str(stderr)
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return {
        "enabled": True,
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout_seconds": claude_timeout,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "debug_log_path": str(debug_file),
    }


def print_provider_urls(urls: dict[str, str], *, anthropic_api_key: str) -> None:
    api_key_display = anthropic_api_key if anthropic_api_key == DEFAULT_ANTHROPIC_API_KEY else "<redacted>"
    print("Claude Code:")
    print(f"  export ANTHROPIC_BASE_URL={urls['anthropic_base_url']}")
    print(f"  export ANTHROPIC_API_KEY={api_key_display}")
    print("OpenAI-compatible:")
    print(f"  base_url={urls['openai_base_url']}")
    print(f"Reward info: {urls['reward_info_url']}")


async def run_debug_session_once(
    *,
    session_id: str,
    output_dir: Path,
    backend: str = "fake",
    run_claude: bool = False,
    claude_prompt: str = DEFAULT_CLAUDE_PROMPT,
    claude_model: str = DEFAULT_CLAUDE_MODEL,
    anthropic_api_key: str = DEFAULT_ANTHROPIC_API_KEY,
    claude_timeout: float = 300.0,
    backend_base_url: str | None = None,
    backend_model: str | None = None,
    tokenizer: str | None = None,
    prompt_length: int | None = None,
    response_length: int | None = None,
) -> DebugSessionResult:
    args = argparse.Namespace(
        backend=backend,
        backend_base_url=backend_base_url,
        backend_model=backend_model,
        tokenizer=tokenizer,
    )
    gateway_backend = build_backend(args)
    gateway_tokenizer = build_tokenizer(args)
    actor = _GatewayActor(
        GatewayActorConfig(
            tokenizer=gateway_tokenizer,
            prompt_length=prompt_length,
            response_length=response_length,
        ),
        gateway_backend,
    )

    await actor.start()
    trajectories: list[Trajectory] = []
    urls: dict[str, str] = {}
    debug_snapshot: dict[str, Any] = {
        "session_state": None,
        "message_history": [],
        "active_tool_schemas": None,
        "active_chains": [],
        "provider_urls": {},
        "note": "session was not created before debug snapshot was captured",
    }
    claude_metadata: dict[str, Any] = {
        "enabled": run_claude,
        "returncode": None,
        "timed_out": False,
        "timeout_seconds": claude_timeout,
        "stdout_path": None,
        "stderr_path": None,
        "debug_log_path": None,
    }
    try:
        handle = await actor.create_session(session_id)
        urls = provider_urls(handle)
        print_provider_urls(urls, anthropic_api_key=anthropic_api_key)
        if run_claude:
            claude_metadata = await run_claude_once(
                urls=urls,
                output_dir=output_dir,
                session_id=session_id,
                claude_prompt=claude_prompt,
                claude_model=claude_model,
                anthropic_api_key=anthropic_api_key,
                claude_timeout=claude_timeout,
            )
        else:
            await wait_for_manual_finalize()
        debug_snapshot = capture_debug_snapshot(actor, session_id, urls)
        trajectories = await actor.finalize_session(session_id)
    finally:
        await actor.shutdown()

    metadata = {"backend": backend, "claude": claude_metadata}
    output_path = write_trajectories_jsonl(
        output_dir=output_dir,
        session_id=session_id,
        trajectories=trajectories,
        metadata=metadata,
    )
    debug_snapshot_path = write_debug_snapshot_json(
        output_dir=output_dir,
        session_id=session_id,
        snapshot=debug_snapshot,
    )
    metadata_path = write_session_metadata_json(
        output_dir=output_dir,
        session_id=session_id,
        metadata=metadata,
        provider_urls=urls,
        trajectories_count=len(trajectories),
        trajectories_path=output_path,
        debug_snapshot_path=debug_snapshot_path,
    )
    return DebugSessionResult(
        session_id=session_id,
        output_path=output_path,
        metadata_path=metadata_path,
        debug_snapshot_path=debug_snapshot_path,
        trajectories_count=len(trajectories),
        provider_urls=urls,
        claude_returncode=claude_metadata["returncode"],
    )


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = await run_debug_session_once(
        session_id=args.session_id,
        output_dir=args.output_dir,
        backend=args.backend,
        run_claude=args.run_claude,
        claude_prompt=args.claude_prompt,
        claude_model=args.claude_model,
        claude_timeout=args.claude_timeout,
        anthropic_api_key=args.anthropic_api_key,
        backend_base_url=args.backend_base_url,
        backend_model=args.backend_model,
        tokenizer=args.tokenizer,
        prompt_length=args.prompt_length,
        response_length=args.response_length,
    )
    print(f"Wrote {result.trajectories_count} trajectories to {result.output_path}")
    print(f"Wrote debug snapshot to {result.debug_snapshot_path}")
    print(f"Wrote session metadata to {result.metadata_path}")
    if result.claude_returncode not in (None, 0):
        return result.claude_returncode
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
