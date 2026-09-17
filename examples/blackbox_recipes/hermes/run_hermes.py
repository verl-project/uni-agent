"""Run the pinned Hermes ``AIAgent`` inside the sidecar image.

The input is a recipe-owned JSON object on stdin.  Human-readable output and a
runner log are written to files; stdout contains only one
marked JSON envelope so the host can parse it deterministically.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
RESULT_MARKER = "HERMES_RECIPE_RESULT "
HERMES_COMMIT = os.getenv("HERMES_COMMIT", "5eb99eb2844b22ebb723711b8e6a0bbb80bb5f04")
MAX_INPUT_BYTES = 16 * 1024 * 1024


def _add_pinned_runtime_to_path() -> None:
    runtime_root = Path(__file__).resolve().parents[1] / "src"
    if runtime_root.is_dir() and str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))


def _safe_error(exc: BaseException, secrets: list[str] | None = None) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for secret in secrets or []:
        if secret:
            message = message.replace(secret, "<redacted>")
    return message[:4000]


def _load_input() -> dict[str, Any]:
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("runner input exceeds size limit")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("runner input must be a JSON object")
    required = {"schema_version", "run_id", "messages", "workdir", "model", "sampling", "limits", "approval_mode"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"runner input missing fields: {missing}")
    unknown = sorted(set(value) - required)
    if unknown:
        raise ValueError(f"runner input contains unknown fields: {unknown}")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported runner schema_version={value['schema_version']!r}")
    if not isinstance(value["run_id"], str) or not value["run_id"]:
        raise ValueError("runner input run_id must be a non-empty string")
    if not isinstance(value["workdir"], str) or not value["workdir"].startswith("/"):
        raise ValueError("runner input workdir must be an absolute path")
    if value["approval_mode"] not in {"off", "default"}:
        raise ValueError("runner input approval_mode must be 'off' or 'default'")
    if not isinstance(value["messages"], list) or not value["messages"]:
        raise ValueError("runner input messages must be a non-empty list")
    saw_user = False
    for index, message in enumerate(value["messages"]):
        if not isinstance(message, dict) or message.get("role") not in {"system", "user"}:
            raise ValueError(f"runner input message {index} must be a system/user object")
        if message["role"] == "system" and saw_user:
            raise ValueError("runner input system messages must precede user messages")
        if message["role"] == "user":
            saw_user = True
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"runner input message {index} content must be non-empty text")
    model = value["model"]
    if not isinstance(model, dict) or set(model) != {"endpoint", "name", "api_key"}:
        raise ValueError("runner input model must contain endpoint, name and api_key only")
    if not all(isinstance(model[key], str) and model[key] for key in model):
        raise ValueError("runner input model fields must be non-empty strings")
    sampling = value["sampling"]
    if not isinstance(sampling, dict) or not set(sampling).issubset({"temperature", "top_p", "top_k"}):
        raise ValueError("runner input sampling has an invalid shape")
    if any(not isinstance(item, int | float) or isinstance(item, bool) for item in sampling.values()):
        raise ValueError("runner input sampling values must be numeric")
    limits = value["limits"]
    if not isinstance(limits, dict) or set(limits) != {
        "max_iterations",
        "max_tokens",
        "run_budget_seconds",
        "terminal_timeout",
    }:
        raise ValueError("runner input limits have an invalid shape")
    if (
        not isinstance(limits["max_iterations"], int)
        or isinstance(limits["max_iterations"], bool)
        or limits["max_iterations"] < 1
    ):
        raise ValueError("runner input max_iterations must be a positive integer")
    for key in ("run_budget_seconds", "terminal_timeout"):
        if not isinstance(limits[key], int | float) or isinstance(limits[key], bool) or limits[key] <= 0:
            raise ValueError(f"runner input {key} must be positive")
    if limits["max_tokens"] is not None and (
        not isinstance(limits["max_tokens"], int) or isinstance(limits["max_tokens"], bool) or limits["max_tokens"] < 1
    ):
        raise ValueError("runner input max_tokens must be a positive integer or null")
    return value


def _write_json(path: str, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _recipe_config(*, approval_mode: str) -> dict[str, Any]:
    """Return only isolated, deterministic Hermes config knobs."""

    return {
        "compression": {
            "enabled": False,
            "micro_compact": False,
            "proactive_prune_tokens": 0,
            "idle_compact_after_seconds": 0,
        },
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "delegation": {"orchestrator_enabled": False, "max_spawn_depth": 1, "fallback_providers": []},
        "security": {"allow_lazy_installs": False},
        "approvals": {
            "mode": "off" if approval_mode == "off" else "smart",
            "unattended_mode": "approve" if approval_mode == "off" else "deny",
            "single_query_mode": "approve" if approval_mode == "off" else "deny",
            "timeout": 1,
        },
        "agent": {"api_max_retries": 1, "verify_on_stop": False},
    }


def map_hermes_result(raw: object) -> dict[str, Any]:
    """Map Hermes' loose finalizer envelope to the recipe's explicit status."""

    result = raw if isinstance(raw, dict) else {}
    final_response = result.get("final_response")
    reason = str(result.get("turn_exit_reason") or "")
    lowered_reason = reason.lower()
    failed = bool(result.get("failed"))
    partial = bool(result.get("partial"))
    interrupted = bool(result.get("interrupted"))
    error = result.get("error")
    if error or failed or partial or interrupted:
        status = "error"
        finished = False
    elif any(token in lowered_reason for token in ("budget", "iteration", "max_tokens", "run_timeout")):
        status = "budget_exhausted"
        finished = False
    elif result.get("completed") is True and isinstance(final_response, str):
        status = "completed"
        finished = True
    elif lowered_reason in {"text_response", "final_response"} and isinstance(final_response, str):
        status = "completed"
        finished = True
    else:
        status = "error"
        finished = False
        if not error:
            error = "Hermes returned no unambiguous completed final response"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": "",
        "status": status,
        "finished": finished,
        "final_response": final_response if isinstance(final_response, str) else None,
        "error": str(error)[:4000] if error else None,
        "hermes_commit": HERMES_COMMIT,
        "stop_reason": reason or None,
    }


def run(input_data: dict[str, Any], *, result_path: str, log_path: str) -> dict[str, Any]:
    model = input_data["model"]
    workdir = input_data["workdir"]
    limits = input_data["limits"]
    run_id = input_data["run_id"]
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "home").mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        __import__("yaml").safe_dump(_recipe_config(approval_mode=input_data["approval_mode"]), sort_keys=False),
        encoding="utf-8",
    )
    os.environ["HOME"] = str(home / "home")
    os.environ["TERMINAL_CWD"] = workdir
    os.environ["TERMINAL_ENV"] = "local"
    os.environ["TERMINAL_TIMEOUT"] = str(limits["terminal_timeout"])
    os.chdir(workdir)

    agent = None
    raw_result: object = None
    secrets = [model["endpoint"], model["api_key"]]
    with (
        Path(log_path).open("w", encoding="utf-8") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            # Import only after HERMES_HOME and the isolated config exist.  The
            # upstream facade imports approval/env bootstrap code at module load.
            _add_pinned_runtime_to_path()
            from run_agent import AIAgent

            request_overrides = dict(input_data.get("sampling") or {})
            agent = AIAgent(
                base_url=model["endpoint"],
                api_key=model["api_key"],
                provider="custom",
                api_mode="chat_completions",
                model=model["name"],
                max_iterations=limits["max_iterations"],
                max_tokens=limits["max_tokens"],
                run_budget_seconds=limits["run_budget_seconds"],
                enabled_toolsets=["terminal", "file"],
                save_trajectories=False,
                quiet_mode=True,
                skip_memory=True,
                skip_background_review=True,
                skip_context_files=True,
                load_soul_identity=False,
                fallback_model=None,
                request_overrides=request_overrides,
                session_id=run_id,
                platform="api_server",
            )
            system_parts = [m["content"] for m in input_data["messages"] if m["role"] == "system"]
            user_parts = [m["content"] for m in input_data["messages"] if m["role"] == "user"]
            system_message = "\n\n".join(system_parts) or None
            user_message = "\n\n".join(user_parts)
            raw_result = agent.run_conversation(
                user_message=user_message,
                system_message=system_message,
                task_id=run_id,
            )
            envelope = map_hermes_result(raw_result)
        except BaseException as exc:
            log.write(_safe_error(exc, secrets) + "\n")
            redacted_traceback = traceback.format_exc()
            for secret in secrets:
                if secret:
                    redacted_traceback = redacted_traceback.replace(secret, "<redacted>")
            log.write(redacted_traceback)
            envelope = {
                "schema_version": SCHEMA_VERSION,
                "run_id": "",
                "status": "error",
                "finished": False,
                "final_response": None,
                "error": _safe_error(exc, secrets),
                "hermes_commit": HERMES_COMMIT,
                "stop_reason": None,
            }
        finally:
            if agent is not None:
                try:
                    agent.close()
                except BaseException as exc:
                    close_error = _safe_error(exc, secrets)
                    envelope.setdefault("cleanup_errors", []).append(close_error)
        envelope["run_id"] = run_id
        _write_json(result_path, envelope)
    return envelope


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pinned Hermes agent for one Uni-Agent episode.")
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--messages-path")
    parser.add_argument("--log-path", required=True)
    args = parser.parse_args(argv)
    try:
        input_data = _load_input()
        envelope = run(
            input_data,
            result_path=args.result_path,
            log_path=args.log_path,
        )
    except BaseException as exc:
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "run_id": "",
            "status": "error",
            "finished": False,
            "final_response": None,
            "error": _safe_error(exc),
            "hermes_commit": HERMES_COMMIT,
            "stop_reason": None,
        }
        try:
            _write_json(args.result_path, envelope)
        except BaseException:
            pass
    print(RESULT_MARKER + json.dumps(envelope, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
