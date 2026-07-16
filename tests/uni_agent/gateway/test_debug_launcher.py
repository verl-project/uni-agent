import asyncio
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from examples.gateway import debug_launcher
from uni_agent.gateway.session import SessionHandle, Trajectory


def test_trajectory_to_record_is_json_serializable_and_preserves_fields():
    """Trajectory output records keep every training-visible field and can be
    written as plain JSON."""
    trajectory = Trajectory(
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        response_mask=[1, 0],
        response_logprobs=[-0.1, 0.0],
        reward_info={"ok": True},
        reward_score=1.5,
        num_turns=2,
        multi_modal_data={"images": ["image://one"]},
        extra_fields={"finish_reason": "completed"},
    )

    record = debug_launcher.trajectory_to_record(
        session_id="debug-session",
        trajectory_index=7,
        trajectory=trajectory,
        metadata={"backend": "fake"},
        created_at="2026-06-24T00:00:00Z",
    )

    assert json.loads(json.dumps(record)) == record
    assert record == {
        "schema_version": 1,
        "session_id": "debug-session",
        "trajectory_index": 7,
        "created_at": "2026-06-24T00:00:00Z",
        "metadata": {"backend": "fake"},
        "trajectory": {
            "prompt_ids": [1, 2],
            "response_ids": [3, 4],
            "response_mask": [1, 0],
            "response_logprobs": [-0.1, 0.0],
            "reward_info": {"ok": True},
            "reward_score": 1.5,
            "num_turns": 2,
            "multi_modal_data": {"images": ["image://one"]},
            "extra_fields": {"finish_reason": "completed"},
        },
    }


def test_write_trajectories_jsonl_writes_expected_file(tmp_path):
    """Finalized trajectories are written under the session output directory as
    one JSON record per line with stable trajectory indexes."""
    trajectories = [
        Trajectory(prompt_ids=[1], response_ids=[2], response_mask=[1], num_turns=1),
        Trajectory(prompt_ids=[3], response_ids=[4], response_mask=[1], response_logprobs=[-0.2], num_turns=2),
    ]

    path = debug_launcher.write_trajectories_jsonl(
        output_dir=tmp_path,
        session_id="s1",
        trajectories=trajectories,
        metadata={"backend": "fake"},
        created_at="2026-06-24T01:02:03Z",
    )

    assert path == tmp_path / "s1" / "trajectories.jsonl"
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert records[0]["session_id"] == "s1"
    assert records[0]["trajectory_index"] == 0
    assert records[0]["trajectory"]["prompt_ids"] == [1]
    assert records[1]["trajectory_index"] == 1
    assert records[1]["trajectory"]["response_logprobs"] == [-0.2]


def test_write_session_metadata_json_writes_run_context_without_trajectories(tmp_path):
    """Session metadata is written even when no trajectories were captured so a
    failed smoke still leaves inspectable run context."""
    path = debug_launcher.write_session_metadata_json(
        output_dir=tmp_path,
        session_id="s-empty",
        metadata={
            "backend": "fake",
            "claude": {
                "enabled": True,
                "returncode": 1,
                "stdout_path": "/tmp/stdout",
                "stderr_path": "/tmp/stderr",
                "debug_log_path": "/tmp/debug",
            },
        },
        provider_urls={"anthropic_base_url": "http://127.0.0.1:9000/sessions/s-empty"},
        trajectories_count=0,
        trajectories_path=tmp_path / "s-empty" / "trajectories.jsonl",
        debug_snapshot_path=tmp_path / "s-empty" / "debug_snapshot.json",
        created_at="2026-06-24T01:02:03Z",
    )

    assert path == tmp_path / "s-empty" / "session_metadata.json"
    record = json.loads(path.read_text())
    assert record["schema_version"] == 1
    assert record["session_id"] == "s-empty"
    assert record["metadata"]["claude"]["returncode"] == 1
    assert record["trajectories_count"] == 0
    assert record["trajectories_path"].endswith("trajectories.jsonl")
    assert record["debug_snapshot_path"].endswith("debug_snapshot.json")


def test_write_debug_snapshot_json_writes_message_history_and_session_state(tmp_path):
    """Debug snapshots preserve normalized conversation state separately from
    token-level trajectories for post-run diagnosis."""
    path = debug_launcher.write_debug_snapshot_json(
        output_dir=tmp_path,
        session_id="s-debug",
        snapshot={
            "session_state": {"phase": "ACTIVE", "has_active_trajectory": True},
            "message_history": [{"role": "user", "content": "hi"}],
        },
        created_at="2026-06-24T01:02:03Z",
    )

    assert path == tmp_path / "s-debug" / "debug_snapshot.json"
    record = json.loads(path.read_text())
    assert record["schema_version"] == 1
    assert record["session_id"] == "s-debug"
    assert record["created_at"] == "2026-06-24T01:02:03Z"
    assert record["session_state"]["has_active_trajectory"] is True
    assert record["message_history"] == [{"role": "user", "content": "hi"}]


def test_capture_debug_snapshot_preserves_multiple_active_chains():
    chains = [
        SimpleNamespace(
            chain_id=1,
            message_history=[{"role": "user", "content": "main"}],
            active_tool_schemas=None,
        ),
        SimpleNamespace(
            chain_id=2,
            message_history=[{"role": "user", "content": "subagent"}],
            active_tool_schemas=[{"type": "function"}],
        ),
    ]
    session = SimpleNamespace(active_chains=chains, snapshot_state=lambda: {"num_active_chains": 2})
    actor = SimpleNamespace(_sessions={"s-debug": session})

    snapshot = debug_launcher.capture_debug_snapshot(actor, "s-debug", {"openai_base_url": "http://local"})

    assert [chain["chain_id"] for chain in snapshot["active_chains"]] == [1, 2]
    assert snapshot["active_chains"][1]["message_history"][0]["content"] == "subagent"
    assert snapshot["message_history"] == []
    assert snapshot["active_tool_schemas"] is None
    assert "multiple chains" in snapshot["note"]


def test_provider_urls_strips_v1_for_anthropic_and_keeps_openai_base_url():
    """The printed Anthropic base URL omits /v1 for Claude Code while the
    OpenAI-compatible URL keeps the session-scoped /v1 suffix."""
    handle = SessionHandle(
        session_id="abc",
        base_url="http://127.0.0.1:8000/sessions/abc/v1",
        reward_info_url="http://127.0.0.1:8000/sessions/abc/reward_info",
    )

    assert debug_launcher.provider_urls(handle) == {
        "anthropic_base_url": "http://127.0.0.1:8000/sessions/abc",
        "openai_base_url": "http://127.0.0.1:8000/sessions/abc/v1",
        "reward_info_url": "http://127.0.0.1:8000/sessions/abc/reward_info",
    }


def test_build_claude_env_removes_auth_and_proxy_vars_and_sets_expected_vars():
    """Claude Code smoke env construction removes conflicting auth/proxy
    variables and injects the local gateway Anthropic endpoint."""
    base_env = {
        "PATH": "/bin",
        "ANTHROPIC_AUTH_TOKEN": "real-token",
        "CLAUDE_CODE_API_BASE_URL": "https://real.example",
        "CLAUDE_CODE_API_KEY": "real-key",
        "HTTP_PROXY": "http://proxy.example",
        "custom_proxy_setting": "proxy",
        "ANTHROPIC_API_KEY": "old-key",
    }

    env = debug_launcher.build_claude_env(
        base_env,
        anthropic_base_url="http://10.1.2.3:9000/sessions/s1",
        anthropic_api_key="dummy-local-key",
    )

    assert env["PATH"] == "/bin"
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert "CLAUDE_CODE_API_BASE_URL" not in env
    assert "CLAUDE_CODE_API_KEY" not in env
    assert "HTTP_PROXY" not in env
    assert "custom_proxy_setting" not in env
    assert env["ANTHROPIC_BASE_URL"] == "http://10.1.2.3:9000/sessions/s1"
    assert env["ANTHROPIC_API_KEY"] == "dummy-local-key"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["NO_PROXY"] == "10.1.2.3,127.0.0.1,localhost"


def test_parse_args_minimal_and_openai_required_args_validation(tmp_path):
    """The fake backend has minimal CLI requirements, while openai-completions
    requires backend URL, backend model, and tokenizer path."""
    args = debug_launcher.parse_args(["--session-id", "s1", "--output-dir", str(tmp_path)])

    assert args.backend == "fake"
    assert args.session_id == "s1"
    assert args.output_dir == tmp_path
    assert args.anthropic_api_key == "dummy-local-key"
    assert args.claude_timeout == 300.0

    with pytest.raises(SystemExit):
        debug_launcher.parse_args(
            [
                "--backend",
                "openai-completions",
                "--session-id",
                "s1",
                "--output-dir",
                str(tmp_path),
            ]
        )

    args = debug_launcher.parse_args(
        [
            "--backend",
            "openai-completions",
            "--session-id",
            "s1",
            "--output-dir",
            str(tmp_path),
            "--backend-base-url",
            "http://127.0.0.1:8080",
            "--backend-model",
            "debug-model",
            "--tokenizer",
            "/tmp/tokenizer",
        ]
    )
    assert args.backend_base_url == "http://127.0.0.1:8080"
    assert args.backend_model == "debug-model"
    assert args.tokenizer == "/tmp/tokenizer"


def test_real_tokenizer_wrapper_flattens_encoding_template_results():
    """Real tokenizer wrappers flatten tokenizers Encoding objects returned by
    apply_chat_template while delegating other tokenizer methods."""

    class FakeEncoding:
        def __init__(self, ids):
            self.ids = ids

    class FakeTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [FakeEncoding([1, 2]), FakeEncoding([3])]

        def decode(self, token_ids, skip_special_tokens=True):
            return "decoded"

    wrapper = debug_launcher.TemplateResultTokenIdsWrapper(FakeTokenizer())

    assert wrapper.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=True) == [1, 2, 3]
    assert wrapper.decode([1, 2, 3]) == "decoded"


def test_build_claude_command(tmp_path):
    """One-shot Claude Code smoke uses the bare, non-persistent CLI mode with
    captured debug output, prompt text output, and the requested frontend model."""
    debug_file = tmp_path / "claude-debug.log"

    command = debug_launcher.build_claude_command(
        prompt="Reply with OK only.",
        model="claude-sonnet-4-5",
        debug_file=debug_file,
    )

    assert command[0] == "claude"
    assert "--bare" in command
    assert "--no-session-persistence" in command
    assert command[command.index("--debug-file") + 1] == str(debug_file)
    assert command[command.index("-p") + 1] == "Reply with OK only."
    assert command[command.index("--output-format") + 1] == "text"
    assert command[command.index("--model") + 1] == "claude-sonnet-4-5"


@pytest.mark.asyncio
async def test_openai_completions_backend_generate_sends_prompt_token_ids_and_parses_token_ids_logprobs(
    monkeypatch,
):
    """The OpenAI completions debug backend sends token-id prompts with
    return_token_ids enabled and maps token ids, logprobs, and finish reason."""
    requests = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {
                        "token_ids": [79, 75],
                        "logprobs": {"token_logprobs": [-0.3, -0.4]},
                        "finish_reason": "length",
                    }
                ]
            }

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json):
            requests.append({"url": url, "json": json, "timeout": self.timeout})
            return FakeResponse()

    monkeypatch.setattr(debug_launcher.httpx, "AsyncClient", FakeAsyncClient)
    backend = debug_launcher.OpenAICompletionsBackend(
        backend_base_url="http://backend.example/",
        backend_model="debug-model",
    )

    output = await backend.generate(
        request_id="s1",
        prompt_ids=[10, 11, 12],
        sampling_params={
            "max_tokens": 8,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "stop": ["END"],
            "ignored": True,
        },
    )

    assert requests == [
        {
            "url": "http://backend.example/completions",
            "json": {
                "model": "debug-model",
                "prompt": [10, 11, 12],
                "max_tokens": 8,
                "request_id": "s1",
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
                "stop": ["END"],
                "return_token_ids": True,
                "logprobs": 1,
                "add_special_tokens": False,
            },
            "timeout": 60.0,
        }
    ]
    assert output.token_ids == [79, 75]
    assert output.log_probs == [-0.3, -0.4]
    assert output.stop_reason == "length"


@pytest.mark.asyncio
async def test_openai_completions_backend_requires_token_ids(monkeypatch):
    """A completions backend response without token_ids is rejected because
    trajectory response tokens must come from backend token truth."""

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"text": "OK", "finish_reason": "stop"}]}

    class FakeAsyncClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json):
            return FakeResponse()

    monkeypatch.setattr(debug_launcher.httpx, "AsyncClient", FakeAsyncClient)
    backend = debug_launcher.OpenAICompletionsBackend(
        backend_base_url="http://backend.example",
        backend_model="debug-model",
    )

    with pytest.raises(RuntimeError, match="return_token_ids=true"):
        await backend.generate(request_id="s1", prompt_ids=[1], sampling_params={})


@pytest.mark.asyncio
async def test_openai_completions_backend_http_error_includes_request_id_and_body(monkeypatch):
    """HTTP errors from the completions backend include the gateway request id,
    status code, and response body in the raised diagnostic."""

    class FakeResponse:
        is_error = True
        status_code = 400
        text = "invalid token ids"

        def raise_for_status(self):
            pass

        def json(self):
            return {}

    class FakeAsyncClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json):
            return FakeResponse()

    monkeypatch.setattr(debug_launcher.httpx, "AsyncClient", FakeAsyncClient)
    backend = debug_launcher.OpenAICompletionsBackend(
        backend_base_url="http://backend.example",
        backend_model="debug-model",
    )

    with pytest.raises(RuntimeError, match="s1.*400.*invalid token ids"):
        await backend.generate(request_id="s1", prompt_ids=[1], sampling_params={})


@pytest.mark.asyncio
async def test_run_claude_once_records_timeout_metadata(monkeypatch, tmp_path):
    """A timed-out one-shot Claude subprocess records timeout metadata and
    preserves captured stdout/stderr files for diagnosis."""

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs["timeout"], output="partial stdout", stderr="slow")

    monkeypatch.setattr(debug_launcher.subprocess, "run", fake_run)

    metadata = await debug_launcher.run_claude_once(
        urls={"anthropic_base_url": "http://127.0.0.1:9000/sessions/s-timeout"},
        output_dir=tmp_path,
        session_id="s-timeout",
        claude_prompt="Reply with OK only.",
        claude_model="claude-sonnet-4-5",
        anthropic_api_key="dummy-local-key",
        claude_timeout=0.01,
    )

    assert metadata["returncode"] == 124
    assert metadata["timed_out"] is True
    assert metadata["timeout_seconds"] == 0.01
    assert Path(metadata["stdout_path"]).read_text() == "partial stdout"
    assert Path(metadata["stderr_path"]).read_text() == "slow"


@pytest.mark.asyncio
async def test_wait_for_manual_finalize_accepts_enter_from_tty_stdin():
    """Manual launcher mode can be finalized with Enter instead of requiring a
    signal-only shutdown path."""
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r", encoding="utf-8")

    class EnterStdin:
        def isatty(self):
            return True

        def fileno(self):
            return reader.fileno()

        def readline(self):
            return reader.readline()

    try:
        task = asyncio.create_task(debug_launcher.wait_for_manual_finalize(stdin=EnterStdin()))
        await asyncio.sleep(0)
        os.write(write_fd, b"\n")
        trigger = await asyncio.wait_for(task, timeout=1.0)
    finally:
        reader.close()
        os.close(write_fd)

    assert trigger == "stdin"


@pytest.mark.asyncio
async def test_run_debug_session_once_fake_creates_finalizes_and_writes_trajectories(
    monkeypatch,
    tmp_path,
):
    """The fake debug session starts the gateway, simulates Claude hitting the
    Anthropic route, finalizes the session, and writes trajectory artifacts."""
    post_urls = []
    captured_run = {}

    original_post = httpx.AsyncClient.post

    async def recording_post(self, url, *args, **kwargs):
        post_urls.append(url)
        return await original_post(self, url, *args, **kwargs)

    def fake_run(cmd, *, env, check, text, capture_output, timeout):
        captured_run["cmd"] = cmd
        captured_run["env"] = env
        captured_run["check"] = check
        captured_run["timeout"] = timeout
        probe_response = httpx.head(env["ANTHROPIC_BASE_URL"], timeout=5.0)
        assert probe_response.status_code == 404
        url = f"{env['ANTHROPIC_BASE_URL']}/v1/messages"
        post_urls.append(url)
        response = httpx.post(
            url,
            params={"beta": "true"},
            headers={"x-api-key": env["ANTHROPIC_API_KEY"]},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "Reply with OK only."}],
            },
            timeout=5.0,
        )
        response.raise_for_status()
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="OK\n", stderr="debug log\n")

    monkeypatch.setattr(httpx.AsyncClient, "post", recording_post)
    monkeypatch.setattr(debug_launcher.subprocess, "run", fake_run)

    result = await debug_launcher.run_debug_session_once(
        session_id="fake-session",
        output_dir=tmp_path,
        backend="fake",
        run_claude=True,
        claude_prompt="Reply with OK only.",
        claude_model="claude-sonnet-4-5",
        anthropic_api_key="dummy-local-key",
    )

    assert result.session_id == "fake-session"
    assert result.output_path == tmp_path / "fake-session" / "trajectories.jsonl"
    assert result.metadata_path == tmp_path / "fake-session" / "session_metadata.json"
    assert result.debug_snapshot_path == tmp_path / "fake-session" / "debug_snapshot.json"
    assert result.trajectories_count == 1
    assert captured_run["cmd"][0] == "claude"
    assert captured_run["check"] is False
    assert captured_run["timeout"] == 300.0
    assert captured_run["env"]["ANTHROPIC_BASE_URL"].endswith("/sessions/fake-session")
    assert captured_run["env"]["ANTHROPIC_API_KEY"] == "dummy-local-key"
    assert any(url.endswith("/sessions/fake-session/v1/messages") for url in post_urls)
    records = [json.loads(line) for line in result.output_path.read_text().splitlines()]
    session_metadata = json.loads(result.metadata_path.read_text())
    assert session_metadata["metadata"]["claude"]["returncode"] == 0
    assert session_metadata["trajectories_count"] == 1
    assert session_metadata["debug_snapshot_path"].endswith("debug_snapshot.json")
    assert records[0]["session_id"] == "fake-session"
    assert records[0]["metadata"]["backend"] == "fake"
    assert records[0]["metadata"]["claude"]["enabled"] is True
    assert records[0]["metadata"]["claude"]["returncode"] == 0
    stdout_path = Path(records[0]["metadata"]["claude"]["stdout_path"])
    stderr_path = Path(records[0]["metadata"]["claude"]["stderr_path"])
    assert stdout_path.read_text() == "OK\n"
    assert stderr_path.read_text() == "debug log\n"
    assert records[0]["metadata"]["claude"]["debug_log_path"].endswith("claude-debug.log")
    assert records[0]["trajectory"]["response_ids"] == [79, 75]
    debug_snapshot = json.loads(result.debug_snapshot_path.read_text())
    assert debug_snapshot["session_state"]["has_active_trajectory"] is True
    assert debug_snapshot["message_history"][0] == {"role": "user", "content": "Reply with OK only."}
    assert debug_snapshot["active_chains"][0]["message_history"] == debug_snapshot["message_history"]


@pytest.mark.asyncio
async def test_async_main_returns_nonzero_when_claude_fails(monkeypatch, tmp_path):
    """The CLI exits with Claude's non-zero return code when one-shot Claude
    execution fails after artifacts are written."""

    async def fake_run_debug_session_once(**kwargs):
        return debug_launcher.DebugSessionResult(
            session_id=kwargs["session_id"],
            output_path=tmp_path / kwargs["session_id"] / "trajectories.jsonl",
            metadata_path=tmp_path / kwargs["session_id"] / "session_metadata.json",
            debug_snapshot_path=tmp_path / kwargs["session_id"] / "debug_snapshot.json",
            trajectories_count=0,
            provider_urls={},
            claude_returncode=1,
        )

    monkeypatch.setattr(debug_launcher, "run_debug_session_once", fake_run_debug_session_once)

    exit_code = await debug_launcher.async_main(
        [
            "--session-id",
            "failed-claude",
            "--output-dir",
            str(tmp_path),
            "--run-claude",
        ]
    )

    assert exit_code == 1
