import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from examples.multi_policy.external_mas.execution import (
    ExternalMASError,
    ExternalMASRequest,
    ExternalMASTimeout,
    _apply_overrides,
    _parse_stdout_json,
    inject_gateway_config,
    run_external_mas,
)


@pytest.mark.cpu
@pytest.mark.level0
def test_external_config_injection_is_copy_on_write():
    config = {"llm": {"base_url": "old"}, "agents": {"a": {}}}
    role_urls = {"a": "http://gw/a/v1", "b": "http://gw/b/v1"}
    out = inject_gateway_config(config, role_urls=role_urls)
    # The global endpoint falls back to the first role's URL for MAS configs
    # that share one endpoint at the root.
    assert out["llm"]["base_url"] == "http://gw/a/v1"
    assert out["agents"]["a"]["llm"]["base_url"] == "http://gw/a/v1"
    assert out["agents"]["b"]["llm"]["base_url"] == "http://gw/b/v1"
    assert out["agents"]["a"]["model"] == "a"
    assert out["agents"]["b"]["model"] == "b"
    assert config["llm"]["base_url"] == "old"


@pytest.mark.cpu
@pytest.mark.level0
def test_external_config_injection_requires_roles():
    with pytest.raises(ValueError, match="role_urls"):
        inject_gateway_config({}, role_urls={})


@pytest.mark.cpu
@pytest.mark.level0
def test_external_config_overrides_apply_dotted_paths_without_mutating_template():
    template = {"output": {"prediction_path": "shared.jsonl", "include_trace": True}}
    resolved = _apply_overrides(
        template,
        {
            "output.prediction_path": None,
            "output.report_path": None,
            "output.include_trace": False,
        },
    )
    assert resolved["output"] == {
        "prediction_path": None,
        "report_path": None,
        "include_trace": False,
    }
    assert template["output"] == {"prediction_path": "shared.jsonl", "include_trace": True}


@pytest.mark.cpu
@pytest.mark.level0
def test_external_stdout_json_parser_accepts_log_prefix():
    assert _parse_stdout_json('log\n{"final_answer":"ok"}\n') == {"final_answer": "ok"}


@pytest.mark.cpu
@pytest.mark.level0
def test_external_stdout_json_parser_returns_last_top_level_object():
    stdout = '{"final_answer":"first"}\n{"final_answer":"second"}\n'
    assert _parse_stdout_json(stdout) == {"final_answer": "second"}


@pytest.mark.cpu
@pytest.mark.level0
def test_external_stdout_json_parser_skips_nested_trace_objects():
    payload = {
        "final_answer": "ok",
        "final_response": "<answer>ok</answer>",
        "trace": [
            {"output": "literal { brace"},
            {"output": '{"fake": 1}'},
            {"nested": {"deep": [{"value": 1}]}},
        ],
    }
    stdout = (
        "Verifier Agent: working\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\nsaved outputs: /tmp/search_predictions.jsonl\n"
    )

    assert _parse_stdout_json(stdout) == payload


@pytest.mark.cpu
@pytest.mark.level0
def test_external_stdout_json_parser_skips_consumed_ranges(monkeypatch):
    calls = []
    original = json.JSONDecoder.raw_decode

    def counting(self, value, idx=0):
        calls.append(idx)
        return original(self, value, idx)

    monkeypatch.setattr(json.JSONDecoder, "raw_decode", counting)
    payload = {
        "final_answer": "ok",
        "trace": [{"output": "{"} for _ in range(20)],
    }
    stdout = "log\n" + json.dumps(payload, indent=2) + "\nsaved outputs: x\n"

    assert _parse_stdout_json(stdout) == payload
    # json.loads may attempt one decode before the fallback; the outer result
    # itself must then be decoded once, without walking nested trace objects.
    assert len(calls) <= 2


@pytest.mark.cpu
@pytest.mark.level0
def test_external_mas_task_config_is_recipe_local():
    from uni_agent.tasks import TaskConfigResolver

    root = Path(__file__).resolve().parents[3]
    resolver = TaskConfigResolver.from_file(str(root / "examples/multi_policy/external_mas/task_config.yaml"))
    resolved = resolver.resolve(
        {
            "name": "multi_policy_external_mas",
            "prompt": [{"role": "user", "content": "question"}],
            "session_group_id": "group-1",
            "role_urls": {"verifier": "http://gateway/verifier/v1"},
        }
    )

    assert resolved["sandbox"]["provider"] == "local"
    launcher = resolved["launcher_kwargs"]
    assert launcher["command"]["argv"][-1] == "{prompt}"
    assert launcher["config"]["template_path"].endswith("/configs/inference.yaml")
    assert launcher["config"]["config_dir"] == "/tmp"
    assert launcher["result"]["mode"] == "stdout_json"


class _FakeSandbox:
    """Minimal sandbox: records the rendered config and returns a canned result."""

    def __init__(self, *, exit_code: int = 0, stdout: str = "{}", stderr: str = "") -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.written: dict[str, str] = {}
        self.execs: list[dict] = []
        self.started = 0
        self.stopped = 0

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    async def __aenter__(self) -> "_FakeSandbox":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    async def write_file(self, path: str, content: str) -> None:
        self.written[path] = content

    def _result(self) -> SimpleNamespace:
        return SimpleNamespace(exit_code=self.exit_code, stdout=self.stdout, stderr=self.stderr)

    async def exec(self, argv, *, timeout=None, workdir=None, env=None):
        self.execs.append({"argv": list(argv), "timeout": timeout, "workdir": workdir, "env": env})
        if list(argv)[:1] == ["rm"]:
            return SimpleNamespace(exit_code=0, stdout="", stderr="")
        return self._result()

    async def exec_shell(self, script, *, timeout=None, workdir=None, env=None):
        self.execs.append({"script": script, "timeout": timeout, "workdir": workdir, "env": env})
        return self._result()


def _request(session_group_id: str = "group-1") -> ExternalMASRequest:
    return ExternalMASRequest(
        prompt="question",
        role_urls={"planner": "http://gateway/planner/v1"},
        session_group_id=session_group_id,
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_external_runner_applies_config_overrides(tmp_path):
    template_path = tmp_path / "mas.yaml"
    template_path.write_text(
        "output:\n  prediction_path: shared.jsonl\n  include_trace: true\nllm: {}\nagents: {}\n",
        encoding="utf-8",
    )
    sandbox = _FakeSandbox()

    await run_external_mas(
        request=_request(),
        sandbox=sandbox,
        command={"argv": ["fake-mas", "--config", "{config_path}"]},
        config={
            "template_path": str(template_path),
            "overrides": {
                "output.prediction_path": None,
                "output.report_path": None,
                "output.include_trace": False,
            },
        },
        result={"mode": "none"},
    )

    ((config_path, payload),) = sandbox.written.items()
    template = yaml.safe_load(payload)
    assert template["output"] == {
        "prediction_path": None,
        "report_path": None,
        "include_trace": False,
    }
    run = next(entry for entry in sandbox.execs if entry["argv"][:1] != ["rm"])
    assert run["argv"][2] == config_path


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_external_runner_writes_process_logs_on_success(tmp_path):
    template_path = tmp_path / "mas.yaml"
    template_path.write_text("llm: {}\nagents: {}\n", encoding="utf-8")
    sandbox = _FakeSandbox(stdout="stdout text", stderr="stderr text")
    log_dir = tmp_path / "logs"

    result = await run_external_mas(
        request=_request(),
        sandbox=sandbox,
        command={"argv": ["fake-mas"], "work_dir": str(tmp_path)},
        config={"template_path": str(template_path)},
        result={"mode": "none"},
        logs={"directory": str(log_dir)},
    )

    assert result == {}
    assert (log_dir / "group-1.stdout.log").read_text(encoding="utf-8") == "stdout text"
    assert (log_dir / "group-1.stderr.log").read_text(encoding="utf-8") == "stderr text"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_external_runner_writes_process_logs_on_nonzero_exit(tmp_path):
    template_path = tmp_path / "mas.yaml"
    template_path.write_text("llm: {}\nagents: {}\n", encoding="utf-8")
    sandbox = _FakeSandbox(exit_code=7, stdout="partial stdout", stderr="boom stderr")
    log_dir = tmp_path / "logs"

    with pytest.raises(ExternalMASError) as exc_info:
        await run_external_mas(
            request=_request(),
            sandbox=sandbox,
            command={"argv": ["fake-mas"], "work_dir": str(tmp_path)},
            config={"template_path": str(template_path)},
            result={"mode": "none"},
            logs={"directory": str(log_dir)},
        )

    assert exc_info.value.exit_code == 7
    assert (log_dir / "group-1.stdout.log").read_text(encoding="utf-8") == "partial stdout"
    assert (log_dir / "group-1.stderr.log").read_text(encoding="utf-8") == "boom stderr"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_external_runner_reports_timeout_without_partial_output(tmp_path):
    template_path = tmp_path / "mas.yaml"
    template_path.write_text("llm: {}\nagents: {}\n", encoding="utf-8")
    # The base sandbox contract reports a timeout as exit code -1 and drops the
    # output captured so far, so the logs can only hold the synthetic message.
    sandbox = _FakeSandbox(exit_code=-1, stdout="", stderr="local exec timed out after 5s")
    log_dir = tmp_path / "logs"

    with pytest.raises(ExternalMASTimeout):
        await run_external_mas(
            request=_request(),
            sandbox=sandbox,
            command={"argv": ["fake-mas"], "work_dir": str(tmp_path)},
            config={"template_path": str(template_path)},
            timeout={"seconds": 5},
            result={"mode": "none"},
            logs={"directory": str(log_dir)},
        )

    assert (log_dir / "group-1.stdout.log").read_text(encoding="utf-8") == ""
    assert "timed out" in (log_dir / "group-1.stderr.log").read_text(encoding="utf-8")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_external_runner_log_write_failure_does_not_replace_process_error(tmp_path):
    template_path = tmp_path / "mas.yaml"
    template_path.write_text("llm: {}\nagents: {}\n", encoding="utf-8")
    sandbox = _FakeSandbox(exit_code=9, stdout="stdout", stderr="stderr")
    blocked_log_dir = tmp_path / "blocked"
    blocked_log_dir.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ExternalMASError) as exc_info:
        await run_external_mas(
            request=_request(),
            sandbox=sandbox,
            command={"argv": ["fake-mas"], "work_dir": str(tmp_path)},
            config={"template_path": str(template_path)},
            result={"mode": "none"},
            logs={"directory": str(blocked_log_dir)},
        )

    assert exc_info.value.exit_code == 9


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_external_runner_does_not_expose_single_gateway_url_placeholder(tmp_path):
    template_path = tmp_path / "mas.yaml"
    template_path.write_text("llm: {}\nagents: {}\n", encoding="utf-8")
    sandbox = _FakeSandbox()

    with pytest.raises(ValueError, match="gateway_url"):
        await run_external_mas(
            request=_request(),
            sandbox=sandbox,
            command={"argv": ["mas", "--gateway", "{gateway_url}"]},
            config={"template_path": str(template_path)},
        )
