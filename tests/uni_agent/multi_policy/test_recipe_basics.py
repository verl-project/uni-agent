import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from examples.multi_policy.external_mas import execution as execution_module
from examples.multi_policy.external_mas.execution import (
    ExternalMASRequest,
    _apply_overrides,
    _parse_stdout_json,
    inject_gateway_config,
    run_external_mas,
)
from examples.multi_policy.external_mas.process_backend import (
    LocalProcessError,
    LocalProcessResult,
    LocalProcessTimeout,
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
@pytest.mark.asyncio
async def test_external_runner_applies_config_overrides(monkeypatch, tmp_path):
    template_path = tmp_path / "mas.yaml"
    template_path.write_text(
        "output:\n  prediction_path: shared.jsonl\n  include_trace: true\nllm: {}\nagents: {}\n",
        encoding="utf-8",
    )
    captured = {}

    async def fake_run(command, **kwargs):
        config_path = Path(command[command.index("--config") + 1])
        captured["template"] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return LocalProcessResult(0, "{}", "")

    monkeypatch.setattr(execution_module, "run_local_process", fake_run)

    await run_external_mas(
        request=ExternalMASRequest(
            prompt="question",
            role_urls={"planner": "http://gateway/planner/v1"},
            session_group_id="group-1",
        ),
        execution={"backend": "local_process"},
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

    assert captured["template"]["output"] == {
        "prediction_path": None,
        "report_path": None,
        "include_trace": False,
    }


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

    launcher = resolved["launcher_kwargs"]
    assert launcher["execution"]["backend"] == "local_process"
    assert launcher["command"]["argv"][-1] == "{prompt}"
    assert launcher["config"]["template_path"].endswith("/configs/inference.yaml")
    assert launcher["result"]["mode"] == "stdout_json"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_external_runner_writes_process_logs_on_success(monkeypatch, tmp_path):
    template_path = tmp_path / "mas.yaml"
    template_path.write_text("llm: {}\nagents: {}\n", encoding="utf-8")
    process = AsyncMock(return_value=LocalProcessResult(0, "stdout text", "stderr text"))
    monkeypatch.setattr(execution_module, "run_local_process", process)
    log_dir = tmp_path / "logs"

    result = await run_external_mas(
        request=ExternalMASRequest(
            prompt="question",
            role_urls={"planner": "http://gateway/planner/v1"},
            session_group_id="group-1",
        ),
        execution={"backend": "local_process"},
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
async def test_external_runner_writes_process_logs_on_nonzero_exit(monkeypatch, tmp_path):
    template_path = tmp_path / "mas.yaml"
    template_path.write_text("llm: {}\nagents: {}\n", encoding="utf-8")
    process = AsyncMock(side_effect=LocalProcessError(7, "partial stdout", "boom stderr"))
    monkeypatch.setattr(execution_module, "run_local_process", process)
    log_dir = tmp_path / "logs"

    with pytest.raises(LocalProcessError) as exc_info:
        await run_external_mas(
            request=ExternalMASRequest(
                prompt="question",
                role_urls={"planner": "http://gateway/planner/v1"},
                session_group_id="group-1",
            ),
            execution={"backend": "local_process"},
            command={"argv": ["fake-mas"], "work_dir": str(tmp_path)},
            config={"template_path": str(template_path)},
            result={"mode": "none"},
            logs={"directory": str(log_dir)},
        )

    assert exc_info.value.returncode == 7
    assert (log_dir / "group-1.stdout.log").read_text(encoding="utf-8") == "partial stdout"
    assert (log_dir / "group-1.stderr.log").read_text(encoding="utf-8") == "boom stderr"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_external_runner_writes_process_logs_on_timeout(monkeypatch, tmp_path):
    template_path = tmp_path / "mas.yaml"
    template_path.write_text("llm: {}\nagents: {}\n", encoding="utf-8")
    process = AsyncMock(
        side_effect=LocalProcessTimeout(
            "timeout",
            stdout="partial stdout",
            stderr="timeout stderr",
        )
    )
    monkeypatch.setattr(execution_module, "run_local_process", process)
    log_dir = tmp_path / "logs"

    with pytest.raises(LocalProcessTimeout):
        await run_external_mas(
            request=ExternalMASRequest(
                prompt="question",
                role_urls={"planner": "http://gateway/planner/v1"},
                session_group_id="group-1",
            ),
            execution={"backend": "local_process"},
            command={"argv": ["fake-mas"], "work_dir": str(tmp_path)},
            config={"template_path": str(template_path)},
            result={"mode": "none"},
            logs={"directory": str(log_dir)},
        )

    assert (log_dir / "group-1.stdout.log").read_text(encoding="utf-8") == "partial stdout"
    assert (log_dir / "group-1.stderr.log").read_text(encoding="utf-8") == "timeout stderr"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_external_runner_log_write_failure_does_not_replace_process_error(monkeypatch, tmp_path):
    template_path = tmp_path / "mas.yaml"
    template_path.write_text("llm: {}\nagents: {}\n", encoding="utf-8")
    process = AsyncMock(side_effect=LocalProcessError(9, "stdout", "stderr"))
    monkeypatch.setattr(execution_module, "run_local_process", process)
    blocked_log_dir = tmp_path / "blocked"
    blocked_log_dir.write_text("not a directory", encoding="utf-8")

    with pytest.raises(LocalProcessError) as exc_info:
        await run_external_mas(
            request=ExternalMASRequest(
                prompt="question",
                role_urls={"planner": "http://gateway/planner/v1"},
                session_group_id="group-1",
            ),
            execution={"backend": "local_process"},
            command={"argv": ["fake-mas"], "work_dir": str(tmp_path)},
            config={"template_path": str(template_path)},
            result={"mode": "none"},
            logs={"directory": str(blocked_log_dir)},
        )

    assert exc_info.value.returncode == 9


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_external_runner_does_not_expose_single_gateway_url_placeholder(monkeypatch, tmp_path):
    template_path = tmp_path / "mas.yaml"
    template_path.write_text("llm: {}\nagents: {}\n", encoding="utf-8")
    process = AsyncMock()
    monkeypatch.setattr(execution_module, "run_local_process", process)
    request = ExternalMASRequest(
        prompt="question",
        role_urls={"planner": "http://gateway/planner/v1"},
        session_group_id="group-1",
    )

    with pytest.raises(ValueError, match="gateway_url"):
        await run_external_mas(
            request=request,
            execution={"backend": "local_process"},
            command={"argv": ["mas", "--gateway", "{gateway_url}"]},
            config={"template_path": str(template_path)},
        )

    process.assert_not_awaited()
