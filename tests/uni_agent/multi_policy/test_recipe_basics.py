from unittest.mock import AsyncMock

import pytest

from examples.multi_policy.external_mas import external_mas_runner as runner_module
from examples.multi_policy.external_mas.external_mas_runner import (
    ExternalMASRequest,
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
def test_external_stdout_json_parser_accepts_log_prefix():
    assert _parse_stdout_json('log\n{"final_answer":"ok"}\n') == {"final_answer": "ok"}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_external_runner_does_not_expose_single_gateway_url_placeholder(monkeypatch, tmp_path):
    template_path = tmp_path / "mas.yaml"
    template_path.write_text("llm: {}\nagents: {}\n", encoding="utf-8")
    process = AsyncMock()
    monkeypatch.setattr(runner_module, "run_local_process", process)
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
