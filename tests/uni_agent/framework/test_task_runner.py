import pytest

from uni_agent.framework import task_runner as task_runner_module
from uni_agent.framework.task_runner import (
    _extract_upstream,
    _inject_gateway_tunnel,
    _inject_sandbox_route_key,
    _rewrite_gateway_url,
    compute_score,
    run_task,
    score_from_runner_result,
)
from uni_agent.gateway.session import SessionHandle
from uni_agent.sandbox import Sandbox
from uni_agent.sandbox.registry import SANDBOX_REGISTRY
from uni_agent.tasks import TaskConfig, TaskResult


@pytest.mark.cpu
@pytest.mark.level0
def test_rewrite_gateway_url_replaces_host_with_tunnel_port():
    assert _rewrite_gateway_url("http://gateway.example:40169/sessions/abc/v1", 38197) == (
        "http://127.0.0.1:38197/sessions/abc/v1"
    )


@pytest.mark.cpu
@pytest.mark.level0
def test_rewrite_gateway_url_custom_proxy_port():
    assert _rewrite_gateway_url("http://gateway:8000/v1", 4242) == "http://127.0.0.1:4242/v1"


@pytest.mark.cpu
@pytest.mark.level0
def test_extract_upstream_returns_host_port():
    assert _extract_upstream("http://gateway.example:40169/sessions/abc/v1") == "gateway.example:40169"


@pytest.mark.cpu
@pytest.mark.level0
def test_extract_upstream_none_without_port():
    assert _extract_upstream("http://gateway/v1") is None


@pytest.mark.cpu
@pytest.mark.level0
def test_inject_gateway_tunnel_rewrites_upstream_and_base_url():
    task = {
        "sandbox": {"provider": "openyuanrong", "sandbox_kwargs": {"proxy_port": 38197, "image": "x"}},
        "agent": {"step_limit": 10},
    }
    merged = _inject_gateway_tunnel(task, "http://gateway.example:40169/sessions/abc/v1")

    assert merged["sandbox"]["sandbox_kwargs"]["upstream"] == "gateway.example:40169"
    assert merged["sandbox"]["sandbox_kwargs"]["proxy_port"] == 38197
    # The agent receives the tunnel-rewritten base_url; unrelated keys are preserved.
    assert merged["agent"]["model"]["base_url"] == "http://127.0.0.1:38197/sessions/abc/v1"
    assert merged["agent"]["step_limit"] == 10


@pytest.mark.cpu
@pytest.mark.level1
@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://gateway.example:443/sessions/abc/v1", "https://gateway.example:443"),
        ("http://[::1]:8000/sessions/abc/v1", "[::1]:8000"),
        ("https://[2001:db8::1]:443/v1", "https://[2001:db8::1]:443"),
        ("https://gateway.example/v1", "https://gateway.example:443"),
        ("http://gateway.example/v1", "gateway.example:80"),
    ],
)
def test_agent_dx_tunnel_preserves_https_and_ipv6(base_url, expected):
    task = {"sandbox": {"provider": "agent_dx", "sandbox_kwargs": {"proxy_port": 38197}}}
    merged = _inject_gateway_tunnel(task, base_url)
    assert merged["sandbox"]["sandbox_kwargs"]["upstream"] == expected
    assert merged["agent"]["model"]["base_url"].startswith("http://127.0.0.1:38197/")
    assert "upstream" not in task["sandbox"]["sandbox_kwargs"]


@pytest.mark.cpu
@pytest.mark.level1
@pytest.mark.parametrize("port", [True, 0, -1, 1, 65536, "38197"])
def test_tunnel_rejects_invalid_proxy_port(port):
    task = {"sandbox": {"provider": "agent_dx", "sandbox_kwargs": {"proxy_port": port}}}
    with pytest.raises(ValueError, match="proxy_port"):
        _inject_gateway_tunnel(task, "http://gateway:8000/v1")


@pytest.mark.cpu
@pytest.mark.level1
@pytest.mark.parametrize("url", ["ftp://gateway:123/v1", "http:/v1", "https://user:secret@gateway/v1"])
def test_agent_dx_tunnel_rejects_unsupported_or_credentialed_urls(url):
    task = {"sandbox": {"provider": "agent_dx", "sandbox_kwargs": {"proxy_port": 38197}}}
    with pytest.raises(ValueError, match="HTTP\\(S\\) base_url without embedded credentials"):
        _inject_gateway_tunnel(task, url)


@pytest.mark.cpu
@pytest.mark.level1
def test_gateway_tunnel_uses_provider_owned_upstream_format(monkeypatch):
    class CustomSandbox(Sandbox):
        supports_reverse_tunnel = True

        @classmethod
        def tunnel_upstream(cls, base_url):
            assert base_url == "custom://gateway/sessions/abc/v1"
            return "https://provider-gateway:8443"

    monkeypatch.setitem(SANDBOX_REGISTRY, "custom_tunnel", CustomSandbox)
    task = {"sandbox": {"provider": "custom_tunnel", "sandbox_kwargs": {"proxy_port": 38197}}}
    merged = _inject_gateway_tunnel(task, "custom://gateway/sessions/abc/v1")
    assert merged["sandbox"]["sandbox_kwargs"]["upstream"] == "https://provider-gateway:8443"
    assert merged["agent"]["model"]["base_url"] == "http://127.0.0.1:38197/sessions/abc/v1"


@pytest.mark.cpu
@pytest.mark.level0
def test_inject_gateway_tunnel_raises_without_port():
    task = {"sandbox": {"provider": "openyuanrong", "sandbox_kwargs": {"proxy_port": 38197}}}
    with pytest.raises(ValueError, match="cannot derive gateway tunnel upstream"):
        _inject_gateway_tunnel(task, "http://gateway.example/v1")


@pytest.mark.cpu
@pytest.mark.level0
def test_inject_gateway_tunnel_rejects_non_yuanrong_sandbox():
    task = {"sandbox": {"provider": "local", "sandbox_kwargs": {"proxy_port": 38197}}}
    with pytest.raises(ValueError, match="supports_reverse_tunnel=True"):
        _inject_gateway_tunnel(task, "http://gateway.example:40169/v1")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_run_task_binds_raw_prompt_to_sample_task_config(monkeypatch, tmp_path):
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text(
        """
- name: test_task
""".strip()
    )
    captured = {}

    class _FakeTask:
        def __init__(self, config):
            self.config = TaskConfig(
                name=config["name"],
                sandbox={"provider": "local"},
                prompt=config["prompt"],
                metadata=config["metadata"],
            )

        async def run(self):
            captured["config"] = self.config
            return TaskResult(reward=1.0, accuracy=1.0, finished=True)

    monkeypatch.setattr(task_runner_module, "get_task", _FakeTask)
    source_prompt = [{"role": "user", "content": "Canonical source problem"}]

    await task_runner_module.run_task(
        session=SessionHandle(
            session_id="test-session",
            base_url="http://gateway/sessions/test/v1",
        ),
        raw_prompt=source_prompt,
        tools_kwargs={
            "task": {
                "name": "test_task",
                "metadata": {"problem_statement": "METADATA PROBLEM"},
            }
        },
        task_config_path=str(config_path),
    )

    assert captured["config"].prompt == source_prompt


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_run_task_returns_task_result_while_ignoring_framework_tool_config(monkeypatch):
    task_result = TaskResult(
        reward=0.5,
        accuracy=1.0,
        finished=False,
        extra_info={"report": {"resolved": 1}},
    )

    class _Resolver:
        def resolve(self, sample_config, runtime_model):
            assert sample_config["name"] == "stub"
            assert runtime_model["base_url"] == "http://gateway/session/v1"
            return {"name": "stub"}

    class _Task:
        async def run(self):
            return task_result

    monkeypatch.setattr(task_runner_module, "TaskConfigResolver", _Resolver)
    monkeypatch.setattr(task_runner_module, "get_task", lambda task: _Task())

    result = await run_task(
        session=SessionHandle(session_id="session", base_url="http://gateway/session/v1"),
        tools_kwargs={"task": {"name": "stub"}},
        tool_config=[object()],
    )

    assert result is task_result


# --------------------------- routes traffic-split key injection ---------------------------


def _routes_task(**sandbox_kwargs):
    return {"sandbox": {"provider": "routes", "sandbox_kwargs": dict(sandbox_kwargs)}}


@pytest.mark.cpu
@pytest.mark.level0
def test_inject_sandbox_route_key_uses_gateway_session_id():
    task = _routes_task(routes=[{"name": "a"}])
    injected = _inject_sandbox_route_key(task, "session-123")

    assert injected["sandbox"]["sandbox_kwargs"]["route_key"] == "session-123"
    # The other routes sandbox kwargs survive and the input is untouched.
    assert injected["sandbox"]["sandbox_kwargs"]["routes"] == [{"name": "a"}]
    assert "route_key" not in task["sandbox"]["sandbox_kwargs"]


@pytest.mark.cpu
@pytest.mark.level0
def test_inject_sandbox_route_key_keeps_explicit_route_key():
    task = _routes_task(route_key="pinned-by-config")
    assert _inject_sandbox_route_key(task, "session-123") is task


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("provider", ["local", "e2b", "agentenv"])
def test_inject_sandbox_route_key_skips_non_routes_providers(provider):
    task = {"sandbox": {"provider": provider, "sandbox_kwargs": {}}}
    assert _inject_sandbox_route_key(task, "session-123") is task


@pytest.mark.cpu
@pytest.mark.level0
def test_inject_sandbox_route_key_skips_empty_session_id():
    task = _routes_task()
    for session_id in (None, ""):
        assert _inject_sandbox_route_key(task, session_id) is task


@pytest.mark.cpu
@pytest.mark.level0
def test_inject_sandbox_route_key_does_not_mutate_input():
    task = _routes_task(routes=[{"name": "a"}], salt="rollout")
    snapshot = {"sandbox": {"provider": "routes", "sandbox_kwargs": {"routes": [{"name": "a"}], "salt": "rollout"}}}

    injected = _inject_sandbox_route_key(task, "session-123")

    assert task == snapshot
    assert injected is not task
    assert injected["sandbox"] is not task["sandbox"]
    assert injected["sandbox"]["sandbox_kwargs"] is not task["sandbox"]["sandbox_kwargs"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_run_task_injects_session_id_as_routes_route_key(monkeypatch):
    captured = {}

    class _Resolver:
        def resolve(self, sample_config, runtime_model):
            return {
                "name": "stub",
                "sandbox": {"provider": "routes", "sandbox_kwargs": {"routes": []}},
                "prompt": [],
            }

    class _Task:
        async def run(self):
            return TaskResult(reward=0.0, accuracy=1.0, finished=True)

    def _get_task(task):
        captured["task"] = task
        return _Task()

    monkeypatch.setattr(task_runner_module, "TaskConfigResolver", _Resolver)
    monkeypatch.setattr(task_runner_module, "get_task", _get_task)

    await run_task(
        session=SessionHandle(session_id="sess-42", base_url="http://gateway/sessions/sess-42/v1"),
        tools_kwargs={"task": {"name": "stub"}},
    )

    assert captured["task"]["sandbox"]["sandbox_kwargs"]["route_key"] == "sess-42"


@pytest.mark.cpu
@pytest.mark.level0
def test_score_from_runner_result_passes_through_runner_reward_info():
    assert compute_score is score_from_runner_result
    assert score_from_runner_result(
        data_source="stub",
        solution_str="unused",
        ground_truth="unused",
        extra_info={
            "runner_reward_info": {
                "reward": "0.5",
                "metrics": {"acc": True, "format": 0.8, "score": 0.75},
                "reward_context": {"trace": "unused by pass-through"},
            }
        },
        reward_router_address="http://reward-router",
        reward_model_tokenizer=object(),
    ) == {"score": 0.5, "acc": True, "format": 0.8}


@pytest.mark.cpu
@pytest.mark.level0
def test_agentenv_gateway_uses_direct_port():
    task = {"sandbox": {"provider": "agentenv", "sandbox_kwargs": {"proxy_port": 38197}}}
    result = _inject_gateway_tunnel(task, "https://gateway.example:40169/sessions/abc/v1")
    assert result["agent"]["model"]["base_url"] == "https://gateway.example:38197/sessions/abc/v1"
    assert result["sandbox"]["sandbox_kwargs"] == {"proxy_port": 38197}
